import os
import json
import subprocess
from unittest.mock import MagicMock

import pytest

from deployment.hardware_diagnostics.checker import HardwareChecker
from config.config_loader import ConfigLoader


class DummyCompleted:
    def __init__(self, out=''):
        self.stdout = out
        self.stderr = ''


@pytest.fixture
def small_config(tmp_path):
    # create a minimal default_config.yaml-like object via ConfigLoader path override
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [imu, gps, camera]\nhardware_diagnostics:\n  enable_simulation: false\n  log_dir: "{}"\n'.format(str(tmp_path)))
    return ConfigLoader(str(cfg_path))


def mock_run_factory(mapping):
    def _mock_run(cmd, capture_output=True, text=True, timeout=5):
        key = ' '.join(cmd)
        if key in mapping:
            return DummyCompleted(mapping[key])
        # default empty
        return DummyCompleted('')

    return _mock_run


def test_simulation_mode(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [imu, gps, camera]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    res = checker.detect_hardware()
    assert set(res['found_sensors']) == set(['imu', 'gps', 'camera'])
    assert all(res['details'][s]['details'] == 'Simulated' for s in res['found_sensors'])


def test_detect_all_found(monkeypatch, small_config, tmp_path):
    mapping = {
        'i2cdetect -y 1': '     0x68',
        'v4l2-ctl --list-devices': '/dev/video0',
        'bluetoothctl list': 'Controller hci0',
        'ip link show can0': 'can0: <BROADCAST,NOARP,UP>'
    }
    monkeypatch.setattr('subprocess.run', mock_run_factory(mapping))
    checker = HardwareChecker(small_config)
    res = checker.detect_hardware()
    # At least IMU should be reported if i2c mapping contains 0x68 in local config
    # local mapping in package maps 0x68 -> imu
    assert 'imu' in res['found_sensors'] or 'gps' in res['found_sensors']


def test_detect_all_missing(monkeypatch, small_config, tmp_path):
    # no commands return anything
    monkeypatch.setattr('subprocess.run', mock_run_factory({}))
    checker = HardwareChecker(small_config)
    res = checker.detect_hardware()
    assert set(res['missing_sensors']) == set(['imu', 'gps', 'camera'])


def test_timeout_handling(monkeypatch, small_config):
    def raise_timeout(cmd, capture_output=True, text=True, timeout=5):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr('subprocess.run', raise_timeout)
    checker = HardwareChecker(small_config)
    res = checker.detect_hardware()
    assert 'imu' in res['missing_sensors']


def test_missing_tool(monkeypatch, small_config):
    def raise_not_found(cmd, capture_output=True, text=True, timeout=5):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr('subprocess.run', raise_not_found)
    checker = HardwareChecker(small_config)
    res = checker.detect_hardware()
    assert 'imu' in res['missing_sensors']


def test_unexpected_devices(monkeypatch, small_config):
    mapping = {
        'i2cdetect -y 1': ' 0x99'
    }
    monkeypatch.setattr('subprocess.run', mock_run_factory(mapping))
    checker = HardwareChecker(small_config)
    res = checker.detect_hardware()
    assert isinstance(res['unexpected_devices'], list)


def test_log_file_creation(monkeypatch, small_config, tmp_path):
    mapping = {
        'i2cdetect -y 1': '     0x68',
    }
    monkeypatch.setattr('subprocess.run', mock_run_factory(mapping))
    cfg = small_config
    checker = HardwareChecker(cfg)
    res = checker.detect_hardware()
    log_files = list(tmp_path.glob('*'))
    # summary json should be created under log_dir
    assert any('hardware_summary' in name for name in os.listdir(checker.log_dir))


def test_can_satisfies_bms(monkeypatch, tmp_path):
    # Create a config that expects 'bms' and uses the package local can_interfaces (can0)
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [bms]\nhardware_diagnostics:\n  enable_simulation: false\n  log_dir: "{}"\n'.format(str(tmp_path)))
    mapping = {
        'ip link show can0': 'can0: <BROADCAST,NOARP,UP>'
    }
    monkeypatch.setattr('subprocess.run', mock_run_factory(mapping))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    res = checker.detect_hardware()
    assert 'bms' in res['found_sensors']


def test_self_test_timeout(monkeypatch, small_config):
    # simulate timeout by having _run_with_timeout raise TimeoutError
    cfg = small_config
    checker = HardwareChecker(cfg)

    def raise_timeout(fn, timeout_s):
        raise TimeoutError('function timed out after {}s'.format(timeout_s))

    monkeypatch.setattr(checker, '_run_with_timeout', raise_timeout)
    res = checker.self_test('imu')
    assert res['status'] == 'FAILED'
    assert 'timeout' in (res.get('reason') or '').lower()


def test_imu_validation_failure(monkeypatch, small_config):
    # Provide a dummy IMUHandler that returns an invalid az (0)
    import types, sys

    mod = types.ModuleType('data_collection.sensors.imu_handler')

    class DummyIMU:
        def __init__(self, cfg):
            pass

        def read(self):
            return {'az': 0}

    mod.IMUHandler = DummyIMU
    monkeypatch.setitem(sys.modules, 'data_collection.sensors.imu_handler', mod)

    checker = HardwareChecker(small_config)
    res = checker.self_test('imu')
    assert res['status'] == 'FAILED'
    assert res.get('reason') == 'gravity_check_failed'


def test_buzzer_gpio_calls(monkeypatch, small_config):
    import types, sys
    # create a fake RPi.GPIO module
    gpio_mod = types.ModuleType('RPi.GPIO')
    calls = {'setmode': False, 'setup': False, 'output': []}

    def setmode(mode):
        calls['setmode'] = True

    def setup(pin, mode):
        calls['setup'] = True

    def output(pin, state):
        calls['output'].append((pin, state))

    gpio_mod.setmode = setmode
    gpio_mod.setup = setup
    gpio_mod.output = output
    gpio_mod.BCM = 11
    gpio_mod.OUT = 22

    monkeypatch.setitem(sys.modules, 'RPi.GPIO', gpio_mod)

    # Ensure the checker knows about the buzzer pin
    cfg = small_config
    checker = HardwareChecker(cfg)
    checker.local['gpio_pins'] = {'buzzer': 18}
    res = checker._test_buzzer()
    assert res['status'] == 'OK'
    assert calls['setmode'] is True
    assert calls['setup'] is True
    assert any(p == 18 for p, _ in calls['output'])


def test_ble_scan_success(monkeypatch, small_config):
    cfg = small_config
    checker = HardwareChecker(cfg)
    # simulate bluetoothctl scan
    def fake_run(cmd, capture_output=True, text=True, timeout=5):
        class C:
            stdout = 'Discovery started\nDevice 00:11:22:33'
            stderr = ''

        return C()

    monkeypatch.setattr('subprocess.run', fake_run)
    res = checker._test_ble_mesh()
    assert res['status'] == 'OK'


def test_esp32_heartbeat(monkeypatch, small_config):
    import types, sys

    # fake serial module
    class FakeSerial:
        def __init__(self, port, timeout=2):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def write(self, b):
            pass

        def read(self, n):
            return b'OK'

    serial_mod = types.ModuleType('serial')
    serial_mod.Serial = FakeSerial
    monkeypatch.setitem(sys.modules, 'serial', serial_mod)

    checker = HardwareChecker(small_config)
    checker.esp32_uart = '/dev/ttyUSB2'
    res = checker._test_esp32()
    assert res['status'] == 'OK'


def test_self_test_simulation_mode(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [imu, gps]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    # simulated self_test should return OK
    res = checker.self_test('imu')
    assert res['status'] == 'OK'


def test_run_all_tests_simulation(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [imu, gps]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    summary = checker.run_all_tests()
    assert set(summary['passed_tests']) == set(['imu', 'gps'])


def test_self_test_unknown_component(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [foo]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    res = checker.self_test('unknown_sensor')
    assert res['status'] == 'FAILED'
    assert res['reason'] == 'unknown_component'
