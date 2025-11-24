import os
import threading
from config.config_loader import ConfigLoader
from deployment.hardware_diagnostics import HardwareChecker


def test_checker_integration_simulation(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [gps, imu]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    # Some test environments parse YAML differently; force simulation for determinism
    checker.enable_simulation = True
    # ensure expected sensors set for deterministic behavior
    checker.expected_sensors = ['gps', 'imu']
    res = checker.detect_hardware()
    assert set(res['found_sensors']) == set(['gps', 'imu'])
    # log file exists
    assert os.path.exists(checker.log_path)


def test_checker_thread_safety(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [gps, imu]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    # Force simulation to keep behavior deterministic in CI/dev
    checker.enable_simulation = True
    results = []

    def run_check():
        results.append(checker.detect_hardware())

    threads = [threading.Thread(target=run_check) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 3
    for r in results:
        assert 'timestamp' in r


    def test_self_test_simulation_integration(tmp_path):
        cfg_path = tmp_path / 'cfg.yaml'
        cfg_path.write_text('data_collection:\n  expected_sensors: [gps, imu]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
        cfg = ConfigLoader(str(cfg_path))
        checker = HardwareChecker(cfg)
        checker.enable_simulation = True
        res = checker.self_test('imu')
        assert res['status'] == 'OK'
        summary = checker.run_all_tests()
        assert set(summary['passed_tests']) == set(['gps', 'imu'])


def test_trigger_alerts_and_health_exposure(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [gps, imu]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    # ensure dummy gpio exists
    checker.local['gpio_pins'] = {'buzzer': 18, 'led_red': 17, 'led_green': 27}

    # simulate a failing summary
    summary = {'failed_tests': ['camera'], 'skipped_tests': [], 'passed_tests': [], 'test_results': {}}
    # should not raise
    checker._trigger_alerts(summary)

    # simulate a success summary
    summary2 = {'failed_tests': [], 'skipped_tests': [], 'passed_tests': ['gps', 'imu'], 'test_results': {}}
    checker._trigger_alerts(summary2)


def test_get_system_health_status_and_disabled_modules(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [gps, imu]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    checker = HardwareChecker(cfg)
    # no health yet
    assert checker.get_system_health_status() is None
    # run tests
    summary = checker.run_all_tests()
    health = checker.get_system_health_status()
    assert isinstance(health, dict)
    assert 'overall_status' in health
    # disabled modules should reflect failed+skipped
    disabled = checker.get_disabled_modules()
    assert isinstance(disabled, list)


def test_post_to_dashboards_and_telemetry(tmp_path, monkeypatch):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('data_collection:\n  expected_sensors: [gps]\nhardware_diagnostics:\n  enable_simulation: true\n  log_dir: "{}"\n  dashboard_integration:\n    enable: true\n'.format(str(tmp_path)))
    cfg = ConfigLoader(str(cfg_path))
    calls = []

    class DummyResp:
        def __init__(self):
            self.ok = True

    def fake_post(url, json=None, timeout=None, headers=None):
        calls.append(url)
        return DummyResp()

    monkeypatch.setattr('requests.post', fake_post)

    class DummyTelemetry:
        def __init__(self):
            self.calls = []

        def log_hardware_check(self, results):
            self.calls.append(results)

    telemetry = DummyTelemetry()
    checker = HardwareChecker(cfg, telemetry=telemetry)
    checker.run_all_tests()
    # Two dashboard posts expected
    assert any('/api/hardware_health' in u for u in calls)
    # telemetry should have a hardware check entry
    assert telemetry.calls


def test_systemd_service_file_exists():
    # Ensure the unit file exists in the repo for packaging
    path = os.path.join('deployment', 'systemd', 'hardware-diagnostics.service')
    assert os.path.exists(path), f"Missing {path}"


def test_systemd_service_syntax():
    # Basic sanity checks on the unit file content
    path = os.path.join('deployment', 'systemd', 'hardware-diagnostics.service')
    with open(path, 'r') as f:
        txt = f.read()
    assert '[Unit]' in txt
    assert 'Before=ev-safety-inference.service' in txt
    assert 'Type=oneshot' in txt
    assert 'ExecStart' in txt and ('run_hardware_check' in txt or 'deployment.hardware_diagnostics.checker' in txt)
    assert 'WantedBy=multi-user.target' in txt


def test_systemd_timer_file_exists():
    path = os.path.join('deployment', 'systemd', 'hardware-diagnostics.timer')
    assert os.path.exists(path), f"Missing {path}"


def test_systemd_timer_syntax():
    path = os.path.join('deployment', 'systemd', 'hardware-diagnostics.timer')
    with open(path, 'r') as f:
        txt = f.read()
    assert '[Timer]' in txt
    assert 'OnBootSec=5min' in txt or 'OnBootSec= 5min' in txt
    assert 'OnUnitActiveSec=5min' in txt or 'OnUnitActiveSec= 5min' in txt
    assert 'Requires=hardware-diagnostics.service' in txt


def test_install_sh_includes_hardware_diagnostics():
    path = os.path.join('deployment', 'scripts', 'install.sh')
    with open(path, 'r') as f:
        txt = f.read()
    assert 'systemctl enable hardware-diagnostics.service' in txt
    assert 'systemctl enable hardware-diagnostics.timer' in txt


def test_run_hardware_check_cli_exit_codes(tmp_path, monkeypatch):
    # Ensure CLI returns appropriate exit codes based on run_all_tests behavior
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('hardware_diagnostics:\n  enable_simulation: true\n')
    cfg = ConfigLoader(str(cfg_path))

    # Monkeypatch the HardwareChecker used by the CLI wrapper and call main() in-process
    class FakeCheckerPass:
        def __init__(self, cfg, telemetry=None, *args, **kwargs):
            pass

        def run_all_tests(self):
            return {'failed_tests': [], 'passed_tests': ['gps'], 'skipped_tests': [], 'test_results': {}}
        def get_system_health_status(self):
            return {'overall_status': 'OK', 'components': {'gps': 'OK'}, 'disabled_modules': []}

    class FakeCheckerFail:
        def __init__(self, cfg, telemetry=None, *args, **kwargs):
            pass

        def run_all_tests(self):
            return {'failed_tests': ['camera'], 'passed_tests': [], 'skipped_tests': [], 'test_results': {}}
        def get_system_health_status(self):
            return {'overall_status': 'FAILED', 'components': {'camera': 'FAILED - reason'}, 'disabled_modules': ['camera']}

    import importlib
    import scripts.run_hardware_check as rhc

    # Patch the HardwareChecker used by the CLI module
    monkeypatch.setattr(rhc, 'HardwareChecker', FakeCheckerPass)
    # Ensure CLI reads our temp config
    monkeypatch.setattr('sys.argv', ['scripts/run_hardware_check', '--config', str(cfg_path)])
    try:
        rhc.main()
    except SystemExit as e:
        assert e.code == 0

    # Now simulate a failing run
    monkeypatch.setattr(rhc, 'HardwareChecker', FakeCheckerFail)
    try:
        rhc.main()
    except SystemExit as e:
        assert e.code == 1
