import pytest

from deployment.hardware_diagnostics.checker import HardwareChecker

class DummyCfg:
    def __init__(self):
        self._d = {
            'sensors.voltage_current.voltage_range': [3.0, 5.0],
            'sensors.voltage_current.current_range': [-10.0, 10.0],
            'sensors.voltage_current.temp_range': [-20.0, 70.0],
        }
    def get(self, key, default=None):
        return self._d.get(key, default)


class DummyVCH:
    def __init__(self, sample):
        self._sample = sample
    def read(self):
        return self._sample


def test_voltage_sensor_ok(monkeypatch):
    cfg = DummyCfg()
    hc = HardwareChecker(cfg)
    # monkeypatch VoltageCurrentHandler import to return sample
    sample = {'pack_voltage_v': 4.2}
    # Insert a fake module into sys.modules so the runtime import inside checker will pick it up
    import types, sys
    mod = types.ModuleType('data_collection.sensors.voltage_current_handler')
    mod.VoltageCurrentHandler = lambda cfg: DummyVCH(sample)
    sys.modules['data_collection.sensors.voltage_current_handler'] = mod
    res = hc._test_voltage_sensor()
    assert res['status'] == 'OK'
    assert res['details']['voltage'] == 4.2


def test_voltage_sensor_out_of_range(monkeypatch):
    cfg = DummyCfg()
    hc = HardwareChecker(cfg)
    sample = {'pack_voltage_v': 10.0}
    import types, sys
    mod = types.ModuleType('data_collection.sensors.voltage_current_handler')
    mod.VoltageCurrentHandler = lambda cfg: DummyVCH(sample)
    sys.modules['data_collection.sensors.voltage_current_handler'] = mod
    res = hc._test_voltage_sensor()
    assert res['status'] == 'FAILED'
    assert res['reason'] == 'out_of_range'


def test_voltage_sensor_no_sample(monkeypatch):
    cfg = DummyCfg()
    hc = HardwareChecker(cfg)
    class BadVCH:
        def read(self):
            return None
    import types, sys
    mod = types.ModuleType('data_collection.sensors.voltage_current_handler')
    mod.VoltageCurrentHandler = lambda cfg: BadVCH()
    sys.modules['data_collection.sensors.voltage_current_handler'] = mod
    res = hc._test_voltage_sensor()
    assert res['status'] == 'FAILED'
    assert res['reason'] == 'no_sample'


def test_current_and_temp_checks(monkeypatch):
    cfg = DummyCfg()
    hc = HardwareChecker(cfg)
    sample = {'pack_voltage_v': 4.0, 'pack_current_a': 12.0, 'max_temp_c': 80.0}
    import types, sys
    mod = types.ModuleType('data_collection.sensors.voltage_current_handler')
    mod.VoltageCurrentHandler = lambda cfg: DummyVCH(sample)
    sys.modules['data_collection.sensors.voltage_current_handler'] = mod
    res_c = hc._test_current_sensor()
    res_t = hc._test_temp_sensor()
    assert res_c['status'] == 'FAILED'
    assert res_c['reason'] == 'out_of_range'
    assert res_t['status'] == 'FAILED'
    assert res_t['reason'] == 'out_of_range'
