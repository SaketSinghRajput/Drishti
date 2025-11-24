import json
import pytest
from pathlib import Path
from data_collection.sensors.voltage_current_handler import VoltageCurrentHandler

class DummyConfig:
    def __init__(self, d):
        self._d = d
    def get(self, key, default=None):
        parts = key.split('.')
        cur = self._d
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur


def test_simulation_read_and_out_of_range(tmp_path):
    csv = tmp_path / 'vc.csv'
    csv.write_text('timestamp,voltage_v,current_a,temp_c\n2025-01-01T00:00:00Z,4.2,2.5,25.0\n2025-01-01T00:00:01Z,10.0,2.5,25.0\n')
    cfg = DummyConfig({'sensors': {'voltage_current': {'enable': True, 'enable_simulation': True, 'csv_simulator_path': str(csv)}}})
    h = VoltageCurrentHandler(cfg)
    assert h.is_available()
    s1 = h.read()
    assert s1 is not None
    assert 'pack_voltage_v' in s1
    # second row has out-of-range voltage but read should return it
    s2 = h.read()
    assert s2 is not None
    assert s2['pack_voltage_v'] == 10.0
    # health should reflect out-of-range event
    health = h.get_health()
    assert health['error_count'] >= 1
    assert health['status'] in ('warning', 'error')
    assert 'last_out_of_range' in health


def test_disabled_health():
    cfg = DummyConfig({'sensors': {'voltage_current': {'enable': False}}})
    h = VoltageCurrentHandler(cfg)
    assert h.is_available() is False
    health = h.get_health()
    assert health['status'] in ('disabled', 'unknown')


def test_missing_csv_file_returns_none_and_marks_error(tmp_path):
    # path does not exist
    cfg = DummyConfig({'sensors': {'voltage_current': {'enable': True, 'enable_simulation': True, 'csv_simulator_path': str(tmp_path / 'missing.csv')}}})
    h = VoltageCurrentHandler(cfg)
    # is_available should be False because file missing
    assert h.is_available() is False
    s = h.read()
    # read may return None and increment error_count
    assert s is None
    health = h.get_health()
    assert health['error_count'] >= 1


def test_production_i2c_read_monkeypatched(monkeypatch):
    # simulate circuit environment and I2C devices
    from data_collection.sensors import voltage_current_handler as vmod

    class DummyCFG:
        def get(self, key, default=None):
            mapping = {'sensors.voltage_current.enable': True, 'sensors.voltage_current.enable_simulation': False}
            return mapping.get(key, default)

    # force module to think CircuitPython is available
    monkeypatch.setattr(vmod, '_has_circuit', True)

    # dummy INA219-like object
    class DummyINA:
        def __init__(self):
            self.bus_voltage = 4.2
            self.current = 2.5

    # dummy AnalogIn factory
    def _dummy_analogin(ads, ch):
        class C:
            voltage = 0.25
        return C()

    monkeypatch.setattr(vmod, 'AnalogIn', _dummy_analogin)

    cfg = DummyCFG()
    h = VoltageCurrentHandler(cfg)
    # inject dummy devices
    h._ina219 = DummyINA()
    h._ads = object()
    res = h.read()
    assert isinstance(res, dict)
    assert res.get('pack_voltage_v') == pytest.approx(4.2)
    assert res.get('pack_current_a') == pytest.approx(2.5)
    assert res.get('max_temp_c') == pytest.approx(25.0)
