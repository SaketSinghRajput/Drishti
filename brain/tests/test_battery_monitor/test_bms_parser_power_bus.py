import pytest
from battery_monitor.ingestion.bms_parser import BMSCANParser


def test_normalize_with_power_bus_data():
    parser = BMSCANParser()
    raw_can = {'pack_voltage_mv': 4800, 'pack_current_ma': 3000, 'soc_promille': 800, 'soh_promille': 950}
    power_bus = {'pack_voltage_v': 4.2, 'pack_current_a': 3.0, 'max_temp_c': 30.0, 'power_w': 12.6}
    out = parser.normalize_telemetry(raw_can, power_bus_data=power_bus)
    assert out.get('pack_voltage') == 4.2
    assert out.get('pack_current') == 3.0
    assert out.get('pack_temp_c') == 30.0
    assert out.get('power_w') == 12.6


def test_normalize_without_power_bus():
    parser = BMSCANParser()
    raw_can = {'pack_voltage_mv': 4800, 'pack_current_ma': 3000, 'soc_promille': 800, 'soh_promille': 950}
    out = parser.normalize_telemetry(raw_can)
    # CAN-derived value should be used (4800 mV -> 4.8 V)
    assert out.get('pack_voltage') == pytest.approx(4.8)
    # pack_temp_c and power_w should not be present when no power_bus_data provided
    assert 'pack_temp_c' not in out
    assert 'power_w' not in out
