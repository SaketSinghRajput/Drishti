import struct

from battery_monitor.ingestion.bms_parser import BMSCANParser


def test_parse_pack_frame():
    p = BMSCANParser()
    # create fake pack frame: v_mv=48120, i_ma=-3200, soc=725, soh=991
    data = struct.pack('<HhHH', 48120, -3200, 725, 991)
    out = p.parse_message(0x100, data)
    assert 'pack_voltage_mv' in out
    assert out['pack_voltage_mv'] == 48120


def test_cell_voltage_aggregation():
    p = BMSCANParser()
    # two frames with two voltages each (in mV)
    data1 = struct.pack('<HH', 3700, 3705)
    data2 = struct.pack('<HH', 3695, 3710)
    p.parse_message(0x200, data1)
    p.parse_message(0x200, data2)
    norm = p.normalize_telemetry({'timestamp': '2025-01-01T00:00:00Z'})
    assert 'cell_voltages' in norm
    assert len(norm['cell_voltages']) >= 4
