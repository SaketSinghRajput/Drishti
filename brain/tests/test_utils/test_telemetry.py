import os
import json
import tempfile

import pytest

from config.config_loader import ConfigLoader
from utils.telemetry import TelemetryCollector


def test_log_prediction_and_flush(tmp_path):
    cfg = ConfigLoader()
    cfg.set('telemetry.enabled', True)
    cfg.set('telemetry.output_dir', str(tmp_path))
    cfg.set('telemetry.anonymize_ids', True)
    cfg.set('telemetry.anonymize_location', True)
    cfg.set('telemetry.location_grid_size_m', 100)
    cfg.set('telemetry.buffer_size', 2)
    cfg.set('telemetry.encrypt', False)

    collector = TelemetryCollector(cfg)
    # first log (buffered)
    collector.log_prediction(0.7, 'high', {'latitude': 40.0, 'longitude': -74.0, 'speed': 30}, {'gps': True}, metadata={'vehicle_hash': 'veh1', 'trip_id': 't1'})
    assert len(collector._buffer) == 1
    # second log triggers flush due to buffer_size=2
    collector.log_prediction(0.4, 'medium', {'latitude': 41.0, 'longitude': -75.0}, {'gps': True}, metadata={'vehicle_hash': 'veh2'})
    # allow flush to occur
    collector.close()

    files = [f for f in os.listdir(str(tmp_path)) if f.startswith('telemetry_') and f.endswith('.jsonl')]
    assert files, 'No telemetry files written'
    # read back first file
    with open(os.path.join(str(tmp_path), files[0]), 'r', encoding='utf-8') as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    assert any(l.get('type') == 'prediction' for l in lines)
    # metadata should have hashed vehicle_hash when anonymize enabled
    m = lines[0].get('metadata', {})
    if 'vehicle_hash' in m and m['vehicle_hash'] is not None:
        assert isinstance(m['vehicle_hash'], str) and len(m['vehicle_hash']) >= 8


def test_log_battery_alert_and_get_stats(tmp_path, monkeypatch):
    from config.config_loader import ConfigLoader

    cfg = ConfigLoader()
    cfg.set('telemetry.enabled', True)
    cfg.set('telemetry.output_dir', str(tmp_path))
    cfg.set('telemetry.anonymize_ids', True)
    cfg.set('telemetry.anonymize_location', True)
    cfg.set('telemetry.location_grid_size_m', 100)
    cfg.set('telemetry.buffer_size', 5)
    cfg.set('telemetry.encrypt', False)

    collector = TelemetryCollector(cfg)
    alert = {"source": "rule", "severity": "HIGH", "reason": "overtemp", "rule": "max_cell_temp", "value": 80, "anomaly_score": None}
    snapshot = [{"cell_temps": [25, 80, 26], "metadata": {"vehicle_hash": "veh123"}}]

    collector.log_battery_alert(alert, telemetry_snapshot=snapshot)

    stats = collector.get_stats()
    assert stats.get('buffered_battery_alerts', 0) >= 1

    # flush to disk and read back entries
    out_files = list(tmp_path.glob('telemetry_*.jsonl'))
    # no flush yet because buffer_size=5; force flush by calling close()
    collector.close()
    out_files = list(tmp_path.glob('telemetry_*.jsonl'))
    assert out_files, 'No telemetry file written on close'
    data = [json.loads(l) for l in out_files[0].read_text(encoding='utf-8').splitlines() if l.strip()]
    assert any((d.get('type') == 'battery_alert' or d.get('source') in ('rule', 'ml')) for d in data)
    # anonymization: metadata vehicle_hash should be hashed inside telemetry_snapshot
    first = data[0]
    # timestamp ISO-ish
    assert 'timestamp' in first and 'T' in first['timestamp']
    # check snapshot anonymization
    snap = first.get('telemetry_snapshot')
    if snap and isinstance(snap, list) and len(snap) > 0:
        md = snap[0].get('metadata', {})
        if 'vehicle_hash' in md and md['vehicle_hash'] is not None:
            assert isinstance(md['vehicle_hash'], str) and len(md['vehicle_hash']) >= 8
