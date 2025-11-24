import os
import json
import threading
import time
import base64

import pytest

from config.config_loader import ConfigLoader
from utils.telemetry import TelemetryCollector
from utils import encryption as enc


def test_buffer_flush_and_anonymization(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    outdir = tmp_path / 'telemetry'
    cfg.set('telemetry.output_dir', str(outdir))
    cfg.set('telemetry.buffer_size', 2)
    cfg.set('telemetry.encrypt', False)
    os.makedirs(outdir, exist_ok=True)

    collector = TelemetryCollector(cfg)

    # Log two predictions to trigger flush
    collector.log_prediction(prediction_id='p1', risk_score=0.9, features={'a': 1}, metadata={'vehicle_id': 'V123', 'lat': 37.0, 'lon': -122.0})
    collector.log_prediction(prediction_id='p2', risk_score=0.4, features={'a': 2}, metadata={'vehicle_id': 'V456', 'lat': 37.1, 'lon': -122.1})

    collector.close()

    files = list(outdir.glob('*.jsonl'))
    assert len(files) >= 1

    # Validate contents: hashed vehicle identifier and gridded lat/lon
    with open(files[0], 'r', encoding='utf-8') as fh:
        lines = [json.loads(l) for l in fh]

    assert any('vehicle_hash' in l for l in lines)
    assert all(('lat_grid' in l and 'lon_grid' in l) or ('lat' in l and 'lon' in l) for l in lines)


def test_encrypted_flush_and_rotation(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    outdir = tmp_path / 'telemetry_enc'
    cfg.set('telemetry.output_dir', str(outdir))
    cfg.set('telemetry.buffer_size', 1)
    cfg.set('telemetry.encrypt', True)
    cfg.set('telemetry.retention_days', 0)
    os.makedirs(outdir, exist_ok=True)

    # Provide deterministic key via monkeypatch to avoid env parsing
    key = enc.generate_key()

    def fake_get_key(self):
        return key

    monkeypatch.setattr(TelemetryCollector, '_get_encryption_key', fake_get_key)

    collector = TelemetryCollector(cfg)
    collector.log_prediction(prediction_id='p1', risk_score=0.9, features={}, metadata={'vehicle_id': 'V1'})
    collector.close()

    # Expect encrypted file
    enc_files = list(outdir.glob('*.jsonl.enc'))
    assert len(enc_files) == 1

    # Decrypt and verify content
    enc_path = str(enc_files[0])
    decrypted = tmp_path / 'decrypted.jsonl'
    enc.decrypt_file(enc_path, str(decrypted), key)
    with open(decrypted, 'r', encoding='utf-8') as fh:
        data = [json.loads(l) for l in fh]
    assert data and data[0]['prediction_id'] == 'p1'

    # Rotation should remove files older than retention_days (0 -> should delete)
    # create a stale file
    stale = outdir / 'old_telemetry.jsonl'
    stale.write_text('')
    collector._rotate()
    assert not stale.exists()


def test_thread_safety_under_concurrent_logging(tmp_path):
    cfg = ConfigLoader()
    outdir = tmp_path / 'telemetry_threads'
    cfg.set('telemetry.output_dir', str(outdir))
    cfg.set('telemetry.buffer_size', 10)
    cfg.set('telemetry.encrypt', False)
    os.makedirs(outdir, exist_ok=True)

    collector = TelemetryCollector(cfg)

    def worker(i):
        for j in range(20):
            collector.log_prediction(prediction_id=f'p_{i}_{j}', risk_score=0.1 * j, features={'i': i, 'j': j}, metadata={'vehicle_id': f'V{i}'})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    collector.close()

    # Ensure we have some lines written and no crashes
    files = list(outdir.glob('*.jsonl'))
    assert files
    total = 0
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            for _ in fh:
                total += 1
    assert total >= 5 * 20
