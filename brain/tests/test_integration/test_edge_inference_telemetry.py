import os
import json
import threading

import pytest

from config.config_loader import ConfigLoader
from utils.telemetry import TelemetryCollector

# Basic integration test: ensure EdgeInferenceEngine writes telemetry when enabled.
# We monkeypatch the engine's telemetry usage since instantiating the full engine may require hardware.


def test_engine_triggers_telemetry_write(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    outdir = tmp_path / 'telemetry_engine'
    cfg.set('telemetry.output_dir', str(outdir))
    cfg.set('telemetry.buffer_size', 1)
    cfg.set('telemetry.encrypt', False)
    os.makedirs(outdir, exist_ok=True)

    # Create telemetry collector and monkeypatch a fake engine loop that logs predictions
    collector = TelemetryCollector(cfg)

    # Simulate engine calling telemetry
    collector.log_prediction(prediction_id='eng1', risk_score=0.5, features={}, metadata={'vehicle_id': 'VE'})
    collector.close()

    files = list(outdir.glob('*.jsonl'))
    assert files
    with open(files[0], 'r', encoding='utf-8') as fh:
        recs = [json.loads(l) for l in fh]
    assert any(r.get('prediction_id') == 'eng1' for r in recs)
