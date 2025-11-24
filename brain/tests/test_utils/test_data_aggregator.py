import os
import json

import pandas as pd

from config.config_loader import ConfigLoader
from utils.data_aggregator import DataAggregator


def make_sample_telemetry(tmpdir):
    d = tmpdir
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'telemetry_20250101_000000.jsonl')
    entries = [
        {'timestamp': '2025-01-01T00:00:00', 'type': 'prediction', 'prediction_id': 'p1', 'risk_score': 0.8, 'features': {'min_distance_m': 3.0}, 'metadata': {}},
        {'timestamp': '2025-01-01T00:00:05', 'type': 'outcome', 'prediction_id': 'p1', 'actual_outcome': 'accident', 'time_to_event_s': 5},
    ]
    with open(path, 'w', encoding='utf-8') as fh:
        for e in entries:
            fh.write(json.dumps(e) + '\n')
    return d, path


def test_aggregate_basic(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    cfg.set('retraining.output_dir', str(tmp_path))
    cfg.set('retraining.min_samples_per_class', 1)
    telemetry_dir, p = make_sample_telemetry(str(tmp_path / 'telemetry'))

    aggregator = DataAggregator(cfg)

    # monkeypatch extractor to produce simple features
    def fake_extract(window):
        return {'f_min_dist': window.get('features', {}).get('min_distance_m', 999.0)}

    aggregator.extractor.extract = fake_extract

    out_csv = str(tmp_path / 'dataset.csv')
    path = aggregator.aggregate_for_retraining(telemetry_dir, out_csv, None, None, incremental=False)
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert 'label' in df.columns
