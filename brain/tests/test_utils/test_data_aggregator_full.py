import os
import json

import pandas as pd
import pytest

from config.config_loader import ConfigLoader
from utils.data_aggregator import DataAggregator


def write_telemetry_file(path, entries):
    with open(path, 'w', encoding='utf-8') as fh:
        for e in entries:
            fh.write(json.dumps(e) + '\n')


def test_data_aggregator_matching_and_report(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    tdir = tmp_path / 'telemetry'
    tdir.mkdir()
    cfg.set('telemetry.output_dir', str(tdir))
    cfg.set('retraining.output_dir', str(tmp_path / 'out'))
    cfg.set('retraining.min_samples_per_class', 1)

    # prediction and matching outcome
    entries = [
        {'timestamp': '2025-01-01T00:00:00', 'type': 'prediction', 'prediction_id': 'p1', 'features': {'d': 3}},
        {'timestamp': '2025-01-01T00:00:05', 'type': 'outcome', 'prediction_id': 'p1', 'actual_outcome': 'accident'},
    ]
    fpath = tdir / 't1.jsonl'
    write_telemetry_file(str(fpath), entries)

    aggregator = DataAggregator(cfg)

    # monkeypatch extractor
    def fake_extract(window):
        return {'f_d': window.get('features', {}).get('d', 0)}

    aggregator.extractor.extract = fake_extract

    out_csv = str(tmp_path / 'dataset.csv')
    csv_path = aggregator.aggregate_for_retraining(str(tdir), out_csv, None, None, incremental=False)
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert 'label' in df.columns


def test_data_aggregator_insufficient_samples_raises(tmp_path):
    cfg = ConfigLoader()
    tdir = tmp_path / 'telemetry2'
    tdir.mkdir()
    cfg.set('telemetry.output_dir', str(tdir))
    cfg.set('retraining.output_dir', str(tmp_path / 'out2'))
    cfg.set('retraining.min_samples_per_class', 10)

    # only one sample
    entries = [
        {'timestamp': '2025-01-01T00:00:00', 'type': 'prediction', 'prediction_id': 'p1', 'features': {'d': 3}},
        {'timestamp': '2025-01-01T00:00:05', 'type': 'outcome', 'prediction_id': 'p1', 'actual_outcome': 'accident'},
    ]
    fpath = tdir / 't1.jsonl'
    write_telemetry_file(str(fpath), entries)

    aggregator = DataAggregator(cfg)

    # monkeypatch extractor to simple
    aggregator.extractor.extract = lambda w: {'f': 1}

    out_csv = str(tmp_path / 'dataset2.csv')
    # Should raise or produce a dataset that doesn't meet min_samples; expect ValueError
    with pytest.raises(Exception):
        aggregator.aggregate_for_retraining(str(tdir), out_csv, None, None, incremental=False)
