import os
import json
import tempfile
from unittest import mock

import pytest

from config.config_loader import ConfigLoader
from utils.cloud_sync import CloudSyncManager
from utils import encryption as enc


def make_telemetry_file(directory, contents):
    p = os.path.join(directory, 't1.jsonl')
    with open(p, 'w', encoding='utf-8') as fh:
        for c in contents:
            fh.write(json.dumps(c) + '\n')
    return p


def test_cloudsync_aggregates_and_uploads(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    tdir = tmp_path / 'telemetry'
    tdir.mkdir()
    cfg.set('telemetry.output_dir', str(tdir))
    cfg.set('telemetry.encrypt', False)

    # Create a telemetry file
    contents = [{'type': 'prediction', 'risk_score': 0.9}, {'type': 'outcome', 'actual_outcome': 'accident'}]
    p = make_telemetry_file(str(tdir), contents)

    # Mock requests.post to simulate retry then success
    calls = {'n': 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls['n'] += 1
        if calls['n'] == 1:
            raise Exception('network')
        return mock.Mock(status_code=200, json=lambda: {'ok': True})

    monkeypatch.setattr('requests.post', fake_post)

    manager = CloudSyncManager(cfg)
    # monkeypatch key retrieval to None (no decryption needed)
    monkeypatch.setattr(CloudSyncManager, '_get_key', lambda self: None)

    # Run sync
    manager.sync()

    # telemetry file should be renamed to .synced
    assert os.path.exists(p + '.synced')
    # sync log should exist
    sync_log = tdir / 'sync_log.jsonl'
    assert sync_log.exists()
    with open(sync_log, 'r', encoding='utf-8') as fh:
        logs = [json.loads(l) for l in fh]
    assert logs


def test_cloudsync_handles_encrypted_files(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    tdir = tmp_path / 'telemetry_enc'
    tdir.mkdir()
    cfg.set('telemetry.output_dir', str(tdir))
    cfg.set('telemetry.encrypt', True)

    key = enc.generate_key()
    data = {'type': 'prediction', 'prediction_id': 'px'}
    p = tdir / 'tenc.jsonl'
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(data) + '\n')
    encf = str(tdir / 'tenc.jsonl.enc')
    enc.encrypt_file(str(p), encf, key)
    p.unlink()

    # monkeypatch post to succeed
    monkeypatch.setattr('requests.post', lambda *a, **k: mock.Mock(status_code=200, json=lambda: {'ok': True}))

    manager = CloudSyncManager(cfg)
    monkeypatch.setattr(CloudSyncManager, '_get_key', lambda self: key)

    manager.sync()

    # encrypted file should be renamed .synced
    assert os.path.exists(encf + '.synced')
import os
import json
import tempfile
from unittest import mock

import pytest

from config.config_loader import ConfigLoader
from utils.cloud_sync import CloudSyncManager
from utils import encryption


def make_sample_telemetry_file(dirpath, name='telemetry_test.jsonl'):
    path = os.path.join(dirpath, name)
    entries = [
        {'timestamp': '2025-01-01T00:00:00Z', 'type': 'prediction', 'risk_score': 0.7, 'metadata': {'vehicle_hash': 'v1'}},
        {'timestamp': '2025-01-01T00:00:05Z', 'type': 'outcome', 'actual_outcome': 'accident', 'prediction_id': 'p1'},
        {'timestamp': '2025-01-01T00:00:10Z', 'type': 'false_alarm', 'prediction_id': 'p2'},
    ]
    with open(path, 'w', encoding='utf-8') as fh:
        for e in entries:
            fh.write(json.dumps(e) + '\n')
    return path


def test_sync_plain_and_encrypted(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    cfg.set('telemetry.output_dir', str(tmp_path))
    cfg.set('telemetry.cloud_sync.enabled', True)
    cfg.set('telemetry.cloud_sync.endpoint_url', 'https://example.com/ingest')
    cfg.set('telemetry.cloud_sync.api_key', 'testkey')

    # create plain file
    plain = make_sample_telemetry_file(str(tmp_path), 'telemetry_20250101_000000.jsonl')

    # create encrypted file
    key = encryption.generate_key()
    # set env var as hex
    os.environ['TELEMETRY_ENCRYPTION_KEY'] = key.hex()
    enc_plain = make_sample_telemetry_file(str(tmp_path), 'telemetry_20250101_010000.jsonl')
    enc_path = enc_plain + '.enc'
    encryption.encrypt_file(enc_plain, enc_path, key)
    os.remove(enc_plain)

    mgr = CloudSyncManager(cfg)

    # mock requests.post to succeed
    with mock.patch('utils.cloud_sync.requests.post') as mock_post:
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        res = mgr.sync()
        assert res['status'] in ('ok', 'ok')
        assert res['synced'] >= 1

    # verify .synced files exist for those that were renamed
    files = os.listdir(str(tmp_path))
    assert any(fn.endswith('.synced') for fn in files)
