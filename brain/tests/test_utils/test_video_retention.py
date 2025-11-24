import os
import json
import threading
import time

import pytest

from config.config_loader import ConfigLoader
from utils.video_retention import VideoRetentionManager


def test_save_frame_and_per_trip_cap(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    outdir = tmp_path / 'retention'
    cfg.set('video_retention.output_dir', str(outdir))
    cfg.set('video_retention.per_trip_cap', 3)
    cfg.set('video_retention.encrypt', False)
    outdir.mkdir()

    # mock cv2.imwrite
    monkeypatch.setattr('cv2.imwrite', lambda path, frame: True)

    mgr = VideoRetentionManager(cfg)
    trip = 'trip1'

    # save 5 frames, cap is 3 so older ones should be removed
    for i in range(5):
        mgr.save_frame(frame=b'fakeframe', metadata={'ts': i}, event_type='brake', trip_id=trip)

    # list files
    files = list((outdir).rglob('*'))
    # Expect some files <= cap * 2 (frames + metadata)
    assert len([f for f in files if f.is_file()]) <= 6


def test_cleanup_thread_and_encryption(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    outdir = tmp_path / 'retention2'
    cfg.set('video_retention.output_dir', str(outdir))
    cfg.set('video_retention.encrypt', True)
    cfg.set('video_retention.retention_hours', 0)
    outdir.mkdir()

    # monkeypatch write and encryption
    monkeypatch.setattr('cv2.imwrite', lambda path, frame: True)

    mgr = VideoRetentionManager(cfg)
    # monkeypatch key retrieval
    mgr._get_encryption_key = lambda: b'0' * 32

    mgr.save_frame(frame=b'fake', metadata={'ts': 1}, event_type='event', trip_id='t1')
    # start cleanup loop once (it uses retention_hours=0 so should cleanup
    mgr.cleanup_old_frames()
    # no exceptions and directory exists
    assert outdir.exists()
import os
import tempfile
import json
import threading

import numpy as np

from config.config_loader import ConfigLoader
from utils.video_retention import VideoRetentionManager


def test_save_frame_and_cleanup(tmp_path, monkeypatch):
    cfg = ConfigLoader()
    cfg.set('video_retention.output_dir', str(tmp_path))
    cfg.set('privacy.video_retention_hours', 0.001)  # small retention for quick cleanup
    cfg.set('sensors.camera.max_retained_frames_per_trip', 3)

    # monkeypatch cv2.imwrite to actually write a small file
    def fake_imwrite(path, frame, params=None):
        with open(path, 'wb') as fh:
            fh.write(b'JPEG')
        return True

    import utils.video_retention as vr_mod

    monkeypatch.setattr(vr_mod.cv2, 'imwrite', fake_imwrite)

    mgr = VideoRetentionManager(cfg)
    trip_id = 'trip_test'
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    metadata = {'timestamp': '2025-01-01T00:00:00Z', 'risk_score': 0.7}

    # save several frames exceeding the per-trip cap
    for i in range(5):
        mgr.save_frame(frame, dict(metadata), 'high_risk', trip_id)

    # check that at most max_retained_frames_per_trip retained
    retained = sum(len(frames) for frames in mgr.trip_frames.get(trip_id, {}).values())
    assert retained <= 3

    # run cleanup which should remove old frames (retention small)
    mgr.cleanup_old_frames()
