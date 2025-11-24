import time
import threading
import json

import numpy as np
import pytest

from data_collection.sensors.driver_drowsiness import detection_engine
from data_collection.sensors.driver_drowsiness import camera_interface as cam_if
from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler


def test_DrowsinessDetector_process_frame_no_face(mock_mediapipe_empty, mock_frame):
    det = detection_engine.DrowsinessDetector(config={"use_mediapipe": True})
    det._initialized = True
    det._mp = object()
    det._mp_face_mesh = mock_mediapipe_empty
    res = det.process_frame(mock_frame)
    assert res.get("no_face_detected") is True
    assert res.get("confidence", 0.0) == 0.0
    assert res.get("total_blinks") == 0


def test_process_frame_events_seq(mock_mediapipe_landmarks_closed, mock_frame, monkeypatch):
    # configure thresholds shorter for tests
    det = detection_engine.DrowsinessDetector(config={"use_mediapipe": True, "blink_duration_threshold_ms": 100, "micro_sleep_duration_s": 1.0, "yawn_duration_threshold_ms": 100})
    det._initialized = True
    det._mp = object()
    det._mp_face_mesh = mock_mediapipe_landmarks_closed
    # avoid calling cv2.solvePnP in tests (OpenCV may require >=6 points)
    import data_collection.sensors.driver_drowsiness.utils as utils
    monkeypatch.setattr(utils, "get_head_pose", lambda *a, **k: {"pitch": 10.0, "yaw": 5.0, "roll": 0.0})

    # first frame: eyes closed -> start timer
    res1 = det.process_frame(mock_frame)
    assert isinstance(res1.get("ear", 0.0), float)

    # wait to exceed blink threshold
    time.sleep(0.12)
    res2 = det.process_frame(mock_frame)
    assert res2.get("total_blinks", 0) >= 1

    # wait to exceed micro-sleep duration
    time.sleep(1.05)
    res3 = det.process_frame(mock_frame)
    assert isinstance(res3.get("micro_sleep_flag"), bool)


def test_head_pose_tilted_landmarks(mock_mediapipe_landmarks_closed, mock_frame, monkeypatch):
    det = detection_engine.DrowsinessDetector(config={"use_mediapipe": True})
    det._initialized = True
    det._mp = object()
    det._mp_face_mesh = mock_mediapipe_landmarks_closed
    import data_collection.sensors.driver_drowsiness.utils as utils
    monkeypatch.setattr(utils, "get_head_pose", lambda *a, **k: {"pitch": 12.0, "yaw": 6.0, "roll": 1.0})
    res = det.process_frame(mock_frame)
    hp = res.get("head_pose", {})
    assert isinstance(hp, dict)
    assert "pitch" in hp and "yaw" in hp and "roll" in hp


def test_CameraInterface_read_success_fail(monkeypatch, mock_camera_cap, mock_frame):
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: mock_camera_cap)
    cam = cam_if.CameraInterface(device=0, width=640, height=480, fps=30)
    f = cam.read_frame()
    assert f is not None
    # exhaust frames
    for _ in range(10):
        _ = cam.read_frame()
    status = cam.get_status()
    assert "failed_reads" in status
    cam.release()


def test_Handler_read_valid_and_schema(driver_cfg, mock_frame):
    cfg = driver_cfg._data if hasattr(driver_cfg, "_data") else driver_cfg
    h = DriverFatigueHandler(cfg, use_cv=False)

    class SeqDetector:
        def __init__(self):
            self.count = 0

        def process_frame(self, frame):
            self.count += 1
            if self.count == 1:
                return {"no_face_detected": False, "confidence": 1.0, "ear": 0.2, "mar": 0.7, "head_pose": {"pitch": 10.0, "yaw": 5.0, "roll": 0.0}, "blinks_detected": 1, "total_blinks": 1, "yawns_detected": 1, "total_yawns": 1, "micro_sleep_flag": False}
            return {"no_face_detected": False, "confidence": 1.0, "ear": 0.3, "mar": 0.2, "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, "blinks_detected": 0, "total_blinks": 1, "yawns_detected": 0, "total_yawns": 1, "micro_sleep_flag": False}

    class FakeCam:
        def read_frame(self):
            return mock_frame

        def get_status(self):
            return True

        def release(self):
            pass

    h._camera = FakeCam()
    h._detector = SeqDetector()

    s1 = h.read()
    assert s1 is not None
    assert 0.0 <= s1["drowsiness_score"] <= 1.0
    assert "perclos" in s1 and "blink_rate" in s1
    time.sleep(0.1)
    s2 = h.read()
    assert isinstance(s2["perclos"], float)
    assert isinstance(s2["blink_rate"], float)


def test_Handler_no_face_low_conf_preserves_counts(driver_cfg, mock_frame):
    cfg = driver_cfg._data if hasattr(driver_cfg, "_data") else driver_cfg
    h = DriverFatigueHandler(cfg, use_cv=False)

    class NoFaceDetector:
        def process_frame(self, frame):
            return {"no_face_detected": True, "confidence": 0.2, "total_blinks": 3, "total_yawns": 2, "micro_sleep_flag": False}

    class FakeCam:
        def read_frame(self):
            return mock_frame

        def get_status(self):
            return True

        def release(self):
            pass

    h._camera = FakeCam()
    h._detector = NoFaceDetector()

    s = h.read()
    assert s["drowsiness_score"] == 0.0
    assert s["blink_rate"] == 0.0
    assert s["perclos"] == 0.0
    assert s["yawning_count"] == 2


def test_handler_concurrent_reads(driver_cfg, mock_frame):
    cfg = driver_cfg._data if hasattr(driver_cfg, "_data") else driver_cfg
    h = DriverFatigueHandler(cfg, use_cv=False)

    class FastDetector:
        def process_frame(self, frame):
            return {"no_face_detected": False, "confidence": 1.0, "ear": 0.3, "mar": 0.2, "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, "blinks_detected": 0, "total_blinks": 0, "yawns_detected": 0, "total_yawns": 0, "micro_sleep_flag": False}

    class FakeCam:
        def read_frame(self):
            return mock_frame

        def get_status(self):
            return True

        def release(self):
            pass

    h._camera = FakeCam()
    h._detector = FastDetector()

    results = []

    def read_once():
        try:
            r = h.read()
            results.append(r)
        except Exception as e:
            results.append(e)

    threads = [threading.Thread(target=read_once) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    for r in results:
        assert isinstance(r, dict)


def test_detector_performance_basic(mock_mediapipe_landmarks_closed, mock_frame, monkeypatch):
    det = detection_engine.DrowsinessDetector(config={"use_mediapipe": True})
    det._initialized = True
    det._mp = object()
    det._mp_face_mesh = mock_mediapipe_landmarks_closed
    import data_collection.sensors.driver_drowsiness.utils as utils
    monkeypatch.setattr(utils, "get_head_pose", lambda *a, **k: {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
    start = time.perf_counter()
    _ = det.process_frame(mock_frame)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5
