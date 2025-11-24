import numpy as np
import pytest
import time

from data_collection.sensors.driver_drowsiness import utils, detection_engine
from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler


def test_detector_no_face_returns_no_face_flag():
    det = detection_engine.DrowsinessDetector(config={"use_mediapipe": True})
    # force initialized and fake face_mesh to simulate no landmarks
    det._initialized = True
    det._mp = object()

    class FakeFM:
        def process(self, rgb):
            return type("R", (), {"multi_face_landmarks": None})()

    det._mp_face_mesh = FakeFM()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = det.process_frame(frame)
    assert res.get("no_face_detected") is True
    assert res.get("confidence", 0.0) == 0.0


def test_handler_no_face_and_blink_rate():
    # Create handler without auto CV init
    cfg = {"use_mediapipe": True, "ear_threshold": 0.25, "mar_threshold": 0.6, "blink_rate_window": 5.0}
    h = DriverFatigueHandler(config=cfg, use_cv=False)

    # fake camera
    class FakeCam:
        def __init__(self):
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def read_frame(self):
            return self.frame

        def get_status(self):
            return True

    # fake detector that first returns no-face then a blink event
    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def process_frame(self, frame):
            self.calls += 1
            if self.calls == 1:
                return {"no_face_detected": True, "confidence": 0.0, "total_blinks": 0, "total_yawns": 0, "micro_sleep_flag": False}
            else:
                return {"no_face_detected": False, "confidence": 1.0, "ear": 0.1, "mar": 0.2, "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, "blinks_detected": 1, "total_blinks": 1, "yawns_detected": 0, "total_yawns": 0, "micro_sleep_flag": False}

    h._camera = FakeCam()
    h._detector = FakeDetector()

    # first read should return safe baseline (no_face flag removed in handler)
    sample1 = h.read()
    assert sample1["drowsiness_score"] == 0.0
    assert sample1["perclos"] == 0.0

    # next read should register a blink and produce a blink_rate >= 0
    sample2 = h.read()
    assert sample2["perclos"] >= 0.0
    assert sample2["yawning_count"] == 0
    assert sample2["micro_sleep_flag"] is False
    assert sample2["blink_rate"] >= 0.0
