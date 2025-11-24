import time
import numpy as np
import pytest
import time

from data_collection.sensors.driver_drowsiness import detection_engine
from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler


def test_DrowsinessDetector_no_face_behavior():
    # Detector should report no_face_detected when mediapipe returns no landmarks
    det = detection_engine.DrowsinessDetector(config={"use_mediapipe": True})
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


def test_handler_read_no_face_and_event_sequence():
    cfg = {"sensors": {"driver_fatigue": {"ear_threshold": 0.25, "mar_threshold": 0.6, "blink_rate_window": 2.0}}}
    h = DriverFatigueHandler(cfg, use_cv=False)

    class FakeCam:
        def __init__(self):
            self._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def read_frame(self):
            return self._last_frame

        def get_status(self):
            return True

        def release(self):
            pass

    class SeqDetector:
        def __init__(self):
            self.n = 0

        def process_frame(self, frame):
            self.n += 1
            if self.n == 1:
                return {"no_face_detected": True, "confidence": 0.0, "total_blinks": 0, "total_yawns": 0, "micro_sleep_flag": False}
            if self.n == 2:
                # closed eye -> blink event
                return {"no_face_detected": False, "confidence": 1.0, "ear": 0.1, "mar": 0.2, "head_pose": {"pitch": 5.0, "yaw": 2.0, "roll": 0.0}, "blinks_detected": 1, "total_blinks": 1, "yawns_detected": 0, "total_yawns": 0, "micro_sleep_flag": False}
            return {"no_face_detected": False, "confidence": 1.0, "ear": 0.3, "mar": 0.2, "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, "blinks_detected": 0, "total_blinks": 1, "yawns_detected": 0, "total_yawns": 0, "micro_sleep_flag": False}

    h._camera = FakeCam()
    h._detector = SeqDetector()

    s1 = h.read()
    assert s1["drowsiness_score"] == 0.0
    assert s1["perclos"] == 0.0

    s2 = h.read()
    assert s2["perclos"] >= 0.0
    assert s2["blink_rate"] >= 0.0

    time.sleep(2.1)
    s3 = h.read()
    assert s3["perclos"] >= 0.0


def test_compute_ear_known_points():
    from data_collection.sensors.driver_drowsiness import utils

    p1 = (0.0, 0.0)
    p2 = (0.0, 1.0)
    p3 = (1.0, 1.0)
    p4 = (4.0, 0.0)
    p5 = (1.0, -1.0)
    p6 = (0.0, -1.0)
    ear = utils.compute_ear([p1, p2, p3, p4, p5, p6])
    assert ear > 0.0


def test_compute_mar_simple():
    from data_collection.sensors.driver_drowsiness import utils

    left = (0.0, 0.0)
    right = (4.0, 0.0)
    top = (2.0, 1.0)
    bottom = (2.0, -1.0)
    mar = utils.compute_mar([left, None, None, top, None, None, right, bottom])
    assert pytest.approx(mar, rel=1e-2) == pytest.approx(2.0 / 4.0, rel=1e-2)


def test_preprocess_ir_frame_type_and_shape():
    from data_collection.sensors.driver_drowsiness import utils

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = utils.preprocess_ir_frame(img, enable_hist_eq=True, enable_clahe=False)
    assert out.ndim == 2
    assert out.shape[0] == 480 and out.shape[1] == 640
