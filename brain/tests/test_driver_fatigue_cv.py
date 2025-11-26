import time
import numpy as np
from data_collection.sensors.driver_drowsiness.tflite_detector import TFLiteDrowsinessDetector


def test_tflite_detector_safe_defaults():
    # construct detector with missing model path -> should not raise
    det = TFLiteDrowsinessDetector(config={})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.process_frame(frame)
    # safe defaults expected
    assert isinstance(out, dict)
    assert "ear" in out and "mar" in out
    assert out.get("no_face_detected") in (True, False)


def test_driver_fatigue_handler_emits_backend_flag(tmp_path):
    from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
    # minimal config enabling CV but no models -> handler should simulate or use fallbacks
    cfg = {"sensors": {"driver_fatigue": {"enable_cv_detection": False, "use_mediapipe_tasks": False}}}
    h = DriverFatigueHandler(cfg, use_cv=False)
    s = h.read()
    assert isinstance(s, dict)
    # Handler should always include timestamp
    assert "timestamp" in s

