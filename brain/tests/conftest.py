import os
import json
import tempfile
from datetime import datetime, timezone

import pytest

from config.config_loader import ConfigLoader
import threading


# Ensure ConfigLoader singleton is reset between tests to avoid state leakage
@pytest.fixture(autouse=True)
def reset_config_loader_singleton():
    """Reset ConfigLoader singleton before each test to ensure test isolation."""
    from config.config_loader import ConfigLoader as _CL

    _CL._instance = None
    _CL._instance_lock = threading.Lock()
    yield
    # reset again after test for extra safety
    _CL._instance = None


@pytest.fixture(scope="session")
def temp_data_dir():
    d = tempfile.mkdtemp(prefix="test_data_")
    yield d
    # cleanup
    try:
        for fn in os.listdir(d):
            os.remove(os.path.join(d, fn))
        os.rmdir(d)
    except Exception:
        pass


@pytest.fixture
def sample_config(tmp_path):
    cfg = {
        "sensors": {"gps": {"sample_rate": 10}},
        "model": {"prediction_horizon_s": 10},
    }
    p = tmp_path / "test_config.yaml"
    with open(p, "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(cfg, f)
    loader = ConfigLoader(str(p))
    return loader


@pytest.fixture
def mock_frame():
    import numpy as np

    # simple black frame 480x640 BGR
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def driver_cfg(tmp_path):
    cfg = {
        "sensors": {
            "driver_fatigue": {
                "sample_rate": 1,
                "ear_threshold": 0.25,
                "mar_threshold": 0.6,
                "face_detection_confidence": 0.5,
                "blink_rate_window": 5.0,
                "head_tilt_threshold_deg": 30.0,
            }
        }
    }
    p = tmp_path / "driver_cfg.yaml"
    with open(p, "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(cfg, f)
    from config.config_loader import ConfigLoader

    return ConfigLoader(str(p))


@pytest.fixture
def mock_mediapipe_empty():
    """Return an object that mimics MediaPipe FaceMesh processing with no landmarks."""
    class EmptyResult:
        def __init__(self):
            self.multi_face_landmarks = None

    class FakeFaceMesh:
        def process(self, rgb):
            return EmptyResult()

    return FakeFaceMesh()


@pytest.fixture
def mock_mediapipe_landmarks_closed():
    """Return a fake results object with landmarks representing closed eyes and open mouth."""
    class Landmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class FakeLM:
        def __init__(self, landmarks):
            self.landmark = landmarks

    class FakeResult:
        def __init__(self, landmarks):
            self.multi_face_landmarks = [FakeLM(landmarks)]

    class FakeFaceMesh:
        def __init__(self, landmarks):
            self._landmarks = landmarks

        def process(self, rgb):
            return FakeResult(self._landmarks)

    # create simplified landmarks list with at least 320+ indices used by detector utils
    lm = []
    # populate with default points on a grid; some indices will be modified to simulate closed eyes and open mouth
    for i in range(400):
        lm.append(Landmark(0.5, 0.5))

    # tweak eye and mouth points to simulate closed eyes (small vertical spread) and open mouth (large vertical spread)
    # left eye indices approx 33,160,158,133,153,144
    for idx in [33, 160, 158, 133, 153, 144]:
        lm[idx] = Landmark(0.45, 0.5)
    # right eye
    for idx in [362, 385, 387, 263, 373, 380]:
        lm[idx] = Landmark(0.55, 0.5)
    # mouth: 61,291,78,308,13,14,308,14 -> make vertical separation larger
    lm[61] = Landmark(0.48, 0.45)
    lm[291] = Landmark(0.52, 0.55)
    lm[78] = Landmark(0.5, 0.44)
    lm[308] = Landmark(0.5, 0.56)
    lm[13] = Landmark(0.5, 0.46)
    lm[14] = Landmark(0.5, 0.54)

    return FakeFaceMesh(lm)


@pytest.fixture
def mock_camera_cap(monkeypatch, mock_frame):
    """Provide a mock CameraInterface-like object that yields frames and simulates read failures."""
    class MockCap:
        def __init__(self):
            self._frames = [mock_frame.copy() for _ in range(5)]
            self._index = 0
            self._opened = True
            self._failed_reads = 0

        def read(self):
            if self._index < len(self._frames):
                f = self._frames[self._index]
                self._index += 1
                return True, f
            self._failed_reads += 1
            return False, None

        def isOpened(self):
            return self._opened

        def release(self):
            self._opened = False

    return MockCap()


@pytest.fixture
def sample_sensor_window():
    # minimal valid sample matching schema
    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "vehicle_hash": "vh_abc123",
        "gps": {"lat": 37.0, "lon": -122.0, "speed_kph": 50.0, "heading": 180},
        "imu": {"yaw_rate": 0.1, "pitch": 0.0, "roll": 0.0, "ax": 0.0, "ay": 0.0, "az": 9.8},
        "vehicle_controls": {"steering_angle": 0.0, "brake_pressure": 0.0, "brake_flag": False, "throttle_position": 0.2},
        "driver_fatigue": {
            "drowsiness_score": 0.1,
            "blink_rate": 10,
            "perclos": 0.1,
            "head_pose": {"pitch": 0, "yaw": 0, "roll": 0, "confidence": 1.0},
            "yawning_count": 0,
            "micro_sleep_flag": False,
        },
    }
