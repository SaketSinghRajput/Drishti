import types
import numpy as np

from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler


class FakeDetector:
    def __init__(self, cfg=None):
        pass
    def process_frame(self, frame):
        return {
            'ear': 0.3,
            'mar': 0.2,
            'head_pose': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
            'confidence': 1.0,
            'no_face_detected': False,
            'blinks_detected': 0,
            'yawns_detected': 0,
            'total_yawns': 0,
            'micro_sleep_flag': False,
        }

class FakeCamera:
    def __init__(self, *args, **kwargs):
        self.is_ir = False
        self.enable_low_power_mode = False
        self.low_power_fps = 2
        self.normal_fps = 30
        self._frame = np.zeros((480,640,3), dtype=np.uint8)
    def read_frame(self):
        return self._frame
    def get_status(self):
        return {'current_fps': self.low_power_fps if False else self.normal_fps, 'low_power_active': False}
    def mark_no_face_detected(self):
        pass
    def clear_no_face_counter(self):
        pass
    def release(self):
        pass


def test_handler_includes_latency_stats(monkeypatch):
    # prepare config dict with profiler enabled
    cfg = {'sensors.driver_fatigue.enable_latency_profiling': True, 'sensors.driver_fatigue.latency_budget_ms': 200}
    # monkeypatch detector and camera imports used in handler
    monkeypatch.setitem(__import__('sys').modules, 'data_collection.sensors.driver_drowsiness.detection_engine', types.SimpleNamespace(DrowsinessDetector=FakeDetector))
    monkeypatch.setitem(__import__('sys').modules, 'data_collection.sensors.driver_drowsiness.camera_interface', types.SimpleNamespace(CameraInterface=FakeCamera))
    # instantiate handler
    handler = DriverFatigueHandler(cfg, use_cv=True)
    s = handler.read()
    assert isinstance(s, dict)
    health = handler.get_health()
    assert 'latency_stats' in health

