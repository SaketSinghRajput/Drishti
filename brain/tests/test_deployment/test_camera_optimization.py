import types
import pytest


def test_camera_backend_preference(monkeypatch):
    # Create a fake cv2 module with VideoCapture and constants
    class FakeCap:
        def __init__(self, *args, **kwargs):
            self._opened = True
            self._props = {}

        def isOpened(self):
            return True

        def set(self, k, v):
            self._props[k] = v

    fake_cv2 = types.SimpleNamespace()
    fake_cv2.CAP_V4L2 = 200
    fake_cv2.CAP_GSTREAMER = 300
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
    fake_cv2.CAP_PROP_FOURCC = 6
    fake_cv2.VideoWriter_fourcc = lambda a, b, c, d: 0x4D4A5047
    fake_cv2.VideoCapture = lambda *a, **k: FakeCap()

    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    # import CameraHandler lazily to pick up monkeypatched cv2
    from data_collection.sensors.camera_handler import CameraHandler
    from config.config_loader import ConfigLoader

    cfg = {"sensors": {"camera": {"device_index": 0, "enable_simulation": False, "preferred_pixel_format": "MJPEG", "backend": "v4l2"}}}
    # wrap into minimal ConfigLoader-like object with get
    class Cfg:
        def __init__(self, d):
            self._d = d

        def get(self, k, default=None):
            if k == "sensors.camera":
                return self._d["sensors"]["camera"]
            return default

    handler = CameraHandler(Cfg(cfg))
    cap = handler._ensure_capture()
    assert cap is not None
