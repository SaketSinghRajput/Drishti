import time
from config.config_loader import ConfigLoader
from data_collection.sensors.camera_handler import CameraHandler


def test_camera_handler_basic_read():
    cfg = ConfigLoader("config/default_config.yaml")
    ch = CameraHandler(cfg)
    assert ch.is_available()
    # call read a few times
    for _ in range(3):
        r = ch.read()
        assert isinstance(r, dict)
        assert "timestamp" in r
        assert "objects" in r
        assert "closest_obj_dist_m" in r


def test_infer_frame_passes_frame_to_hook():
    cfg = ConfigLoader("config/default_config.yaml")

    class HookHandler(CameraHandler):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.hooked = None

        def on_frame_flagged(self, metadata, frame=None):
            # record the call for assertions
            self.hooked = (metadata, frame)

    hh = HookHandler(cfg)
    # force production inference path
    hh.enable_simulation = False
    # provide a dummy model that will produce a flagged detection
    class DummyModel:
        def predict(self, frame):
            return [{"class": "person", "confidence": 0.99, "bbox_pixels": [10, 10, 50, 100]}]

    hh._model = DummyModel()

    # monkeypatch capture to return a simple placeholder frame (no numpy required)
    class DummyCap:
        def read(self):
            # frame can be any object; DummyModel.predict ignores it in this test
            return True, [[0]]

    hh._ensure_capture = lambda: DummyCap()
    # force flagging for the purpose of this unit test
    hh.should_flag_frame = lambda objs, agg=None: True

    # ensure the module-level OpenCV check passes so _infer_frame will use
    # the production code path (tests may not have cv2 installed)
    import data_collection.sensors.camera_handler as cam_mod
    cam_mod._ensure_cv2 = lambda: True

    res = hh._infer_frame(time.time())
    # hook should have been called with a non-None frame
    assert hh.hooked is not None
    meta, frame = hh.hooked
    assert isinstance(meta, dict)
    assert frame is not None
