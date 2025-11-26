import numpy as np
import pytest

from brain.utils import mediapipe_loaders as mpl


class FakeInterpreter:
    def __init__(self, model_path: str, num_threads: int = 1):
        self.model_path = str(model_path)
        self._last_input = None

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [{"index": 0, "shape": [1, 192, 192, 3], "dtype": np.float32}]

    def get_output_details(self):
        # provide a generic single output slot
        return [{"index": 0, "name": "out"}]

    def set_tensor(self, idx, val):
        self._last_input = val

    def invoke(self):
        return None

    def get_tensor(self, idx):
        mp = self.model_path.lower()
        # face detection: return boxes (1, N, 5)
        if "face_detection" in mp:
            return np.array([[[0.1, 0.1, 0.3, 0.3, 0.95]]], dtype=np.float32)
        # face landmark lite: return 1 x 468 x 3
        if "face_landmark" in mp:
            out = np.zeros((1, 468, 3), dtype=np.float32)
            for i in range(468):
                out[0, i, 0] = 0.001 * i
                out[0, i, 1] = 0.002 * i
                out[0, i, 2] = 0.0
            return out
        # iris: small set of points
        if "iris" in mp:
            out = np.array([[[0.5, 0.5], [0.52, 0.5], [0.48, 0.5], [0.5, 0.52], [0.5, 0.48]]]], dtype=np.float32)
            return out
        # pose detection: return one bbox row
        if "pose_detection" in mp:
            return np.array([[[0.2, 0.2, 0.6, 0.6, 0.9]]], dtype=np.float32)
        # pose landmark: 1 x 33 x 3
        if "pose_landmark" in mp:
            out = np.zeros((1, 33, 3), dtype=np.float32)
            # set hips and knees y such that hip_y < knee_y - 0.05 (walking_towards True)
            # hips at index 23/24, knees 25/26
            out[0, 23, 1] = 0.3
            out[0, 24, 1] = 0.32
            out[0, 25, 1] = 0.5
            out[0, 26, 1] = 0.52
            return out
        # default small landmark
        return np.array([[[0.1, 0.1], [0.2, 0.2]]], dtype=np.float32)


def _patch_interp(monkeypatch):
    monkeypatch.setattr(mpl, "_load_interpreter", lambda model_path, num_threads=1: FakeInterpreter(model_path, num_threads))


def test_face_detector_parsing(monkeypatch):
    _patch_interp(monkeypatch)
    fd = mpl.MediaPipeFaceDetector("face_detection_short_range.tflite", config={"face_detection_confidence": 0.5})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = fd.detect(frame)
    assert isinstance(res, list)
    assert len(res) == 1
    assert "bbox" in res[0]
    assert res[0]["score"] >= 0.9


def test_face_landmarker_parsing(monkeypatch):
    _patch_interp(monkeypatch)
    lm = mpl.MediaPipeFaceLandmarker("face_landmark_lite.tflite", config={})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # use a center bbox
    res = lm.detect_landmarks(frame, (0.1, 0.1, 0.5, 0.5))
    assert isinstance(res, list)
    assert len(res) == 468
    # coords normalized
    assert 0.0 <= res[0][0] <= 1.0


def test_iris_landmarker_parsing(monkeypatch):
    _patch_interp(monkeypatch)
    im = mpl.MediaPipeIrisLandmarker("iris_landmark.tflite", config={})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = im.detect_iris(frame, (0.45, 0.45, 0.1, 0.1))
    assert isinstance(out, dict)
    assert "center" in out and isinstance(out["center"], tuple)


def test_full_pipeline_mediapipe_chain(monkeypatch):
    # Patch interpreters for all mediapipe models
    _patch_interp(monkeypatch)
    from data_collection.sensors.driver_drowsiness.tflite_detector import TFLiteDrowsinessDetector

    cfg = {
        "use_mediapipe_tasks": True,
        "mediapipe": {
            "tflite_num_threads": 1,
            "face_detection": {"model_path": "face_detection_short_range.tflite"},
            "face_landmark": {"model_path": "face_landmark_lite.tflite"},
            "iris_landmark": {"model_path": "iris_landmark.tflite"},
        },
    }

    det = TFLiteDrowsinessDetector(config=cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.process_frame(frame)
    assert out.get("backend_used") == "mediapipe_tflite"
    assert out.get("no_face_detected") is False
    assert "ear" in out and isinstance(out["ear"], float)
    assert "iris" in out and isinstance(out["iris"], dict)


def test_tflite_fallback_heuristic():
    # create detector with mediapipe disabled -> should fallback to heuristic outputs
    from data_collection.sensors.driver_drowsiness.tflite_detector import TFLiteDrowsinessDetector

    det = TFLiteDrowsinessDetector(config={"use_mediapipe_tasks": False})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.process_frame(frame)
    assert isinstance(out, dict)
    assert out.get("no_face_detected") in (True, False)


def test_camera_pose_intent(monkeypatch):
    # Verify camera handler enriches person objects with pose_data.walking_towards True
    from config.config_loader import ConfigLoader
    from data_collection.sensors.camera_handler import CameraHandler

    cfg = ConfigLoader("config/default_config.yaml")
    ch = CameraHandler(cfg)
    # force production path
    ch.enable_simulation = False

    # dummy model that returns one person bbox
    class DummyModel:
        def predict(self, frame):
            return [{"class": "person", "confidence": 0.99, "bbox_pixels": [10, 10, 200, 400]}]

    ch._model = DummyModel()

    # monkeypatch OpenCV availability and capture
    import data_collection.sensors.camera_handler as cam_mod
    cam_mod._ensure_cv2 = lambda: True

    class DummyCap:
        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

    ch._ensure_capture = lambda: DummyCap()

    # patch MediaPipe pose detector to a fake instance with detect_landmarks
    class FakePose:
        def detect_landmarks(self, frame, bbox):
            # return 33 keypoints with hips lower (higher y) than knees -> hip_y < knee_y - margin
            kps = [(0.0, 0.0)] * 33
            kps[23] = (0.5, 0.3)
            kps[24] = (0.5, 0.32)
            kps[25] = (0.5, 0.6)
            kps[26] = (0.5, 0.62)
            return kps

    ch._mp_pose_detector = FakePose()

    res = ch._infer_frame(0.0)
    objs = res.get("objects", [])
    assert len(objs) >= 1
    p = next((o for o in objs if o.get("class") == "person"), None)
    assert p is not None
    assert "pose_data" in p
    assert p["pose_data"].get("walking_towards") is True
