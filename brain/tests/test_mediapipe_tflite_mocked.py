import numpy as np

from brain.utils import mediapipe_loaders as mpl


class FakeInterpreter:
    def __init__(self, model_path: str, num_threads: int = 1):
        self.model_path = model_path
        self._last_input = None

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [{"index": 0, "shape": [1, 192, 192, 3], "dtype": np.float32}]

    def get_output_details(self):
        return [{"index": 0, "name": "out"}]

    def set_tensor(self, idx, val):
        self._last_input = val

    def invoke(self):
        return None

    def get_tensor(self, idx):
        # Return plausible outputs depending on model name
        mp = self.model_path.lower()
        if "face_detection" in mp:
            # shape (1,1,5): ymin,xmin,ymax,xmax,score
            return np.array([[[0.1, 0.1, 0.3, 0.3, 0.95]]], dtype=np.float32)
        if "face_landmark" in mp:
            # return 1 x 20 x 2 landmarks (simple symmetric points)
            pts = np.zeros((1, 20, 2), dtype=np.float32)
            for i in range(20):
                pts[0, i, 0] = 0.1 + 0.02 * i
                pts[0, i, 1] = 0.2 + 0.01 * i
            return pts
        if "iris" in mp:
            # return 1 x 6 x 2 (few iris points)
            pts = np.array([[[0.5, 0.5], [0.52, 0.5], [0.48, 0.5], [0.5, 0.52], [0.5, 0.48], [0.51, 0.51]]], dtype=np.float32)
            return pts
        # default: simple landmark-like output
        return np.array([[[0.1, 0.1], [0.2, 0.2]]], dtype=np.float32)


def test_mediapipe_tflite_chain(monkeypatch):
    # Patch the loader's interpreter factory to use the fake interpreter
    monkeypatch.setattr(mpl, "_load_interpreter", lambda model_path, num_threads=1: FakeInterpreter(model_path, num_threads))

    from data_collection.sensors.driver_drowsiness.tflite_detector import TFLiteDrowsinessDetector

    cfg = {
        "use_mediapipe_tasks": True,
        "mediapipe": {
            "face_detection": {"model_path": "face_detection_short_range.tflite"},
            "face_landmark": {"model_path": "face_landmark_lite.tflite"},
            "iris_landmark": {"model_path": "iris_landmark.tflite"},
            "tflite_num_threads": 1,
        },
    }

    det = TFLiteDrowsinessDetector(config=cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    out = det.process_frame(frame)

    assert out.get("backend_used") == "mediapipe_tflite"
    assert out.get("no_face_detected") is False
    assert "ear" in out and isinstance(out["ear"], float)
    assert "mar" in out and isinstance(out["mar"], float)
    assert "head_pose" in out and isinstance(out["head_pose"], dict)
    assert "iris" in out and isinstance(out["iris"], dict)
