import pytest
import numpy as np
import types
from _pytest.monkeypatch import MonkeyPatch
from pathlib import Path

from brain.utils.mediapipe_hand_landmarker import MediaPipeHandLandmarker


class FakeInterpreter:
    def __init__(self, ok=True):
        self._ok = ok
        self._tensor = None
        self._input = None
        self._invoked = False

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{'index': 0, 'shape': [1, 224, 224, 3]}]

    def get_output_details(self):
        return [{'index': 0}]

    def set_tensor(self, idx, val):
        self._input = val

    def get_tensor(self, idx):
        # return shape (1,21,2)
        return np.zeros((1, 21, 2), dtype=np.float32)

    def invoke(self):
        if not self._ok:
            raise RuntimeError('invoke failed')
        self._invoked = True


@pytest.fixture
def frame() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_init_success(monkeypatch: MonkeyPatch, tmp_path: Path):
    # patch tflite interpreter creation
    import brain.utils.mediapipe_hand_landmarker as mdl

    monkeypatch.setattr(mdl, 'tflite', types.SimpleNamespace(Interpreter=lambda model_path: FakeInterpreter()))
    p = tmp_path / 'm.tflite'
    p.write_bytes(b'fake')
    m = MediaPipeHandLandmarker(str(p))
    assert m.is_available()


def test_init_failure(monkeypatch: MonkeyPatch, tmp_path: Path):
    import brain.utils.mediapipe_hand_landmarker as mdl
    def bad_interp(path):
        raise RuntimeError('bad')
    monkeypatch.setattr(mdl, 'tflite', types.SimpleNamespace(Interpreter=bad_interp))
    p = tmp_path / 'm.tflite'
    p.write_bytes(b'fake')
    m = MediaPipeHandLandmarker(str(p))
    assert not m.is_available()


def test_detect_hand_landmarks_success(monkeypatch: MonkeyPatch, frame: np.ndarray, tmp_path: Path):
    import brain.utils.mediapipe_hand_landmarker as mdl
    class OkInterp(FakeInterpreter):
        def get_tensor(self, idx):
            # return normalized keypoints spread in crop coords
            arr = np.zeros((1, 21, 2), dtype=np.float32)
            for i in range(21):
                arr[0, i, 0] = 0.5
                arr[0, i, 1] = 0.5
            return arr
    monkeypatch.setattr(mdl, 'tflite', types.SimpleNamespace(Interpreter=lambda model_path: OkInterp()))
    p = tmp_path / 'm.tflite'
    p.write_bytes(b'fake')
    m = MediaPipeHandLandmarker(str(p))
    kps = m.detect_hand_landmarks(frame, (0.2, 0.2, 0.2, 0.2))
    assert isinstance(kps, list)
    assert len(kps) == 21


def test_detect_hand_landmarks_empty_crop(monkeypatch: MonkeyPatch, frame: np.ndarray, tmp_path: Path):
    import brain.utils.mediapipe_hand_landmarker as mdl
    monkeypatch.setattr(mdl, 'tflite', types.SimpleNamespace(Interpreter=lambda model_path: FakeInterpreter()))
    p = tmp_path / 'm.tflite'
    p.write_bytes(b'fake')
    m = MediaPipeHandLandmarker(str(p))
    # bbox outside frame
    kps = m.detect_hand_landmarks(frame, (10.0, 10.0, 0.1, 0.1))
    assert kps == []


def test_detect_hand_landmarks_invoke_failure(monkeypatch: MonkeyPatch, frame: np.ndarray, tmp_path: Path):
    import brain.utils.mediapipe_hand_landmarker as mdl
    class BadInterp(FakeInterpreter):
        def invoke(self):
            raise RuntimeError('boom')
    monkeypatch.setattr(mdl, 'tflite', types.SimpleNamespace(Interpreter=lambda model_path: BadInterp()))
    p = tmp_path / 'm.tflite'
    p.write_bytes(b'fake')
    m = MediaPipeHandLandmarker(str(p))
    kps = m.detect_hand_landmarks(frame, (0.0, 0.0, 1.0, 1.0))
    assert kps == []


def test_keypoint_normalization(monkeypatch: MonkeyPatch, frame: np.ndarray, tmp_path: Path):
    import brain.utils.mediapipe_hand_landmarker as mdl
    class CInterp(FakeInterpreter):
        def get_tensor(self, idx):
            arr = np.zeros((1, 21, 2), dtype=np.float32)
            # single point at (1.0, 1.0) in crop coordinate -> maps to bottom-right of crop
            arr[0, 0, 0] = 1.0
            arr[0, 0, 1] = 1.0
            return arr
    monkeypatch.setattr(mdl, 'tflite', types.SimpleNamespace(Interpreter=lambda model_path: CInterp()))
    p = tmp_path / 'm.tflite'
    p.write_bytes(b'fake')
    m = MediaPipeHandLandmarker(str(p))
    # Use central crop
    bbox = (0.25, 0.25, 0.5, 0.5)
    kps = m.detect_hand_landmarks(frame, bbox)
    assert len(kps) >= 1
    fx, fy = kps[0]
    # The point should lie within bbox bounds (normalized)
    assert 0.25 <= fx <= 0.75
    assert 0.25 <= fy <= 0.75
