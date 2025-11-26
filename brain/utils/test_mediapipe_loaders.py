import numpy as np
from brain.utils.mediapipe_loaders import MediaPipeFaceDetector, MediaPipeFaceLandmarker, MediaPipeIrisLandmarker


def test_mediapipe_face_detector_no_model():
    fd = MediaPipeFaceDetector(model_path="/non/existent.tflite", config={})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = fd.detect(frame)
    assert isinstance(res, list)


def test_mediapipe_landmarker_no_model():
    lm = MediaPipeFaceLandmarker(model_path="/non/existent.tflite", config={})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = lm.detect_landmarks(frame, (0.1, 0.1, 0.5, 0.5))
    assert isinstance(res, list)


def test_iris_no_model():
    im = MediaPipeIrisLandmarker(model_path="/non/existent.tflite", config={})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = im.detect_iris(frame, (0.4, 0.4, 0.1, 0.1))
    assert isinstance(res, dict)
