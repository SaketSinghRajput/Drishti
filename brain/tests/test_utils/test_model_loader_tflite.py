import numpy as np
from brain.utils import model_loader_tflite


class FakeLoader:
    def __init__(self):
        self.called = False

    def detect(self, frame):
        self.called = True
        h, w = frame.shape[:2]
        # simple normalized bbox centered
        return [{"bbox": (0.25, 0.25, 0.5, 0.5), "score": 0.87}]


def test_predict_delegates_to_mediapipe_loader():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    model = model_loader_tflite.TFLiteDetectionModel("fake.tflite", config={})
    # inject fake mediapipe loader
    fake = FakeLoader()
    model._mediapipe_loader = fake

    res = model.predict(img)
    assert isinstance(res, list)
    assert len(res) == 1
    d = res[0]
    assert 'bbox_pixels' in d
    assert 'confidence' in d
    assert d['confidence'] == 0.87
    # bbox should be pixel coords
    x1, y1, x2, y2 = d['bbox_pixels']
    assert x2 > x1 and y2 > y1
    assert fake.called
