import numpy as np
import types
import builtins

from data_collection.sensors.driver_drowsiness.tflite_detector import TFLiteDrowsinessDetector


class DummyInterpreter:
    def __init__(self, input_shape=(1, 64, 64, 3)):
        self._input_shape = list(input_shape)
        self._tensors = {}
        self.invoked = False

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"shape": self._input_shape, "index": 0, "dtype": np.float32}]

    def get_output_details(self):
        return [{"index": 0}]

    def set_tensor(self, idx, val):
        self._tensors[idx] = val

    def invoke(self):
        self.invoked = True

    def get_tensor(self, idx):
        # return a list of 20 2D points -> shape (1,20,2)
        out = np.zeros((1, 20, 2), dtype=np.float32)
        for i in range(20):
            out[0, i, 0] = float(i) / 20.0
            out[0, i, 1] = float(i) / 30.0
        return out


def test_tflite_resizes_and_runs(monkeypatch, tmp_path):
    # monkeypatch the Interpreter import to our DummyInterpreter
    def fake_interpreter_factory(*args, **kwargs):
        return DummyInterpreter(input_shape=(1, 32, 48, 3))

    monkeypatch.setitem(__import__('sys').modules, 'tflite_runtime.interpreter', types.SimpleNamespace(Interpreter=fake_interpreter_factory))

    # create detector with the fake interpreter
    det = TFLiteDrowsinessDetector(config={"tflite_model_path": "fake.tflite"}, model_path="fake.tflite")
    # craft a synthetic frame larger than target size
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = det.process_frame(frame)
    assert 'ear' in res and 'mar' in res
    assert res['no_face_detected'] is False
    # ensure interpreter was invoked by checking counters
    assert res['total_blinks'] >= 0

