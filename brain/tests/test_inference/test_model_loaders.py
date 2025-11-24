import os
import tempfile
import numpy as np
import pytest
from unittest import mock

from inference.model_loaders import BaseRiskModel, XGBoostRiskModel, TFLiteRiskModel, ONNXRiskModel, load_risk_model


def test_base_interface_raises():
    m = BaseRiskModel(path="/tmp/notexist")
    with pytest.raises(NotImplementedError):
        m.load()


def test_xgboost_model_mocked():
    # Inject fake xgboost module into sys.modules to satisfy lazy imports
    fake_xgb = mock.Mock()
    booster = mock.Mock()
    booster.predict.return_value = np.array([0.42], dtype=np.float32)
    fake_xgb.Booster.return_value = booster

    # Capture the DMatrix input to validate ordering
    captured = {}

    def fake_dmatrix(arr, feature_names=None):
        captured['arr'] = np.asarray(arr)
        return mock.Mock()

    fake_xgb.DMatrix.side_effect = fake_dmatrix

    with mock.patch.dict('sys.modules', {'xgboost': fake_xgb}):
        model = XGBoostRiskModel(path="/tmp/fake.xgb", feature_names=["f1", "f2"]) 
        # Mock os.path.exists
        with mock.patch("os.path.exists", return_value=True):
            model.load()
            out = model.predict({"f2": 2.0, "f1": 1.0})
            # Ensure prediction returns float and DMatrix received aligned array [1.0,2.0]
            assert isinstance(out, float)
            assert 0.0 <= out <= 1.0
            assert 'arr' in captured
            assert captured['arr'].shape[1] == 2
            # first column should be f1 then f2
            assert captured['arr'][0, 0] == 1.0
            assert captured['arr'][0, 1] == 2.0


def test_tflite_model_mocked():
    # Create fake tensorflow with dummy lite.Interpreter
    fake_tf = mock.Mock()
    interp = mock.Mock()
    interp.get_input_details.return_value = [{"index": 0, "dtype": np.float32, "quantization": (None, None)}]
    interp.get_output_details.return_value = [{"index": 0}]
    interp.get_tensor.return_value = np.array([[0.33]])
    interp.set_tensor = mock.Mock()
    fake_tf.lite = mock.Mock()
    fake_tf.lite.Interpreter.return_value = interp

    with mock.patch.dict('sys.modules', {'tensorflow': fake_tf}):
        # Provide feature_names to enforce ordering
        model = TFLiteRiskModel(path="/tmp/fake.tflite", feature_names=["b", "a"])
        with mock.patch("os.path.exists", return_value=True):
            model.load()
            # Ensure interpreter has been set
            model._interpreter = interp
            out = model.predict({"a": 1.0, "b": 2.0})
            assert isinstance(out, float)
            assert 0.0 <= out <= 1.0
            # Verify set_tensor called with aligned array [b,a]
            args, kwargs = interp.set_tensor.call_args
            sent = np.asarray(args[1])
            assert sent.shape[1] == 2
            assert sent[0, 0] == 2.0
            assert sent[0, 1] == 1.0


def test_onnx_model_mocked():
    fake_ort = mock.Mock()
    sess = mock.Mock()
    inp = mock.Mock()
    inp.name = "input"
    sess.get_inputs.return_value = [inp]
    sess.run.return_value = [np.array([0.88])]
    fake_ort.InferenceSession.return_value = sess

    with mock.patch.dict('sys.modules', {'onnxruntime': fake_ort}):
        model = ONNXRiskModel(path="/tmp/fake.onnx", feature_names=["x", "y"]) 
        with mock.patch("os.path.exists", return_value=True):
            model.load()
            model._session = sess
            out = model.predict({"y": 2.0, "x": 1.0})
            assert isinstance(out, float)
            assert 0.0 <= out <= 1.0
            # ensure session.run was called with aligned input [x,y]
            called_args = sess.run.call_args[0][1]
            # dict with input name key
            arr = list(called_args.values())[0]
            assert arr.shape[1] == 2
            assert arr[0, 0] == 1.0
            assert arr[0, 1] == 2.0


def test_load_risk_model_factory_detects_extensions():
    m = load_risk_model("model.xgb")
    assert isinstance(m, XGBoostRiskModel)
    m = load_risk_model("model.tflite")
    assert isinstance(m, TFLiteRiskModel)
    m = load_risk_model("model.onnx")
    assert isinstance(m, ONNXRiskModel)

    with pytest.raises(ValueError):
        load_risk_model("model.unknown")