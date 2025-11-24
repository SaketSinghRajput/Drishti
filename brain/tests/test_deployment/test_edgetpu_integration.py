import pytest

from inference.model_loaders import TFLiteRiskModel


def test_edgetpu_load_fallback():
    # If TFLite/TF not available, skip
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        pytest.skip("TensorFlow/tflite not available")

    # Create a model loader with edgetpu enabled; we expect graceful fallback
    m = TFLiteRiskModel(path="nonexistent.tflite", use_edgetpu=True)
    with pytest.raises(FileNotFoundError):
        m.load()
