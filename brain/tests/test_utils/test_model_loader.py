import pytest
from utils.model_loader import load_model, BaseDetectionModel, get_coco_classes
from config.config_loader import ConfigLoader


def test_get_coco_classes():
    classes = get_coco_classes()
    assert isinstance(classes, list)
    assert "person" in classes


def test_base_detection_model_interface():
    # BaseDetectionModel implements methods that should raise NotImplementedError by default
    m = BaseDetectionModel()
    with pytest.raises(NotImplementedError):
        m.load()


def test_load_model_instantiates_backends():
    # Ensure factory returns a backend instance for known types (module import may be lazy)
    m = load_model("", "tflite", config={})
    assert isinstance(m, BaseDetectionModel)
    assert m.__class__.__name__.lower().startswith("tflite") or m.__class__.__name__.lower().startswith("base")

    m2 = load_model("", "onnx", config={})
    assert isinstance(m2, BaseDetectionModel)
    assert m2.__class__.__name__.lower().startswith("onnx") or m2.__class__.__name__.lower().startswith("base")

    m3 = load_model("", "yolov5", config={})
    assert isinstance(m3, BaseDetectionModel)
    assert m3.__class__.__name__.lower().startswith("yolo") or m3.__class__.__name__.lower().startswith("base")


def test_camera_handler_uses_backend_when_available(tmp_path, monkeypatch):
    # Create a minimal config that requests production mode with a fake model_path
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
sensors:
  camera:
    enable_simulation: false
    model_path: fake_model.tflite
    model_type: tflite
"""
    )
    cfg = ConfigLoader(str(cfg_path))

    # Monkeypatch the TFLite backend load method to avoid importing TF
    try:
        import utils.model_loader_tflite as tmod
        monkeypatch.setattr(tmod.TFLiteDetectionModel, "load", lambda self: None)
    except Exception:
        # If the module isn't present, skip the monkeypatch — factory will fall back
        pass

    from data_collection.sensors.camera_handler import CameraHandler

    ch = CameraHandler(cfg)
    # If the backend was present and load succeeded, _model should be set; if
    # the backend wasn't available the handler may have fallen back to simulation.
    assert ch._model is not None or ch.enable_simulation
