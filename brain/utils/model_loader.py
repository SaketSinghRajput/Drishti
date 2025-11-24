"""Model loading utilities for TFLite, ONNX, and YOLO models.

This module defines the BaseDetectionModel contract and a small factory to
produce concrete backend adapters. Heavy ML packages are lazily imported so
optional dependencies don't crash the application when not installed.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Any, TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class BaseDetectionModel:
    """Abstract detection model interface.

    Subclasses should implement load() and predict(). predict() returns a list of
    dicts with keys: {"class": str, "confidence": float, "bbox_pixels": [x1,y1,x2,y2]}
    """

    def __init__(self, model_path: str = "", classes: Optional[List[str]] = None):
        self.model_path = model_path
        self.classes = classes or get_coco_classes()

    def load(self) -> None:
        raise NotImplementedError()

    def predict(self, image: "np.ndarray") -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def get_input_size(self) -> Tuple[int, int]:
        return (640, 480)

    def get_classes(self) -> List[str]:
        return list(self.classes)


def get_coco_classes() -> List[str]:
    # Minimal COCO subset for runtime simplicity. Users may replace or extend.
    return [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic_light",
        "fire_hydrant",
        "stop_sign",
        "parking_meter",
        "bench",
        # ... expand as needed
    ]


def load_model(model_path: str, model_type: str, config: Optional[dict] = None) -> BaseDetectionModel:
    """Factory to instantiate a detection model.

    model_type: 'tflite', 'onnx', 'yolov5', or free-text names like 'MobileNet-SSD'
    """
    cfg = config or {}
    mtype = (model_type or "").lower()
    if mtype in ("tflite", "mobilenet-ssd", "mobile net-ssd", "mobilenet_ssd"):
        try:
            from . import model_loader_tflite as tfl

            return tfl.TFLiteDetectionModel(model_path, config=cfg)
        except Exception as e:
            # If a model path was supplied and the user explicitly requested TFLite,
            # it's likely they expect real inference — surface the error so callers
            # can decide. If no model_path was provided, fall back to BaseDetectionModel.
            logger.debug("TFLite backend load failed: %s", e)
            if model_path:
                raise ValueError("TFLite backend requested but not available or failed to import: %s" % e)
    if mtype in ("onnx", "onnxruntime"):
        try:
            from . import model_loader_onnx as mto

            return mto.ONNXDetectionModel(model_path, config=cfg)
        except Exception as e:
            logger.debug("ONNX backend load failed: %s", e)
            if model_path:
                raise ValueError("ONNX backend requested but not available or failed to import: %s" % e)
    if mtype.startswith("yolov") or mtype in ("yolov5", "yolov5s", "yolov5n"):
        try:
            from . import model_loader_yolo as yol

            return yol.YOLODetectionModel(model_path, config=cfg)
        except Exception as e:
            logger.debug("YOLO backend load failed: %s", e)
            if model_path:
                raise ValueError("YOLO backend requested but not available or failed to import: %s" % e)

    # Fallback: return a BaseDetectionModel that provides no predictions
    logger.info("No specialized model backend available for '%s' — using BaseDetectionModel (no-op)", model_type)
    return BaseDetectionModel(model_path=model_path)
