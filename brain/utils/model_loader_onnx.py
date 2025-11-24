"""ONNX Runtime detection model wrapper.

Subclass-compatible adapter for ONNXRuntime. Performs lazy imports, basic
preprocessing (resize/normalize), confidence filtering and NMS, and returns
the canonical detection dicts expected by the camera pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model_loader import BaseDetectionModel, get_coco_classes
    import numpy as np
else:
    try:
        from .model_loader import BaseDetectionModel
    except Exception:
        BaseDetectionModel = object


class ONNXDetectionModel(BaseDetectionModel):
    def __init__(self, model_path: str, config: Optional[dict] = None, classes: Optional[List[str]] = None):
        try:
            super().__init__(model_path, classes=classes)
        except Exception:
            self.model_path = model_path
            self.classes = classes
        self.config = config or {}
        self.session = None
        self.input_name = None

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except Exception as e:
            logger.error("onnxruntime not available: %s", e)
            raise
        providers = [self.config.get("onnx_execution_provider", "CPUExecutionProvider")]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name

    def get_input_size(self) -> (int, int):
        if self.session is None:
            return (640, 640)
        inp = self.session.get_inputs()[0]
        shape = inp.shape
        if len(shape) >= 4:
            return (int(shape[2]), int(shape[3]))
        return (640, 640)

    def predict(self, image: 'np.ndarray') -> List[Dict[str, Any]]:
        if self.session is None:
            raise RuntimeError("ONNX session not loaded")
        import numpy as _np
        import cv2 as _cv2
        from . import model_loader as _ml
        from utils import camera_utils as _cu
        import time as _time

        h, w = self.get_input_size()
        img = _cv2.resize(image, (w, h))
        # BGR->RGB
        if img.shape[2] == 3:
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        img = img.astype(_np.float32) / 255.0

        # attempt NCHW if model expects it
        inp = self.session.get_inputs()[0]
        if len(inp.shape) == 4 and inp.shape[1] in (1, 3):
            # model expects NCHW
            inp_data = _np.expand_dims(img.transpose(2, 0, 1), axis=0)
        else:
            inp_data = _np.expand_dims(img, axis=0)

        try:
            start = _time.time()
            outputs = self.session.run(None, {self.input_name: inp_data})
            elapsed = (_time.time() - start) * 1000.0
            logger.debug("ONNX inference time: %.2f ms", elapsed)
        except Exception as e:
            logger.exception("ONNX inference failed: %s", e)
            return []

        # Heuristic parse
        boxes = None
        scores = None
        classes = None
        try:
            if len(outputs) >= 3:
                boxes, scores, classes = outputs[0], outputs[1], outputs[2]
            elif len(outputs) == 2:
                boxes, scores = outputs[0], outputs[1]
                classes = None
            else:
                boxes = outputs[0]
        except Exception:
            return []

        def _safe_float_cfg(v, default: float = 0.5) -> float:
            try:
                return float(v)
            except Exception:
                try:
                    s = str(v)
                    import re

                    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", s)
                    if m:
                        return float(m.group(1))
                except Exception:
                    pass
                return float(default)

        results: List[Dict[str, Any]] = []
        try:
            boxes_arr = _np.array(boxes)
            if boxes_arr.ndim == 3:
                boxes_arr = boxes_arr[0]
            scores_arr = _np.array(scores) if scores is not None else None
            classes_arr = _np.array(classes) if classes is not None else None
            for i in range(len(boxes_arr)):
                score = float(scores_arr[i]) if scores_arr is not None else 1.0
                if score < float(self.config.get("detection_confidence_threshold", 0.5)):
                    continue
                cls_idx = int(classes_arr[i]) if classes_arr is not None else 0
                cls_name = str(cls_idx)
                box = boxes_arr[i]
                if box.max() <= 1.0:
                    h_img, w_img = image.shape[0], image.shape[1]
                    y1, x1, y2, x2 = box
                    bbox = [int(x1 * w_img), int(y1 * h_img), int(x2 * w_img), int(y2 * h_img)]
                else:
                    bbox = [int(float(x)) for x in box]
                results.append({"class": cls_name, "confidence": float(score), "bbox_pixels": bbox})
        except Exception:
            return []

        # filter + NMS
    results = _cu.filter_low_confidence_detections(results, _safe_float_cfg(self.config.get("detection_confidence_threshold", 0.5)))
    results = _cu.nms_detections(results, _safe_float_cfg(self.config.get("nms_iou_threshold", 0.5)))

        if self.classes is None:
            try:
                self.classes = _ml.get_coco_classes()
            except Exception:
                self.classes = None

        if self.classes:
            for d in results:
                try:
                    idx = int(d.get("class", 0))
                    if 0 <= idx < len(self.classes):
                        d["class"] = self.classes[idx]
                except Exception:
                    pass

        return results
