"""TFLite detection model wrapper.

Subclass of BaseDetectionModel that wraps tf.lite.Interpreter. The adapter is
resilient: it performs lazy imports, handles uint8/float inputs, applies
confidence filtering and NMS, and returns normalized detection dicts matching
the project's handler expectations.
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
    # at runtime we can import BaseDetectionModel when needed in __init__
    try:
        from .model_loader import BaseDetectionModel
    except Exception:
        BaseDetectionModel = object  # fallback for static import order


class TFLiteDetectionModel(BaseDetectionModel):
    """Lightweight TFLite adapter.

    Note: we avoid importing TensorFlow at module-import time to keep the project
    runnable in environments without TF. load() performs the lazy import.
    """

    def __init__(self, model_path: str, config: Optional[dict] = None, classes: Optional[List[str]] = None):
        # keep signature compatible with factory usage
        # initialize BaseDetectionModel to get classes defaulting
        try:
            super().__init__(model_path, classes=classes)
        except Exception:
            # if BaseDetectionModel isn't available at import time, set attributes directly
            self.model_path = model_path
            self.classes = classes
        self.config = config or {}
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        # ensure classes set
        if getattr(self, 'classes', None) is None:
            self.classes = classes

    def load(self) -> None:
        try:
            import tensorflow as tf
        except Exception as e:
            logger.error("TensorFlow not available: %s", e)
            raise
        # allocate interpreter
        num_threads = int(self.config.get("tflite_num_threads", 1))
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_input_size(self) -> (int, int):
        if not self.input_details:
            return (300, 300)
        shape = self.input_details[0]["shape"]
        # shape may be [1,h,w,3] or [1,3,h,w]
        if len(shape) == 4:
            return (int(shape[1]), int(shape[2]))
        return (int(shape[1]), int(shape[2]))

    def predict(self, image: 'np.ndarray') -> List[Dict[str, Any]]:
        if self.interpreter is None:
            raise RuntimeError("Interpreter not loaded")
        import numpy as _np
        import cv2 as _cv2
        from . import model_loader as _ml
        from utils import camera_utils as _cu
        import time as _time

        in_h, in_w = self.get_input_size()
        img = _cv2.resize(image, (in_w, in_h))
        # convert BGR->RGB if likely
        if img.shape[2] == 3:
            # model authors often expect RGB; this heuristic is safe for camera inputs
            img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        # handle quantized inputs
        inp_dtype = self.input_details[0]["dtype"]
        if inp_dtype == _np.uint8:
            # convert to uint8 [0,255]
            if img.dtype != _np.uint8:
                img_in = (img * 255.0).astype(_np.uint8)
        else:
            # float inputs normalized to [0,1]
            img_in = img.astype(_np.float32) / 255.0

        # ensure NHWC
        input_data = _np.expand_dims(img_in, axis=0)
        try:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            start = _time.time()
            self.interpreter.invoke()
            elapsed = ( _time.time() - start ) * 1000.0
            logger.debug("TFLite inference time: %.2f ms", elapsed)
        except Exception as e:
            logger.exception("TFLite invoke failed: %s", e)
            return []

        # collect outputs
        boxes = None
        scores = None
        classes = None
        for out in self.output_details:
            name = out.get("name", "").lower()
            try:
                val = self.interpreter.get_tensor(out["index"])
            except Exception:
                val = None
            if "box" in name or "bbox" in name:
                boxes = val
            elif "score" in name or "prob" in name:
                scores = val
            elif "class" in name or "label" in name:
                classes = val

        # positional fallback
        if boxes is None and len(self.output_details) >= 1:
            try:
                boxes = self.interpreter.get_tensor(self.output_details[0]["index"])
            except Exception:
                boxes = None
        if scores is None and len(self.output_details) >= 2:
            try:
                scores = self.interpreter.get_tensor(self.output_details[1]["index"])
            except Exception:
                scores = None
        if classes is None and len(self.output_details) >= 3:
            try:
                classes = self.interpreter.get_tensor(self.output_details[2]["index"])
            except Exception:
                classes = None

        if boxes is None:
            return []

        boxes_arr = _np.array(boxes)
        if boxes_arr.ndim == 3:
            boxes_arr = boxes_arr[0]
        scores_arr = _np.array(scores) if scores is not None else None
        classes_arr = _np.array(classes) if classes is not None else None

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

        dets: List[Dict[str, Any]] = []
        for i in range(len(boxes_arr)):
            try:
                score = float(scores_arr[0][i]) if scores_arr is not None and scores_arr.ndim == 2 else (float(scores_arr[i]) if scores_arr is not None else 1.0)
            except Exception:
                score = 1.0
            if score < float(self.config.get("detection_confidence_threshold", 0.5)):
                continue
            try:
                cls_idx = int(classes_arr[0][i]) if classes_arr is not None and classes_arr.ndim == 2 else (int(classes_arr[i]) if classes_arr is not None else 0)
            except Exception:
                cls_idx = 0
            cls_name = str(cls_idx)
            box = boxes_arr[i]
            # convert normalized y1,x1,y2,x2 to pixels
            if box.max() <= 1.0:
                h_img, w_img = image.shape[0], image.shape[1]
                y1, x1, y2, x2 = box
                bbox = [int(x1 * w_img), int(y1 * h_img), int(x2 * w_img), int(y2 * h_img)]
            else:
                bbox = [int(float(x)) for x in box]
            dets.append({"class": cls_name, "confidence": float(score), "bbox_pixels": bbox})

        # apply confidence filtering and NMS
    dets = _cu.filter_low_confidence_detections(dets, _safe_float_cfg(self.config.get("detection_confidence_threshold", 0.5)))
    dets = _cu.nms_detections(dets, _safe_float_cfg(self.config.get("nms_iou_threshold", 0.5)))

        # map class indices to names if available in factory
        if self.classes is None:
            try:
                self.classes = _ml.get_coco_classes()
            except Exception:
                self.classes = None

        if self.classes:
            for d in dets:
                try:
                    idx = int(d.get("class", 0))
                    if 0 <= idx < len(self.classes):
                        d["class"] = self.classes[idx]
                except Exception:
                    pass

        return dets
