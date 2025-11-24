"""YOLO detection model wrapper.

Attempts to load ultralytics first, falls back to torch.hub yolov5. Returns
List[Dict] with keys 'class', 'confidence', 'bbox_pixels'. Applies confidence
filtering and returns human-readable COCO names when possible.
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


class YOLODetectionModel(BaseDetectionModel):
    def __init__(self, model_path: Optional[str] = None, config: Optional[dict] = None, classes: Optional[List[str]] = None):
        try:
            super().__init__(model_path, classes=classes)
        except Exception:
            self.model_path = model_path
            self.classes = classes
        self.config = config or {}
        self.model = None

    def load(self) -> None:
        # Try ultralytics first
        try:
            from ultralytics import YOLO  # type: ignore

            if self.model_path:
                self.model = YOLO(self.model_path)
            else:
                # load small default
                self.model = YOLO('yolov8n.pt')
            return
        except Exception as e:
            logger.debug("ultralytics not available or failed: %s", e)

        # Fallback to torch.hub loading yolov5 repo
        try:
            import torch

            repo = 'ultralytics/yolov5'
            model_name = self.model_path or 'yolov5s'
            # if model_path provided we load as custom
            if self.model_path:
                self.model = torch.hub.load(repo, 'custom', path=self.model_path, source='github')
            else:
                self.model = torch.hub.load(repo, model_name, pretrained=True)
        except Exception as e:
            logger.error("Failed to load YOLO model via ultralytics or torch.hub: %s", e)
            raise

    def predict(self, image: 'np.ndarray') -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        from utils import camera_utils as _cu
        from . import model_loader as _ml
        import time as _time
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

        try:
            if self.model is None:
                raise RuntimeError("YOLO model not loaded")
            # ultralytics
            if hasattr(self.model, '__call__'):
                try:
                    start = _time.time()
                    res = self.model(image)
                    elapsed = (_time.time() - start) * 1000.0
                    logger.debug("YOLO (ultralytics) inference time: %.2f ms", elapsed)
                except TypeError:
                    start = _time.time()
                    res = self.model.predict(image)
                    elapsed = (_time.time() - start) * 1000.0
                    logger.debug("YOLO (ultralytics) inference time: %.2f ms", elapsed)
                # res may be iterable; pick first
                r0 = res[0] if hasattr(res, '__len__') and len(res) > 0 else res
                boxes = getattr(r0, 'boxes', None)
                if boxes is not None:
                    for b in boxes:
                        try:
                            conf = float(getattr(b, 'conf', getattr(b, 'confidence', 1.0)))
                        except Exception:
                            conf = 1.0
                        cls = int(getattr(b, 'cls', 0))
                        # try to get xyxy list
                        xyxy = None
                        if hasattr(b, 'xyxy'):
                            try:
                                xyxy = b.xyxy.tolist()
                            except Exception:
                                xyxy = list(b.xyxy)
                        elif hasattr(b, 'xyxyn'):
                            try:
                                xyxy = b.xyxyn.tolist()
                            except Exception:
                                xyxy = list(b.xyxyn)
                        if xyxy is None:
                            continue
                        bbox = [int(x) for x in xyxy[:4]]
                        results.append({"class": str(cls), "confidence": conf, "bbox_pixels": bbox})
                    # postprocess
                    results = _cu.filter_low_confidence_detections(results, _safe_float_cfg(self.config.get('detection_confidence_threshold', 0.5)))
                    results = _cu.nms_detections(results, _safe_float_cfg(self.config.get('nms_iou_threshold', 0.5)))
                    if self.classes is None:
                        try:
                            self.classes = _ml.get_coco_classes()
                        except Exception:
                            self.classes = None
                    if self.classes:
                        for d in results:
                            try:
                                idx = int(d.get('class', 0))
                                if 0 <= idx < len(self.classes):
                                    d['class'] = self.classes[idx]
                            except Exception:
                                pass
                    return results

            # torch.hub YOLOv5 -> model returns a results object with .xyxy[0]
            if hasattr(self.model, 'forward') or hasattr(self.model, 'train') or hasattr(self.model, 'autoshape'):
                out = self.model(image)
                xyxy = None
                try:
                    xyxy = out.xyxy[0].cpu().numpy()
                except Exception:
                    try:
                        xyxy = out.xyxy[0].numpy()
                    except Exception:
                        pass
                if xyxy is not None:
                    for row in xyxy:
                        x1, y1, x2, y2, conf, cls = row[:6]
                        if conf < float(self.config.get('detection_confidence_threshold', 0.5)):
                            continue
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        results.append({"class": str(int(cls)), "confidence": float(conf), "bbox_pixels": bbox})
                    results = _cu.nms_detections(results, _safe_float_cfg(self.config.get('nms_iou_threshold', 0.5)))
                    if self.classes is None:
                        try:
                            self.classes = _ml.get_coco_classes()
                        except Exception:
                            self.classes = None
                    if self.classes:
                        for d in results:
                            try:
                                idx = int(d.get('class', 0))
                                if 0 <= idx < len(self.classes):
                                    d['class'] = self.classes[idx]
                            except Exception:
                                pass
                    return results
        except Exception:
            logger.exception("YOLO prediction failed")
            return []

        return results
