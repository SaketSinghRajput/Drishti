"""Lightweight MediaPipe TFLite task loaders.

Provides small wrappers around MediaPipe TFLite models (face detection,
face landmark, iris landmark, pose detection) using the TF-Lite
Interpreter or tflite_runtime. These are conservative helpers intended
to be used by the project's TFLite-based pipelines. They perform lazy
imports and always return safe defaults on failure.

NOTE: MediaPipe TFLite task model tensor layouts vary across versions.
This small adapter attempts to parse common output shapes. It's written
to be robust rather than fully exhaustive; if your export produces
different tensor names/shapes you may need to adapt the parsing code.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _load_interpreter(model_path: str, num_threads: int = 1):
    """Lazy load a TFLite Interpreter (tflite_runtime preferred)."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        interp = Interpreter(model_path=model_path, num_threads=num_threads)
        return interp
    except Exception:
        try:
            import tensorflow as _tf

            interp = _tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
            return interp
        except Exception:
            logger.exception("Failed to import tflite interpreter for %s", model_path)
            return None


def preprocess_for_mediapipe(frame: np.ndarray, input_size: Tuple[int, int], normalize: str = "0-1") -> np.ndarray:
    """Resize and normalize frame to model input.

    normalize: '0-1' or '-1-1'
    Returns HxWx3 float32 array.
    """
    import cv2

    h, w = input_size[1], input_size[0]
    img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img.astype(np.float32)
    if normalize == "0-1":
        img = img / 255.0
    elif normalize == "-1-1":
        img = (img / 127.5) - 1.0
    return img


class MediaPipeFaceDetector:
    """TFLite wrapper for MediaPipe face detection models.

    Minimal API:
      - __init__(model_path, config)
      - get_input_size() -> (w,h)
      - detect(frame) -> list of {bbox: (x,y,w,h) normalized, score: float}
    """

    def __init__(self, model_path: str, config: Optional[dict] = None):
        self.model_path = model_path
        self.config = config or {}
        self._interp = None
        self._input_size = tuple(self.config.get("input_size", [192, 192]))
        self._num_threads = int(self.config.get("tflite_num_threads", 1))
        try:
            self._interp = _load_interpreter(self.model_path, num_threads=self._num_threads)
            if self._interp is not None:
                self._interp.allocate_tensors()
            logger.info("MediaPipeFaceDetector loaded: %s", self.model_path)
        except Exception:
            logger.exception("Failed to load face detection model: %s", self.model_path)
            self._interp = None

    def get_input_size(self) -> Tuple[int, int]:
        return (int(self._input_size[0]), int(self._input_size[1]))

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection and return normalized boxes.

        Returns list of dicts: {'bbox': (x,y,w,h), 'score': float}
        Coordinates are normalized (0..1) relative to frame width/height.
        """
        if self._interp is None:
            return []
        try:
            inp_shape = self.get_input_size()
            img = preprocess_for_mediapipe(frame, (inp_shape[0], inp_shape[1]), normalize="0-1")
            inp = np.expand_dims(img.astype(np.float32), axis=0)
            # set input tensor
            try:
                idx = self._interp.get_input_details()[0]["index"]
                self._interp.set_tensor(idx, inp.astype(self._interp.get_input_details()[0]["dtype"]))
            except Exception:
                # fall back to positional set
                self._interp.set_tensor(0, inp)
            self._interp.invoke()
            # collect output tensors and metadata
            out_details = self._interp.get_output_details()
            outputs = {}
            for out in out_details:
                try:
                    name = out.get("name", "")
                    val = self._interp.get_tensor(out["index"])  # type: ignore
                    outputs[name] = np.asarray(val)
                except Exception:
                    continue

            # Identify boxes and scores tensors by shape and name heuristics
            boxes_arr = None
            scores_arr = None
            for name, arr in outputs.items():
                if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] >= 4:
                    # expected layout: (1, N, 4/6)
                    boxes_arr = arr
                elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] >= 4:
                    # sometimes (1, 4) or (1, N*4) - attempt reshape
                    try:
                        if arr.shape[1] % 4 == 0 and arr.shape[1] > 4:
                            boxes_arr = arr.reshape(1, -1, 4)
                    except Exception:
                        pass
                elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] <= 10:
                    # likely a scores vector (1, N)
                    if scores_arr is None and arr.size > 0:
                        scores_arr = arr.flatten()
                elif arr.ndim == 1 and arr.size > 0 and scores_arr is None:
                    scores_arr = arr

            # fallback: if boxes include score in last dim, extract it
            results = []
            frame_h, frame_w = frame.shape[:2]
            conf_thresh = float(self.config.get("face_detection_confidence", self.config.get("confidence", 0.5)))
            if boxes_arr is not None:
                arr = boxes_arr[0]
                for i, row in enumerate(arr):
                    r = np.asarray(row, dtype=float)
                    score = None
                    if r.size >= 5:
                        # common mediapipe: [ymin, xmin, ymax, xmax, score, ...]
                        ymin, xmin, ymax, xmax = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                        score = float(r[4])
                    elif r.size == 4:
                        ymin, xmin, ymax, xmax = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                    else:
                        continue
                    if scores_arr is not None and i < len(scores_arr):
                        score = float(scores_arr[i])
                    if score is None:
                        score = 1.0
                    if score < conf_thresh:
                        continue
                    w = max(0.0, xmax - xmin)
                    h = max(0.0, ymax - ymin)
                    results.append({"bbox": (xmin, ymin, w, h), "score": float(score)})
            # return filtered list
            return results
        except Exception:
            logger.exception("FaceDetector inference failed")
            return []


class MediaPipeFaceLandmarker:
    """Wrapper for face landmark models producing many landmarks.

    API: detect_landmarks(frame, face_bbox) -> list of (x,y) normalized
    """

    def __init__(self, model_path: str, config: Optional[dict] = None):
        self.model_path = model_path
        self.config = config or {}
        self._interp = _load_interpreter(self.model_path, num_threads=int(self.config.get("tflite_num_threads", 1)))
        if self._interp is not None:
            try:
                self._interp.allocate_tensors()
            except Exception:
                logger.exception("Failed to allocate tensors for face landmarker: %s", model_path)
                self._interp = None

    def detect_landmarks(self, frame: np.ndarray, face_bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        """Return list of (x,y) normalized landmarks for the face crop.

        face_bbox is (xmin, ymin, w, h) normalized to frame.
        """
        if self._interp is None:
            return []
        try:
            frame_h, frame_w = frame.shape[:2]
            xmin, ymin, w, h = face_bbox
            x0 = int(max(0, xmin * frame_w))
            y0 = int(max(0, ymin * frame_h))
            x1 = int(min(frame_w, (xmin + w) * frame_w))
            y1 = int(min(frame_h, (ymin + h) * frame_h))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                return []
            in_h = int(self._interp.get_input_details()[0]["shape"][1])
            in_w = int(self._interp.get_input_details()[0]["shape"][2])
            img = preprocess_for_mediapipe(crop, (in_w, in_h), normalize="0-1")
            inp = np.expand_dims(img.astype(np.float32), axis=0)
            self._interp.set_tensor(self._interp.get_input_details()[0]["index"], inp.astype(self._interp.get_input_details()[0]["dtype"]))
            self._interp.invoke()
            out = None
            for o in self._interp.get_output_details():
                try:
                    val = self._interp.get_tensor(o["index"])  # type: ignore
                    out = val
                    break
                except Exception:
                    continue
            if out is None:
                return []
            a = np.asarray(out)
            # expect (1,N,2) or (1,2N)
            lm = []
            if a.ndim == 3 and a.shape[2] >= 2:
                for p in a[0]:
                    lm.append((float(p[0]), float(p[1])))
            elif a.ndim == 2:
                flat = a[0]
                if flat.size % 2 == 0:
                    for i in range(0, flat.size, 2):
                        lm.append((float(flat[i]), float(flat[i + 1])))
            # convert crop-relative normalized coords back to full-frame normalized coords
            lm_full = []
            for x, y in lm:
                fx = xmin + x * w
                fy = ymin + y * h
                lm_full.append((fx, fy))
            return lm_full
        except Exception:
            logger.exception("FaceLandmarker failed")
            return []


class MediaPipeIrisLandmarker:
    """Wrapper for iris landmark models (per-eye precise landmarks).

    API: detect_iris(frame, eye_bbox) -> dict with 'center':(x,y) normalized, 'landmarks': list
    """

    def __init__(self, model_path: str, config: Optional[dict] = None):
        self.model_path = model_path
        self.config = config or {}
        self._interp = _load_interpreter(self.model_path, num_threads=int(self.config.get("tflite_num_threads", 1)))
        if self._interp is not None:
            try:
                self._interp.allocate_tensors()
            except Exception:
                logger.exception("Failed to allocate iris interpreter: %s", model_path)
                self._interp = None

    def detect_iris(self, frame: np.ndarray, eye_bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
        if self._interp is None:
            return {}
        try:
            fh, fw = frame.shape[:2]
            xmin, ymin, w, h = eye_bbox
            x0 = int(max(0, xmin * fw))
            y0 = int(max(0, ymin * fh))
            x1 = int(min(fw, (xmin + w) * fw))
            y1 = int(min(fh, (ymin + h) * fh))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                return {}
            in_h = int(self._interp.get_input_details()[0]["shape"][1])
            in_w = int(self._interp.get_input_details()[0]["shape"][2])
            img = preprocess_for_mediapipe(crop, (in_w, in_h), normalize="0-1")
            inp = np.expand_dims(img.astype(np.float32), axis=0)
            self._interp.set_tensor(self._interp.get_input_details()[0]["index"], inp.astype(self._interp.get_input_details()[0]["dtype"]))
            self._interp.invoke()
            out = None
            for o in self._interp.get_output_details():
                try:
                    out = self._interp.get_tensor(o["index"])  # type: ignore
                    break
                except Exception:
                    continue
            if out is None:
                return {}
            a = np.asarray(out)
            lm = []
            if a.ndim == 3 and a.shape[2] >= 2:
                for p in a[0]:
                    lm.append((float(p[0]), float(p[1])))
            elif a.ndim == 2:
                flat = a[0]
                if flat.size % 2 == 0:
                    for i in range(0, flat.size, 2):
                        lm.append((float(flat[i]), float(flat[i + 1])))
            # compute center in crop-normalized coords then to frame-normalized
            if not lm:
                return {}
            xs = [p[0] for p in lm]
            ys = [p[1] for p in lm]
            cx = xmin + (sum(xs) / len(xs)) * w
            cy = ymin + (sum(ys) / len(ys)) * h
            return {"center": (cx, cy), "landmarks": lm}
        except Exception:
            logger.exception("IrisLandmarker failed")
            return {}


class MediaPipePoseDetector:
    """TFLite wrapper for simple pose detection (person bbox) or pose landmarks.

    Minimal API: detect_persons(frame) -> list of {'bbox':(xmin,ymin,w,h), 'score':float}
    And detect_landmarks(frame, bbox) -> list of (x,y) normalized keypoints
    """

    def __init__(self, model_path: str, landmark_path: Optional[str] = None, config: Optional[dict] = None):
        self.model_path = model_path
        self.landmark_path = landmark_path
        self.config = config or {}
        self._det_interp = _load_interpreter(self.model_path, num_threads=int(self.config.get("tflite_num_threads", 1)))
        self._lm_interp = _load_interpreter(self.landmark_path, num_threads=int(self.config.get("tflite_num_threads", 1))) if self.landmark_path else None
        try:
            if self._det_interp is not None:
                self._det_interp.allocate_tensors()
            if self._lm_interp is not None:
                self._lm_interp.allocate_tensors()
        except Exception:
            logger.exception("Failed to allocate pose interpreters")

    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self._det_interp is None:
            return []
        try:
            in_h = int(self._det_interp.get_input_details()[0]["shape"][1])
            in_w = int(self._det_interp.get_input_details()[0]["shape"][2])
            img = preprocess_for_mediapipe(frame, (in_w, in_h), normalize="0-1")
            inp = np.expand_dims(img.astype(np.float32), axis=0)
            self._det_interp.set_tensor(self._det_interp.get_input_details()[0]["index"], inp.astype(self._det_interp.get_input_details()[0]["dtype"]))
            self._det_interp.invoke()
            out = None
            for o in self._det_interp.get_output_details():
                try:
                    out = self._det_interp.get_tensor(o["index"])  # type: ignore
                    break
                except Exception:
                    continue
            if out is None:
                return []
            a = np.asarray(out)
            res = []
            if a.ndim >= 2:
                arr = a[0]
                for row in arr:
                    r = np.asarray(row, dtype=float)
                    if r.size >= 4:
                        ymin, xmin, ymax, xmax = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                        score = float(r[4]) if r.size > 4 else 1.0
                        w = max(0.0, xmax - xmin)
                        h = max(0.0, ymax - ymin)
                        res.append({"bbox": (xmin, ymin, w, h), "score": score})
            return res
        except Exception:
            logger.exception("Pose person detection failed")
            return []

    def detect_landmarks(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        if self._lm_interp is None:
            return []
        try:
            fh, fw = frame.shape[:2]
            xmin, ymin, w, h = bbox
            x0 = int(max(0, xmin * fw))
            y0 = int(max(0, ymin * fh))
            x1 = int(min(fw, (xmin + w) * fw))
            y1 = int(min(fh, (ymin + h) * fh))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                return []
            in_h = int(self._lm_interp.get_input_details()[0]["shape"][1])
            in_w = int(self._lm_interp.get_input_details()[0]["shape"][2])
            img = preprocess_for_mediapipe(crop, (in_w, in_h), normalize="0-1")
            inp = np.expand_dims(img.astype(np.float32), axis=0)
            self._lm_interp.set_tensor(self._lm_interp.get_input_details()[0]["index"], inp.astype(self._lm_interp.get_input_details()[0]["dtype"]))
            self._lm_interp.invoke()
            out = None
            for o in self._lm_interp.get_output_details():
                try:
                    out = self._lm_interp.get_tensor(o["index"])  # type: ignore
                    break
                except Exception:
                    continue
            if out is None:
                return []
            a = np.asarray(out)
            lm = []
            if a.ndim == 3 and a.shape[2] >= 2:
                for p in a[0]:
                    lm.append((float(p[0]), float(p[1])))
            elif a.ndim == 2:
                flat = a[0]
                if flat.size % 2 == 0:
                    for i in range(0, flat.size, 2):
                        lm.append((float(flat[i]), float(flat[i + 1])))
            # map to full-frame normalized coords
            lm_full = []
            for x, y in lm:
                fx = xmin + x * w
                fy = ymin + y * h
                lm_full.append((fx, fy))
            return lm_full
        except Exception:
            logger.exception("Pose landmark detection failed")
            return []
