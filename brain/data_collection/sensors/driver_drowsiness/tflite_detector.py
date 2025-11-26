"""TFLite-based drowsiness detector wrapper.

This module provides a lightweight TFLite-backed detector that attempts to
load a TFLite interpreter (either `tflite_runtime` or `tensorflow.lite`) and
run inference to produce landmarks which are then translated into EAR/MAR
and head-pose measurements using existing utilities. If no interpreter is
available the detector will log a warning and return safe fallbacks.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import numpy as np
import cv2

from . import utils

logger = logging.getLogger(__name__)


class TFLiteDrowsinessDetector:
    def __init__(self, config: Optional[dict] = None, model_path: Optional[str] = None):
        self.config = config or {}
        self._lock = threading.Lock()
        self._model_path = model_path or self.config.get("tflite_model_path")
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._available = False

        # whether to prefer MediaPipe TFLite task pipeline
        self.use_mediapipe_tasks = bool(self.config.get("use_mediapipe_tasks", True))
        self._mp_face_detector = None
        self._mp_face_landmarker = None
        self._mp_iris_landmarker = None

        # attempt to initialize the generic single-model interpreter (legacy)
        try:
            try:
                # prefer lightweight runtime
                from tflite_runtime.interpreter import Interpreter
            except Exception:
                from tensorflow.lite import Interpreter  # type: ignore

            if not self._model_path:
                logger.info("TFLite model path not provided for generic detector; it may be disabled")
            else:
                self._interpreter = Interpreter(model_path=self._model_path, num_threads=int(self.config.get("tflite_num_threads", 1)))
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
                self._available = True
                logger.info("TFLiteDrowsinessDetector: loaded generic model %s", self._model_path)
        except Exception:
            logger.exception("Failed to initialize generic TFLite interpreter; continuing (may use mediapipe tasks)")
            self._available = False

        # optionally initialize MediaPipe task loaders if requested
        if self.use_mediapipe_tasks:
            try:
                from brain.utils.mediapipe_loaders import (
                    MediaPipeFaceDetector,
                    MediaPipeFaceLandmarker,
                    MediaPipeIrisLandmarker,
                )
                # Support nested config structure: prefer explicit keys, then mediapipe section
                mp_cfg = self.config.get("mediapipe", {}) if isinstance(self.config, dict) else {}
                face_det_path = self.config.get("face_detection_model") or mp_cfg.get("face_detection", {}).get("model_path")
                face_lm_path = self.config.get("face_landmark_model") or mp_cfg.get("face_landmark", {}).get("model_path")
                iris_path = self.config.get("iris_landmark_model") or mp_cfg.get("iris_landmark", {}).get("model_path")

                if face_det_path:
                    self._mp_face_detector = MediaPipeFaceDetector(face_det_path, mp_cfg)
                if face_lm_path:
                    self._mp_face_landmarker = MediaPipeFaceLandmarker(face_lm_path, mp_cfg)
                if iris_path:
                    self._mp_iris_landmarker = MediaPipeIrisLandmarker(iris_path, mp_cfg)
                logger.info("TFLiteDrowsinessDetector: media-pipe task loaders initialized")
            except Exception:
                logger.exception("Failed to initialize MediaPipe TFLite loaders; falling back to generic interpreter if available")

        # temporal state (same semantics as media-pipe detector)
        self.ear_threshold = float(self.config.get("ear_threshold", 0.25))
        self.mar_threshold = float(self.config.get("mar_threshold", 0.6))
        self.blink_duration_threshold_ms = int(self.config.get("blink_duration_threshold_ms", 200))
        self.yawn_duration_threshold_ms = int(self.config.get("yawn_duration_threshold_ms", 1000))
        self.micro_sleep_duration_s = float(self.config.get("micro_sleep_duration_s", 2.0))

        self._eye_closed_start: Optional[float] = None
        self._mouth_open_start: Optional[float] = None
        self._blink_total = 0
        self._yawn_total = 0
        self._micro_sleep_flag = False

    def get_model_info(self) -> Dict[str, Any]:
        if not self._available:
            return {"available": False}
        return {
            "available": True,
            "input_details": self._input_details,
            "output_details": self._output_details,
        }

    def process_frame(self, frame: np.ndarray, is_ir: bool = False) -> Dict[str, Any]:
        """Run inference on the frame and return detector outputs.

        The function tries to run the TFLite model and map outputs to the same
        dict schema as the MediaPipe-based detector. On any error or if the
        interpreter is not available, returns safe defaults.
        """
        with self._lock:
            try:
                # If MediaPipe TFLite task pipeline is enabled and available, use it
                if self.use_mediapipe_tasks and self._mp_face_detector is not None and self._mp_face_landmarker is not None:
                    try:
                        faces = self._mp_face_detector.detect(frame)
                        if not faces:
                            raise RuntimeError("no faces detected by mediapipe face detector")
                        # pick highest score
                        face = max(faces, key=lambda x: x.get("score", 0.0))
                        bbox = face.get("bbox")
                        lm = self._mp_face_landmarker.detect_landmarks(frame, bbox)
                        if not lm:
                            raise RuntimeError("no landmarks from mediapipe face landmarker")

                        # Prefer standard MediaPipe 468 indices when present
                        ear = 0.0
                        mar = 0.0
                        perclos_iris = None
                        try:
                            if len(lm) >= 468:
                                # use well-known indices
                                left_idx = [33, 160, 158, 133, 153, 144]
                                right_idx = [362, 385, 387, 263, 373, 380]
                                mouth_idx = [61, 291, 78, 308, 13, 14, 308, 14]
                                left_eye = [lm[i] for i in left_idx]
                                right_eye = [lm[i] for i in right_idx]
                                mouth = [lm[i] for i in mouth_idx if i < len(lm)]
                                ear_l = utils.compute_ear(left_eye)
                                ear_r = utils.compute_ear(right_eye)
                                ear = float((ear_l + ear_r) / 2.0)
                                mar = float(utils.compute_mar(mouth)) if mouth else 0.0
                            else:
                                # fallback naive partitioning
                                left_eye = lm[:6]
                                right_eye = lm[6:12]
                                mouth = lm[12:20] if len(lm) >= 20 else []
                                ear = float((utils.compute_ear(left_eye) + utils.compute_ear(right_eye)) / 2.0) if left_eye and right_eye else 0.0
                                mar = float(utils.compute_mar(mouth)) if mouth else 0.0
                        except Exception:
                            logger.exception("Error computing EAR/MAR from mediapipe landmarks")

                        # try iris detection if loader available
                        iris_info = {}
                        if self._mp_iris_landmarker is not None:
                            try:
                                # build approximate eye boxes from landmarks
                                def _bbox_from_landmarks(points):
                                    xs = [p[0] for p in points]
                                    ys = [p[1] for p in points]
                                    xmin = max(0.0, min(xs) - 0.02)
                                    ymin = max(0.0, min(ys) - 0.02)
                                    xmax = min(1.0, max(xs) + 0.02)
                                    ymax = min(1.0, max(ys) + 0.02)
                                    return (xmin, ymin, xmax - xmin, ymax - ymin)

                                if len(lm) >= 6:
                                    left_eye_box = _bbox_from_landmarks(left_eye)
                                    right_eye_box = _bbox_from_landmarks(right_eye)
                                    left_iris = self._mp_iris_landmarker.detect_iris(frame, left_eye_box)
                                    right_iris = self._mp_iris_landmarker.detect_iris(frame, right_eye_box)
                                    iris_info = {"left": left_iris, "right": right_iris}
                                    # perclos computation left as higher-level util; provide raw iris centers
                                    if left_iris.get("center") and right_iris.get("center"):
                                        perclos_iris = 0.0
                            except Exception:
                                logger.exception("Iris detection failed")

                        head_pose = utils.get_head_pose(lm, frame.shape[1], frame.shape[0])

                        # Temporal event detection: simple reuse of thresholds
                        import time

                        now = time.time()
                        blink_event = False
                        yawn_event = False
                        if ear < self.ear_threshold:
                            if self._eye_closed_start is None:
                                self._eye_closed_start = now
                            closed_ms = (now - self._eye_closed_start) * 1000.0
                            if closed_ms >= self.blink_duration_threshold_ms:
                                if not blink_event:
                                    self._blink_total += 1
                                    blink_event = True
                            if (now - self._eye_closed_start) >= self.micro_sleep_duration_s:
                                self._micro_sleep_flag = True
                        else:
                            self._eye_closed_start = None

                        if mar > self.mar_threshold:
                            if self._mouth_open_start is None:
                                self._mouth_open_start = now
                            open_ms = (now - self._mouth_open_start) * 1000.0
                            if open_ms >= self.yawn_duration_threshold_ms:
                                if not yawn_event:
                                    self._yawn_total += 1
                                    yawn_event = True
                        else:
                            self._mouth_open_start = None

                        out = {
                            "ear": float(ear),
                            "mar": float(mar),
                            "head_pose": head_pose,
                            "confidence": float(face.get("score", 1.0)),
                            "no_face_detected": False,
                            "blinks_detected": int(blink_event),
                            "yawns_detected": int(yawn_event),
                            "total_blinks": int(self._blink_total),
                            "total_yawns": int(self._yawn_total),
                            "micro_sleep_flag": bool(self._micro_sleep_flag),
                            "backend_used": "mediapipe_tflite",
                            "iris": iris_info,
                        }
                        if perclos_iris is not None:
                            out["perclos_iris"] = float(perclos_iris)
                        return out
                    except Exception:
                        logger.exception("mediapipe tasks path failed, falling back to generic tflite or heuristics")

                # fallback to legacy generic interpreter or simple heuristic
                if not self._available:
                    # fallback heuristic
                    gray = frame[:, :, 0] if frame.ndim == 3 else frame
                    mean = float(np.mean(gray))
                    ear = 0.3 if mean > 30 else 0.1
                    mar = 0.2 if mean < 120 else 0.4
                    return {
                        "ear": float(ear),
                        "mar": float(mar),
                        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                        "confidence": 0.0,
                        "no_face_detected": True,
                        "blinks_detected": 0,
                        "yawns_detected": 0,
                        "total_blinks": int(self._blink_total),
                        "total_yawns": int(self._yawn_total),
                        "micro_sleep_flag": bool(self._micro_sleep_flag),
                    }

                # Basic preprocess: resize to input shape and normalize to [0,1]
                inp = frame
                inp_det = self._input_details[0]
                # expect input shape like [1, H, W, C]
                h = int(inp_det["shape"][1])
                w = int(inp_det["shape"][2])
                img = utils.preprocess_ir_frame(inp, enable_hist_eq=self.config.get("ir_preprocessing", {}).get("enable_histogram_eq", True), enable_clahe=self.config.get("ir_preprocessing", {}).get("enable_clahe", True)) if is_ir else inp
                if img.ndim == 2:
                    img_in = np.stack([img] * 3, axis=-1)
                else:
                    img_in = img
                # perform a true spatial resize using OpenCV to preserve image content
                try:
                    img_resized = cv2.resize(img_in, (w, h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    # fallback to a safe numpy reshape if resize fails for unexpected reasons
                    img_resized = np.asarray(np.resize(img_in, (h, w, img_in.shape[2])), dtype=np.float32)
                # normalize
                img_resized = img_resized.astype(np.float32) / 255.0
                # set tensor
                self._interpreter.set_tensor(self._input_details[0]["index"], np.expand_dims(img_resized.astype(self._input_details[0]["dtype"]), axis=0))
                self._interpreter.invoke()
                # simplistic handling: try to read first output as landmarks
                out0 = self._interpreter.get_tensor(self._output_details[0]["index"]) if self._output_details else None
                # if outputs do not match expected form, return safe defaults
                if out0 is None:
                    raise RuntimeError("TFLite model produced no outputs")

                # Attempt to parse landmarks: expect shape (1, N, 2) or (1, 2N)
                lm = None
                if out0.ndim == 3 and out0.shape[2] >= 2:
                    lm = [(float(p[0]), float(p[1])) for p in out0[0]]
                elif out0.ndim == 2:
                    flat = out0[0]
                    if flat.size % 2 == 0:
                        lm = [(float(flat[i]), float(flat[i + 1])) for i in range(0, flat.size, 2)]

                if not lm:
                    # fallback
                    return {
                        "ear": 0.0,
                        "mar": 0.0,
                        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                        "confidence": 0.0,
                        "no_face_detected": True,
                        "blinks_detected": 0,
                        "yawns_detected": 0,
                        "total_blinks": int(self._blink_total),
                        "total_yawns": int(self._yawn_total),
                        "micro_sleep_flag": bool(self._micro_sleep_flag),
                    }

                # compute EAR/MAR using existing utils by selecting eye/mouth landmarks
                # For robust mapping, reuse the indices used by MediaPipe when possible
                # but if model outputs canonical 5-point mapping, adapt accordingly.
                # Here we try a naive mapping: use first 6 points as left eye, next 6 as right eye, next 8 as mouth.
                try:
                    left_eye = lm[:6]
                    right_eye = lm[6:12]
                    mouth = lm[12:20]
                    ear_l = utils.compute_ear(left_eye)
                    ear_r = utils.compute_ear(right_eye)
                    ear = float((ear_l + ear_r) / 2.0)
                    mar = float(utils.compute_mar(mouth))
                except Exception:
                    ear = 0.0
                    mar = 0.0

                head_pose = utils.get_head_pose(lm, frame.shape[1], frame.shape[0])

                # Temporal event detection: simple reuse of thresholds
                import time

                now = time.time()
                blink_event = False
                yawn_event = False
                if ear < self.ear_threshold:
                    if self._eye_closed_start is None:
                        self._eye_closed_start = now
                    closed_ms = (now - self._eye_closed_start) * 1000.0
                    if closed_ms >= self.blink_duration_threshold_ms:
                        if not blink_event:
                            self._blink_total += 1
                            blink_event = True
                    if (now - self._eye_closed_start) >= self.micro_sleep_duration_s:
                        self._micro_sleep_flag = True
                else:
                    self._eye_closed_start = None

                if mar > self.mar_threshold:
                    if self._mouth_open_start is None:
                        self._mouth_open_start = now
                    open_ms = (now - self._mouth_open_start) * 1000.0
                    if open_ms >= self.yawn_duration_threshold_ms:
                        if not yawn_event:
                            self._yawn_total += 1
                            yawn_event = True
                else:
                    self._mouth_open_start = None

                return {
                    "ear": float(ear),
                    "mar": float(mar),
                    "head_pose": head_pose,
                    "confidence": 1.0,
                    "no_face_detected": False,
                    "blinks_detected": int(blink_event),
                    "yawns_detected": int(yawn_event),
                    "total_blinks": int(self._blink_total),
                    "total_yawns": int(self._yawn_total),
                    "micro_sleep_flag": bool(self._micro_sleep_flag),
                }
            except Exception:
                logger.exception("TFLite detector inference failed")
                return {
                    "ear": 0.0,
                    "mar": 0.0,
                    "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    "confidence": 0.0,
                    "no_face_detected": True,
                    "blinks_detected": 0,
                    "yawns_detected": 0,
                    "total_blinks": int(self._blink_total),
                    "total_yawns": int(self._yawn_total),
                    "micro_sleep_flag": bool(self._micro_sleep_flag),
                }
