"""DrowsinessDetector using MediaPipe Face Mesh with safe fallbacks.

This module provides a `DrowsinessDetector` that attempts to use MediaPipe
for landmark extraction. If MediaPipe is not available, it falls back to
very small heuristics to keep the API usable in tests and development.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import numpy as np
import cv2

from . import utils
from .tflite_detector import TFLiteDrowsinessDetector

logger = logging.getLogger(__name__)


class DrowsinessDetector:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._lock = threading.Lock()
        # By default prefer MediaPipe TFLite task models when configured.
        # Use explicit config keys to opt in/out: 'use_mediapipe' controls the
        # Python MediaPipe API; 'use_mediapipe_tasks' (or 'use_tflite') enables
        # the TFLite task pipeline which is the preferred low-latency backend.
        self._use_mediapipe = bool(self.config.get("use_mediapipe", True))
        self._use_tflite = bool(self.config.get("use_tflite", self.config.get("use_mediapipe_tasks", True)))
        self._tflite_model_path = self.config.get("tflite_model_path", None)
        self._mp = None
        self._mp_face_mesh = None
        self._mp_drawing = None
        # Haar cascade fallbacks (initialized if MediaPipe isn't available)
        self._face_cascade = None
        self._eye_cascade = None
        self._initialized = False
        # thresholds and durations (defaults chosen to match config/default_config.yaml)
        self.ear_threshold = float(self.config.get("ear_threshold", 0.25))
        self.mar_threshold = float(self.config.get("mar_threshold", 0.6))
        self.blink_duration_threshold_ms = int(self.config.get("blink_duration_threshold_ms", 200))
        self.yawn_duration_threshold_ms = int(self.config.get("yawn_duration_threshold_ms", 1000))
        self.micro_sleep_duration_s = float(self.config.get("micro_sleep_duration_s", 2.0))
        self.face_detection_confidence = float(self.config.get("face_detection_confidence", 0.5))
        # internal state for event detection
        self._eye_closed_start: Optional[float] = None
        self._mouth_open_start: Optional[float] = None
        self._blink_total = 0
        self._yawn_total = 0
        self._eye_closed_in_event = False
        self._mouth_open_in_event = False
        self._micro_sleep_flag = False
        # thresholds and durations (defaults chosen to match config/default_config.yaml)
        self.ear_threshold = float(self.config.get("ear_threshold", 0.25))
        self.mar_threshold = float(self.config.get("mar_threshold", 0.6))
        self.blink_duration_threshold_ms = int(self.config.get("blink_duration_threshold_ms", 200))
        self.yawn_duration_threshold_ms = int(self.config.get("yawn_duration_threshold_ms", 1000))
        self.micro_sleep_duration_s = float(self.config.get("micro_sleep_duration_s", 2.0))
        self.face_detection_confidence = float(self.config.get("face_detection_confidence", 0.5))

        # internal state for temporal event detection
        self._blink_total = 0
        self._yawn_total = 0
        self._eye_closed_start: Optional[float] = None
        self._eye_closed_in_event = False
        self._mouth_open_start: Optional[float] = None
        self._mouth_open_in_event = False
        self._micro_sleep_flag = False

        self._init_mediapipe()
        # optional tflite backend
        self._tflite_detector: Optional[TFLiteDrowsinessDetector] = None
        if self._use_tflite:
            try:
                # pass sensor section config so the detector can read mediapipe paths and flags
                self._tflite_detector = TFLiteDrowsinessDetector(self.config, model_path=self._tflite_model_path)
                if self._tflite_detector.get_model_info().get("available"):
                    logger.info("Using TFLite backend for drowsiness detection")
                else:
                    logger.warning("TFLite backend requested but not available; falling back to MediaPipe/fallback heuristics")
                    self._tflite_detector = None
            except Exception:
                logger.exception("Failed to initialize TFLite detector; will use MediaPipe or fallback")
                self._tflite_detector = None

    def _init_mediapipe(self):
        if not self._use_mediapipe:
            logger.info("MediaPipe disabled by config; using fallback detector")
            return
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
            self._mp_drawing = mp.solutions.drawing_utils
            self._initialized = True
            logger.info("MediaPipe Face Mesh initialized for drowsiness detection")
        except Exception:
            logger.warning("MediaPipe not available; falling back to Haar cascade detector if available")
            self._initialized = False
            try:
                import cv2
                # load cascades for fallback detection
                self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                self._eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
                logger.info("Loaded Haar cascades for fallback detection")
            except Exception:
                logger.exception("Failed to load Haar cascades for fallback detection")

    def process_frame(self, frame: np.ndarray, is_ir: bool = False) -> Dict[str, Any]:
        """Process a BGR frame and return detection results.

        Returns a dict with keys:
          - ear, mar, head_pose (dict), blinks, yawns, confidence
        """
        with self._lock:
            try:
                if is_ir:
                    frame_proc = utils.preprocess_ir_frame(frame, self.config.get("ir_preprocessing", {}).get("enable_histogram_eq", True), self.config.get("ir_preprocessing", {}).get("enable_clahe", True))
                    # MediaPipe expects 3-channel input; convert back if needed
                    if frame_proc.ndim == 2:
                        frame_in = np.stack([frame_proc] * 3, axis=-1)
                    else:
                        frame_in = frame
                else:
                    frame_in = frame

                # If tflite backend is active, delegate to it
                if self._tflite_detector is not None:
                    return self._tflite_detector.process_frame(frame, is_ir=is_ir)

                if self._initialized and self._mp is not None:
                    rgb = frame_in[:, :, ::-1]
                    # optional downscale to speed up MediaPipe processing (configurable)
                    try:
                        if bool(self.config.get("downscale", True)):
                            rgb = cv2.resize(rgb, (320, 240))
                    except Exception:
                        pass
                    results = self._mp_face_mesh.process(rgb)
                    if not results or not results.multi_face_landmarks:
                        return {"ear": 0.0, "mar": 0.0, "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, "confidence": 0.0, "no_face_detected": True, "blinks_detected": 0, "yawns_detected": 0, "total_blinks": self._blink_total, "total_yawns": self._yawn_total, "micro_sleep_flag": False}
                    lm = results.multi_face_landmarks[0]
                    # Extract a few landmarks for EAR/MAR approximations
                    # MediaPipe landmarks are relative to the processed image (rgb).
                    proc_h, proc_w = rgb.shape[:2]
                    orig_h, orig_w = frame_in.shape[:2]
                    sx = orig_w / proc_w
                    sy = orig_h / proc_h
                    coords = [(int(p.x * proc_w * sx), int(p.y * proc_h * sy)) for p in lm.landmark]
                    # safe selection of eye/mouth landmarks: guard against incomplete meshes
                    def safe_select(indices_list, fallback_len):
                        pts = []
                        for i in indices_list:
                            if i < len(coords):
                                pts.append(coords[i])
                        if len(pts) < fallback_len:
                            return []
                        return pts

                    left_eye = safe_select([33, 160, 158, 133, 153, 144], 6)
                    right_eye = safe_select([362, 385, 387, 263, 373, 380], 6)
                    mouth = safe_select([61, 291, 78, 308, 13, 14, 308, 14], 8)

                    ear = 0.0
                    try:
                        if left_eye and right_eye:
                            ear_l = utils.compute_ear(left_eye)
                            ear_r = utils.compute_ear(right_eye)
                            ear = float((ear_l + ear_r) / 2.0)
                        elif left_eye:
                            ear = float(utils.compute_ear(left_eye))
                        elif right_eye:
                            ear = float(utils.compute_ear(right_eye))
                    except Exception:
                        ear = 0.0

                    try:
                        mar = float(utils.compute_mar(mouth)) if mouth else 0.0
                    except Exception:
                        mar = 0.0
                    # head pose: try to compute using selected landmarks
                    # use the original frame dimensions (orig_w, orig_h)
                    head_pose = utils.get_head_pose(coords, orig_w, orig_h)

                    # timestamp for event detection
                    import time

                    now = time.time()
                    blink_event = False
                    yawn_event = False
                    # Eye closure handling
                    if ear < self.ear_threshold:
                        if self._eye_closed_start is None:
                            self._eye_closed_start = now
                        closed_ms = (now - self._eye_closed_start) * 1000.0
                        if closed_ms >= self.blink_duration_threshold_ms and not self._eye_closed_in_event:
                            self._blink_total += 1
                            self._eye_closed_in_event = True
                            blink_event = True
                        if (now - self._eye_closed_start) >= self.micro_sleep_duration_s:
                            self._micro_sleep_flag = True
                    else:
                        # eye opened
                        self._eye_closed_start = None
                        self._eye_closed_in_event = False

                    # Mouth / yawn handling
                    if mar > self.mar_threshold:
                        if self._mouth_open_start is None:
                            self._mouth_open_start = now
                        open_ms = (now - self._mouth_open_start) * 1000.0
                        if open_ms >= self.yawn_duration_threshold_ms and not self._mouth_open_in_event:
                            self._yawn_total += 1
                            self._mouth_open_in_event = True
                            yawn_event = True
                    else:
                        self._mouth_open_start = None
                        self._mouth_open_in_event = False

                    return {
                        "ear": ear,
                        "mar": mar,
                        "head_pose": head_pose,
                        "confidence": 1.0,
                        "no_face_detected": False,
                        "blinks_detected": int(blink_event),
                        "yawns_detected": int(yawn_event),
                        "total_blinks": int(self._blink_total),
                        "total_yawns": int(self._yawn_total),
                        "micro_sleep_flag": bool(self._micro_sleep_flag),
                    }
                else:
                    # Fallback: try Haar cascade face+eye detection (more useful than pure brightness)
                    try:
                        gray = frame_in[:, :, 0] if frame_in.ndim == 3 else frame_in
                        ear = 0.0
                        mar = 0.0
                        detected = False
                        confidence = 0.0
                        if self._face_cascade is not None:
                            faces = self._face_cascade.detectMultiScale(gray, 1.1, 4)
                            if len(faces) > 0:
                                x, y, fw, fh = faces[0]
                                roi_gray = gray[y:y+fh, x:x+fw]
                                eyes = []
                                if self._eye_cascade is not None:
                                    eyes = self._eye_cascade.detectMultiScale(roi_gray)
                                if len(eyes) >= 2:
                                    # compute simple EAR estimation from eye boxes
                                    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
                                    ear_vals = []
                                    for (ex, ey, ew, eh) in eyes:
                                        ear_vals.append(max(0.05, min(0.35, 0.25 * (eh / (ew + 1e-6)) * 2.0)))
                                    ear = float(sum(ear_vals) / len(ear_vals))
                                    detected = True
                                    confidence = 0.6
                                else:
                                    # approximate EAR from face box aspect
                                    ear = 0.22
                                    detected = True
                                    confidence = 0.4
                        if not detected:
                            # fallback to brightness heuristic
                            mean = float(np.mean(gray)) if gray.size else 255.0
                            ear = 0.3 if mean > 30 else 0.1
                            mar = 0.2 if mean < 120 else 0.4
                            confidence = 0.2
                        head_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
                        return {"ear": ear, "mar": mar, "head_pose": head_pose, "confidence": float(confidence), "no_face_detected": not bool(detected), "blinks_detected": 0, "yawns_detected": 0, "total_blinks": int(self._blink_total), "total_yawns": int(self._yawn_total), "micro_sleep_flag": False}
                    except Exception:
                        # worst-case fallback: brightness heuristic
                        gray = frame_in[:, :, 0] if frame_in.ndim == 3 else frame_in
                        mean = float(np.mean(gray))
                        ear = 0.3 if mean > 30 else 0.1
                        mar = 0.2 if mean < 120 else 0.4
                        head_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
                        return {"ear": ear, "mar": mar, "head_pose": head_pose, "confidence": 0.2, "no_face_detected": False, "blinks_detected": 0, "yawns_detected": 0, "total_blinks": int(self._blink_total), "total_yawns": int(self._yawn_total), "micro_sleep_flag": False}
            except Exception:
                logger.exception("Detection failed")
                return {"ear": 0.0, "mar": 0.0, "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}, "blink": False, "yawn": False, "micro_sleep": False, "blink_count": self._blink_total, "yawn_count": self._yawn_total, "confidence": 0.0, "no_face_detected": True}
