"""Camera interface for driver drowsiness detection.

Wraps OpenCV VideoCapture and provides frame reading, preprocessing,
and simple health monitoring. Designed to be testable (video file input)
and to degrade gracefully if no camera is available.
"""
from __future__ import annotations

import time
import threading
from typing import Optional

import cv2
import numpy as np


class CameraInterface:
    def __init__(self, device=0, width=640, height=480, fps=30, video_file: str = None, is_ir: bool = False):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.video_file = video_file
        self.is_ir = bool(is_ir)
        self._lock = threading.Lock()
        self._cap = None
        self._last_frame_ts = 0.0
        self._last_frame = None
        self._failed_reads = 0
        self.status = "ok"
        # adaptive fps / low-power settings
        self.enable_low_power_mode = False
        self.low_power_fps = int(self.fps) if self.fps else 2
        self.normal_fps = int(self.fps)
        self.motion_detection_threshold = float(5.0)
        self.no_face_frames_threshold = int(5)
        self._no_face_frames = 0
        self._last_motion_check_frame = None
        self._low_power_active = False
        self._open()

    def _open(self):
        if self.video_file:
            self._cap = cv2.VideoCapture(self.video_file)
        else:
            self._cap = cv2.VideoCapture(int(self.device))
        try:
            if self._cap is not None and self._cap.isOpened():
                # attempt to set properties; some backends may ignore
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.status = "ok"
            else:
                # mark as not opened but keep object
                self.status = "not_open"
        except Exception:
            self.status = "error"

    def read_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self._failed_reads += 1
                if self._failed_reads > 5:
                    self.status = "error"
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._failed_reads += 1
                if self._failed_reads > 5:
                    self.status = "error"
                return None

            try:
                # ensure target resolution
                if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                    frame = cv2.resize(frame, (self.width, self.height))
            except Exception:
                # ignore resize failures
                pass

            # IR preprocessing: convert and return as 3-channel image for downstream processing
            if self.is_ir:
                try:
                    from . import utils as dd_utils

                    proc = dd_utils.preprocess_ir_frame(frame, enable_hist_eq=True, enable_clahe=True)
                    # proc is 2D gray; convert to 3-channel BGR-like image so detector can treat it like color
                    frame = np.stack([proc] * 3, axis=-1)
                except Exception:
                    # fallback to original frame on error
                    pass

            # detect motion for low-power mode toggling
            try:
                if self.enable_low_power_mode:
                    if self._last_motion_check_frame is not None:
                        motion = False
                        try:
                            from . import utils as dd_utils

                            motion = dd_utils.detect_motion(frame, self._last_motion_check_frame, threshold=self.motion_detection_threshold)
                        except Exception:
                            motion = True
                        if motion:
                            # resume normal
                            self._low_power_active = False
                            self._no_face_frames = 0
                        else:
                            # do not infer no-face from motion absence alone; rely on detector to call mark_no_face_detected()
                            pass
                    self._last_motion_check_frame = frame.copy()
            except Exception:
                pass

            self._last_frame = frame
            self._last_frame_ts = time.time()
            self._failed_reads = 0
            self.status = "ok"

            # implement sleep-based FPS throttling when in low power mode
            if self.enable_low_power_mode and self._low_power_active:
                target = float(self.low_power_fps) if self.low_power_fps > 0 else 1.0
                delay = max(0.0, (1.0 / target) - 0.001)
                time.sleep(delay)

            return frame

    def get_status(self) -> dict:
        return {
            "status": self.status,
            "is_open": self._cap.isOpened() if self._cap is not None else False,
            "failed_reads": self._failed_reads,
            "last_frame_ts": self._last_frame_ts,
            "current_fps": self.low_power_fps if self._low_power_active else self.normal_fps,
            "low_power_active": self._low_power_active,
        }

    def mark_no_face_detected(self) -> None:
        """Increment the no-face counter; if it exceeds threshold, enter low-power mode."""
        with self._lock:
            try:
                self._no_face_frames += 1
                if self._no_face_frames >= self.no_face_frames_threshold:
                    self._low_power_active = True
            except Exception:
                pass

    def clear_no_face_counter(self) -> None:
        """Reset the no-face counter and exit low-power mode."""
        with self._lock:
            try:
                self._no_face_frames = 0
                self._low_power_active = False
            except Exception:
                pass

    def release(self):
        with self._lock:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
