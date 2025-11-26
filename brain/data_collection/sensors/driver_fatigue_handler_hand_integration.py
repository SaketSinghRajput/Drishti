"""
Grip detection helper for DriverFatigueHandler.

This module provides a small, self-contained thread-based grip monitor that can
be instantiated by the existing `DriverFatigueHandler` implementation. It is
designed to be non-intrusive: it accepts frames via a small queue and updates a
thread-safe state dict with the latest `grip_confidence` value.

The design intentionally mirrors the plan: a background daemon thread uses the
`MediaPipeHandLandmarker` to run inference without blocking the main frame loop.

Usage (example integration snippet):

    from brain.utils.mediapipe_hand_landmarker import MediaPipeHandLandmarker, compute_grip_confidence_from_keypoints
    from brain.data_collection.sensors.driver_fatigue_handler_hand_integration import GripMonitor

    grip = GripMonitor(model_path=config['mediapipe']['hand_landmark']['model_path'])
    grip.start()
    # for each frame:
    grip.enqueue(frame)
    # read current state:
    s = grip.get_state()  # {'grip_confidence': 0.82, 'last_updated': 163...}
    grip.stop()

"""

import threading
import queue
import time
from typing import Optional, Dict, Any
import logging

import numpy as np

from brain.utils.mediapipe_hand_landmarker import MediaPipeHandLandmarker, compute_grip_confidence_from_keypoints

logger = logging.getLogger(__name__)


class GripMonitor:
    """Background monitor that runs hand landmark inference and computes grip confidence.

    Args:
        model_path: path to the hand landmark tflite model
        enable_parallel: whether to run thread; if False, monitor runs synchronously when `process_once()` is called
        max_queue: max frames to buffer (default 2)
    """

    def __init__(self, model_path: str, enable_parallel: bool = True, max_queue: int = 2, config: Optional[dict] = None):
        self._model_path = model_path
        self._config = config or {}
        self._hand = MediaPipeHandLandmarker(model_path, config=self._config)
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._state_lock = threading.Lock()
        self._grip_state: Dict[str, Any] = {'grip_confidence': 0.0, 'last_updated': 0.0, 'available': self._hand.is_available()}
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._enable_parallel = enable_parallel

    def start(self):
        if not self._hand.is_available():
            logger.info("GripMonitor: hand landmark model unavailable; not starting thread")
            return
        if self._enable_parallel and (self._thread is None or not self._thread.is_alive()):
            self._running.set()
            self._thread = threading.Thread(target=self._run_loop, name="GripMonitorThread", daemon=True)
            self._thread.start()
            logger.debug("GripMonitor thread started")

    def stop(self, timeout: float = 1.0):
        self._running.clear()
        try:
            if self._thread is not None:
                self._thread.join(timeout=timeout)
        except Exception:
            pass
        logger.debug("GripMonitor stopped")

    def enqueue(self, frame: np.ndarray):
        """Attempt to enqueue a frame for background processing (non-blocking)."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # drop if queue full — we prefer freshness
            pass

    def process_once(self, frame: np.ndarray):
        """Synchronous processing path (if parallel disabled)."""
        self._process_frame(frame)

    def get_state(self) -> Dict[str, Any]:
        with self._state_lock:
            return dict(self._grip_state)

    def _run_loop(self):
        while self._running.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_frame(frame)
            except Exception:
                logger.exception("Exception in GripMonitor processing")

    def _process_frame(self, frame: np.ndarray):
        """Run detection on full frame: attempt to find hand bboxes by simple heuristic.

        Note: The repository likely already has a hand detection path; to keep this
        module self-contained we use a heuristic: search left/right halves for
        skin-colored contours or fallback to full-frame attempt.
        """
        h, w = frame.shape[:2]
        # Simple heuristic: search left and right central regions for hands
        regions = [
            (0.05, 0.4, 0.4, 0.4),  # left central
            (0.55, 0.4, 0.4, 0.4),  # right central
        ]
        best_conf = 0.0
        best_kps = []
        for r in regions:
            try:
                kps = self._hand.detect_hand_landmarks(frame, r)
                if kps:
                    conf = compute_grip_confidence_from_keypoints(kps, frame_size=(w, h))
                    if conf > best_conf:
                        best_conf = conf
                        best_kps = kps
            except Exception:
                logger.exception("Hand detection error")

        # If no region detected, try full-frame quick pass
        if best_conf == 0.0:
            kps = self._hand.detect_hand_landmarks(frame, (0.0, 0.0, 1.0, 1.0))
            if kps:
                best_conf = compute_grip_confidence_from_keypoints(kps, frame_size=(w, h))
                best_kps = kps

        # Update state
        with self._state_lock:
            self._grip_state['grip_confidence'] = float(best_conf)
            self._grip_state['last_updated'] = time.time()
            self._grip_state['available'] = self._hand.is_available()
            self._grip_state['keypoints_count'] = len(best_kps)

    def is_available(self) -> bool:
        return self._hand.is_available()


# Convenience factory used by handlers

def make_grip_monitor_from_config(cfg: dict) -> GripMonitor:
    mp = cfg.get('mediapipe', {})
    hand_cfg = mp.get('hand_landmark', {})
    enable = hand_cfg.get('enabled', False)
    model_path = hand_cfg.get('model_path', 'brain/models/hand_landmark_lite.tflite')
    parallel = hand_cfg.get('parallel_inference', True)
    if not enable:
        return GripMonitor(model_path=model_path, enable_parallel=False)
    gm = GripMonitor(model_path=model_path, enable_parallel=parallel)
    return gm
