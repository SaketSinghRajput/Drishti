"""Driver fatigue handler with optional CV-based detection.

This handler preserves the previous simulation behavior but can be
configured to use the CV-based `DrowsinessDetector` and `CameraInterface`
located in `data_collection.sensors.driver_drowsiness`.
"""
from __future__ import annotations

import random
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from collections import deque

from .base import SensorHandler
from utils.time_utils import get_utc_timestamp
from data_collection.sensors.driver_drowsiness import utils as dd_utils
from data_collection.sensors.driver_drowsiness.profiler import LatencyProfiler
from utils.fatigue_alert_manager import FatigueAlertManager
try:
    import requests
except Exception:
    requests = None

try:
    from data_collection.sensors.ble_fatigue_broadcast import BLEFatigueBroadcast
except Exception:
    BLEFatigueBroadcast = None

try:
    from utils.fatigue_logger import FatigueLogger
except Exception:
    FatigueLogger = None

try:
    from data_collection.sensors.fatigue_api import get_global_state as FatigueAPIState
except Exception:
    FatigueAPIState = None

from utils.ble_utils import hash_node_id

logger = logging.getLogger(__name__)


try:
    from data_collection.sensors.driver_drowsiness.detection_engine import DrowsinessDetector
    from data_collection.sensors.driver_drowsiness.camera_interface import CameraInterface
except Exception:
    DrowsinessDetector = None  # type: ignore
    CameraInterface = None  # type: ignore
try:
    from data_collection.sensors.driver_fatigue_handler_hand_integration import make_grip_monitor_from_config
except Exception:
    try:
        from .driver_fatigue_handler_hand_integration import make_grip_monitor_from_config
    except Exception:
        make_grip_monitor_from_config = None


class DriverFatigueHandler(SensorHandler):
    def __init__(self, config, use_cv: Optional[bool] = None, telemetry_collector: Optional[Any] = None):
        super().__init__(config)
        self.sample_rate = int(self.config.get("sensors.driver_fatigue.sample_rate", 1))
        # rolling state
        self._drowsiness = 0.0
        self._blink_rate = 0.0
        self._perclos = 0.0
        self._yawning = 0
        self._micro_sleep = False
        self._blink_timestamps = deque()
        self._blink_rate_window = float(self.config.get("blink_rate_window", 60.0))
        # PERCLOS tracking: sample timestamps and closed-eye timestamps within window
        self._sample_timestamps = deque()
        self._closed_timestamps = deque()
        self._no_face_count = 0
        self._read_count = 0
        self._total_yawns = 0
        # state classification config
        sc = self.config.get("sensors.driver_fatigue.state_classification", {})
        self._threshold_fatigue_low = float(sc.get("fatigue_low", 0.3))
        self._threshold_drowsy = float(sc.get("drowsy", 0.6))
        self._threshold_micro = float(sc.get("micro", 0.8))
        self._hysteresis_margin = float(sc.get("state_hysteresis_margin", 0.05))
        self._hysteresis_dwell_s = float(sc.get("state_hysteresis_dwell_s", 2.0))
        # enable/disable state classification (guard behavior in read)
        self._state_classification_enabled = bool(sc.get("enable", True))

        # state machine
        self._current_state = "NORMAL"
        self._last_state_change_ts = 0.0
        self._state_history = deque(maxlen=100)
        self._events = deque(maxlen=50)

        # alert manager for local actuation
        try:
            self._alert_mgr = FatigueAlertManager(self.config)
        except Exception:
            logger.exception("Failed to initialize FatigueAlertManager; alerts disabled")
            self._alert_mgr = None
        # BLE broadcast (file outbox) - config-gated
        try:
            # support nested config dicts: sensors -> driver_fatigue -> ble_broadcast
            bc_enable = False
            try:
                bc_enable = bool(self.config.get("sensors", {}).get("driver_fatigue", {}).get("ble_broadcast", {}).get("enable", False))
            except Exception:
                bc_enable = False

            if BLEFatigueBroadcast is not None and bc_enable:
                # wire BLEMeshHandler if configured/enabled
                mesh_handler = None
                try:
                    mh_enable = bool(self.config.get("sensors", {}).get("ble_mesh", {}).get("enable", False))
                except Exception:
                    mh_enable = False
                if mh_enable:
                    try:
                        from data_collection.sensors.ble_mesh_handler import BLEMeshHandler

                        mesh_handler = BLEMeshHandler(self.config)
                    except Exception:
                        mesh_handler = None
                self._ble_broadcast = BLEFatigueBroadcast(self.config, mesh_handler=mesh_handler)
            else:
                self._ble_broadcast = None
        except Exception:
            logger.exception("Failed to initialize BLEFatigueBroadcast")
            self._ble_broadcast = None

        # Fatigue SQLite logger
        try:
            if FatigueLogger is not None and bool(self.config.get("sensors.driver_fatigue", {}).get("fatigue_logging", {}).get("enable", True)):
                # accept telemetry_collector passed by caller (preferred)
                self._fatigue_logger = FatigueLogger(self.config, telemetry_collector=telemetry_collector)
            else:
                self._fatigue_logger = None
        except Exception:
            logger.exception("Failed to initialize FatigueLogger")
            self._fatigue_logger = None

        # API state cache
        try:
            if FatigueAPIState is not None:
                self._api_state = FatigueAPIState(int(self.config.get("sensors.driver_fatigue", {}).get("fatigue_api", {}).get("max_events_cache", 100)))
            else:
                self._api_state = None
        except Exception:
            logger.exception("Failed to initialize FatigueAPIState")
            self._api_state = None

        # central push config
        cp = self.config.get("sensors", {}).get("driver_fatigue", {}).get("central_push", {})
        self._central_enabled = bool(cp.get("enable", False))
        self._central_url = str(cp.get("endpoint_url", "http://localhost:5000/events"))
        self._central_timeout = float(cp.get("timeout_s", 2.0))
        self._central_retries = int(cp.get("retry_attempts", 1))
        self._central_backoff = float(cp.get("retry_backoff_s", 1.0))

        # vehicle id hash for privacy
        try:
            vehicle_id = str(self.config.get("vehicle", {}).get("id", "")) if isinstance(self.config, dict) else ""
            self._vehicle_hash = hash_node_id(vehicle_id, salt=str(self.config.get("privacy", {}).get("node_id_salt", "")))
        except Exception:
            self._vehicle_hash = ""
        # event id counter for correlation
        self._event_counter = 0
        # CV resources
        # allow caller to override CV enable for testing
        cfg_cv = bool(self.config.get("sensors.driver_fatigue.enable_cv_detection", True))
        self._cv_enabled = cfg_cv if use_cv is None else bool(use_cv)
        self._camera = None
        self._detector = None
        # latency profiler
        prof_cfg = self.config.get("sensors.driver_fatigue", {}).get("enable_latency_profiling", True)
        budget_ms = int(self.config.get("sensors.driver_fatigue", {}).get("latency_budget_ms", self.config.get("inference", {}).get("latency_budget_ms", 200)))
        self._profiler = LatencyProfiler(budget_ms=budget_ms) if prof_cfg else None
        if self._cv_enabled and DrowsinessDetector is not None and CameraInterface is not None:
            try:
                cam_id = self.config.get("sensors.driver_fatigue.camera_device_id", 0)
                res = self.config.get("sensors.driver_fatigue.camera_resolution", {"width": 640, "height": 480})
                fps = self.config.get("sensors.driver_fatigue.camera_fps", 30)
                video_file = self.config.get("sensors.driver_fatigue.video_file_path", "") if self.config.get("sensors.driver_fatigue.enable_video_file_input", False) else None
                self._camera = CameraInterface(device=cam_id, width=res.get("width", 640), height=res.get("height", 480), fps=fps, video_file=video_file, is_ir=self.config.get("sensors.driver_fatigue.enable_ir_mode", False))
                # adaptive fps settings
                try:
                    self._camera.enable_low_power_mode = bool(self.config.get("sensors.driver_fatigue", {}).get("enable_low_power_mode", False))
                    self._camera.low_power_fps = int(self.config.get("sensors.driver_fatigue", {}).get("low_power_fps", 2))
                    self._camera.normal_fps = int(self.config.get("sensors.driver_fatigue", {}).get("normal_fps", fps))
                    self._camera.motion_detection_threshold = float(self.config.get("sensors.driver_fatigue", {}).get("motion_detection_threshold", 5.0))
                    self._camera.no_face_frames_threshold = int(self.config.get("sensors.driver_fatigue", {}).get("no_face_frames_threshold", 5))
                except Exception:
                    pass
                    # Pass the full config to DrowsinessDetector so mediapipe/task paths
                    # nested under `mediapipe:` are available to the detector.
                        # Pass full config so nested `mediapipe` section is visible to detectors
                        self._detector = DrowsinessDetector(self.config)
                    logger.info("DriverFatigueHandler: initialized CV detector and camera")
                    # Initialize GripMonitor from config (if available)
                    try:
                        if make_grip_monitor_from_config is not None:
                            self._grip_monitor = make_grip_monitor_from_config(self.config)
                            # start background thread if enabled/available
                            try:
                                if getattr(self, "_grip_monitor", None) is not None:
                                    self._grip_monitor.start()
                            except Exception:
                                logger.exception("Failed to start GripMonitor thread")
                        else:
                            self._grip_monitor = None
                    except Exception:
                        logger.exception("Failed to initialize GripMonitor")
                        self._grip_monitor = None
            except Exception:
                logger.exception("Failed to initialize CV detector; falling back to simulation")
                self._camera = None
                self._detector = None
        else:
            logger.debug("CV detection disabled or not available; using simulation mode")
            # ensure grip monitor attribute exists even when CV disabled
            self._grip_monitor = None

    def get_name(self) -> str:
        return "driver_fatigue"

    def read(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._detector is None or self._camera is None:
                sample = {
                    "timestamp": time.time(),
                    "drowsiness_score": 0.0,
                    "blink_rate": 0.0,
                    "perclos": 0.0,
                    "yawning_count": 0,
                    "micro_sleep_flag": False,
                    "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0},
                }
                return sample
            # profiling: frame read
            t0 = time.perf_counter()
            if self._profiler:
                prof = self._profiler.profile("frame_read")
                prof.__enter__()
            frame = self._camera.read_frame()
            # enqueue frame for grip monitor (non-blocking) if available
            try:
                if getattr(self, '_grip_monitor', None) is not None:
                    try:
                        self._grip_monitor.enqueue(frame)
                    except Exception:
                        # do not block main loop for grip errors
                        logger.exception('GripMonitor enqueue failed')
            except Exception:
                pass
            if self._profiler:
                prof.__exit__(None, None, None)
            if frame is None:
                return None

            # inference
            if self._profiler:
                prof_inf = self._profiler.profile("inference")
                prof_inf.__enter__()
            res = self._detector.process_frame(frame)
            if self._profiler:
                prof_inf.__exit__(None, None, None)
            self._read_count += 1

            ear = float(res.get("ear", 0.0))
            mar = float(res.get("mar", 0.0))
            head_pose = res.get("head_pose", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
            confidence = float(res.get("confidence", 0.0))
            no_face = bool(res.get("no_face_detected", False))

            face_conf_thresh = float(self.config.get("sensors.driver_fatigue.face_detection_confidence", 0.5))
            # feed no-face signal to camera so it can enter low-power when appropriate
            try:
                if self._camera is not None:
                    if no_face:
                        if hasattr(self._camera, "mark_no_face_detected"):
                            self._camera.mark_no_face_detected()
                    else:
                        if hasattr(self._camera, "clear_no_face_counter"):
                            self._camera.clear_no_face_counter()
            except Exception:
                pass
            if no_face or confidence < face_conf_thresh:
                # safe baseline when face not detected / low confidence
                self._no_face_count += 1
                sample = {
                    "timestamp": time.time(),
                    "drowsiness_score": 0.0,
                    "blink_rate": 0.0,
                    "perclos": 0.0,
                    # iris-based perclos when available from detector
                    "perclos_iris": float(res.get("perclos_iris", 0.0)),
                    # backend used by detector (mediapipe_tflite, mediapipe_python, haar, etc.)
                    "backend_used": str(res.get("backend_used", "unknown")),
                    "yawning_count": int(res.get("total_yawns", 0)),
                    "micro_sleep_flag": bool(res.get("micro_sleep_flag", False)),
                    "head_pose": {**head_pose, "confidence": confidence},
                }
                # record profiling total
                if self._profiler:
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    self._profiler.record_total(total_ms)
                    self._profiler.log_warning_if_exceeded()
                return sample

            # normalization using thresholds
            # adjust thresholds for IR/night mode (scoped to sensors.driver_fatigue)
            ear_thresh = float(self.config.get("sensors.driver_fatigue.ear_threshold", 0.25))
            mar_thresh = float(self.config.get("sensors.driver_fatigue.mar_threshold", 0.6))
            try:
                if self._camera is not None and getattr(self._camera, "is_ir", False):
                    ambient = dd_utils.estimate_ambient_light(frame)
                    ear_thresh, mar_thresh = dd_utils.adjust_thresholds_for_ir(
                        ear_thresh,
                        mar_thresh,
                        ambient,
                        ir_ear_offset=float(self.config.get("sensors.driver_fatigue", {}).get("ir_ear_threshold_offset", 0.05)),
                        ir_mar_offset=float(self.config.get("sensors.driver_fatigue", {}).get("ir_mar_threshold_offset", -0.05)),
                        ambient_light_low_threshold=float(self.config.get("sensors.driver_fatigue", {}).get("ambient_light_low_threshold", 30.0)),
                    )
            except Exception:
                pass
            # EAR normalization: treat ear_norm as open-eye metric in [0,1]
            # and convert to a fatigue component as (1 - ear_norm) so that
            # higher values indicate more eye closure / fatigue.
            ear_norm = ear / ear_thresh if ear_thresh > 0 else 1.0
            ear_norm = max(0.0, min(1.0, ear_norm))
            eye_fatigue = 1.0 - ear_norm
            norm_mar = (mar - mar_thresh) / (1.0 - mar_thresh) if mar_thresh < 1.0 else 0.0
            norm_mar = max(0.0, min(1.0, norm_mar))
            head_tilt_norm = (abs(head_pose.get("pitch", 0.0)) + abs(head_pose.get("yaw", 0.0))) / float(self.config.get("sensors.driver_fatigue.head_tilt_threshold", 30.0) * 2.0)
            head_tilt_norm = max(0.0, min(1.0, head_tilt_norm))
            # weighted fatigue score: prefer eye_fatigue/mar/head tilt composition
            drowsiness = 0.5 * eye_fatigue + 0.3 * norm_mar + 0.2 * head_tilt_norm
            drowsiness = max(0.0, min(1.0, float(drowsiness)))
            # State classification with hysteresis (guarded by config flag)
            prev_state = self._current_state
            if self._state_classification_enabled:
                target_state = self._classify_state(drowsiness)
                try:
                    allow = self._should_transition(prev_state, target_state, drowsiness, time.time())
                except Exception:
                    allow = False
                if allow and target_state != prev_state:
                    # perform transition (already under self._lock)
                    self._current_state = target_state
                    self._last_state_change_ts = time.time()
                    self._state_history.append((self._current_state, self._last_state_change_ts))
                    # determine dominant cause for more descriptive reason
                    try:
                        # compare raw normalized components (eye_fatigue, norm_mar, head_tilt_norm)
                        comps = {
                            "EAR": float(eye_fatigue),
                            "MAR": float(norm_mar),
                            "HEAD_TILT": float(head_tilt_norm),
                        }
                        dominant = max(comps, key=lambda k: comps[k])
                        if dominant == "EAR":
                            reason = "EAR below threshold"
                        elif dominant == "MAR":
                            reason = "Yawning detected"
                        else:
                            reason = "Head tilt excessive"
                    except Exception:
                        reason = "score_threshold"
                    try:
                        self._emit_state_event(prev_state, target_state, drowsiness, reason)
                    except Exception:
                        logger.exception("Failed to emit state event")
                    try:
                        if self._alert_mgr is not None:
                            self._alert_mgr.trigger_alert(target_state, drowsiness, reason)
                    except Exception:
                        logger.exception("Failed to trigger alert for state %s", target_state)
            else:
                # classification disabled: ensure deterministic sample tag
                target_state = "NORMAL"
                # force state to NORMAL for deterministic downstream behavior
                self._current_state = "NORMAL"


            # blink handling and rate computation
            blink_detected = int(res.get("blinks_detected", 0))
            now_ts = time.time()
            # track sample timestamp
            self._sample_timestamps.append(now_ts)
            # record blink events separately for blink_rate
            if blink_detected > 0:
                self._blink_timestamps.append(now_ts)
            # closed-eye tracking for PERCLOS (use scoped threshold)
            if ear < float(self.config.get("sensors.driver_fatigue.ear_threshold", 0.25)):
                # mark this sample as closed
                self._closed_timestamps.append(now_ts)

            # purge old timestamps beyond window
            window = float(self._blink_rate_window)
            cutoff = time.time() - window
            while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
                self._blink_timestamps.popleft()
            while self._sample_timestamps and self._sample_timestamps[0] < cutoff:
                self._sample_timestamps.popleft()
            while self._closed_timestamps and self._closed_timestamps[0] < cutoff:
                self._closed_timestamps.popleft()

            # blink rate remains based on recorded blink events
            blinks_in_window = len(self._blink_timestamps)
            blink_rate = (blinks_in_window / window) * 60.0 if window > 0 else 0.0

            yawns_detected = int(res.get("yawns_detected", 0))
            if yawns_detected > 0:
                self._total_yawns += yawns_detected

            # compute PERCLOS: fraction of closed samples in window (0.0 - 1.0)
            total_samples = len(self._sample_timestamps)
            closed_samples = len(self._closed_timestamps)
            perclos = (float(closed_samples) / total_samples) if total_samples > 0 else 0.0
            perclos = max(0.0, min(1.0, perclos))

            sample = {
                "timestamp": time.time(),
                "drowsiness_score": float(drowsiness),
                "fatigue_state": str(self._current_state),
                "blink_rate": float(blink_rate),
                "perclos": float(perclos),
                # iris-based perclos from detector when available
                "perclos_iris": float(res.get("perclos_iris", 0.0)),
                "backend_used": str(res.get("backend_used", "unknown")),
                "yawning_count": int(res.get("total_yawns", 0)),
                "micro_sleep_flag": bool(res.get("micro_sleep_flag", False)),
                "head_pose": {**head_pose, "confidence": confidence},
                "ear": float(ear),
                "mar": float(mar),
            }
            # inject grip confidence if GripMonitor available
            try:
                gm = getattr(self, '_grip_monitor', None)
                if gm is not None:
                    try:
                        state = gm.get_state()
                        sample['grip_confidence'] = float(state.get('grip_confidence', 0.0))
                    except Exception:
                        sample['grip_confidence'] = 0.0
                else:
                    # absent monitor -> default sentinel
                    sample['grip_confidence'] = 0.0
            except Exception:
                sample['grip_confidence'] = 0.0
            # periodic BLE broadcast while in DROWSY/MICROSLEEP
            try:
                if getattr(self, "_ble_broadcast", None) is not None and self._ble_broadcast.enable:
                    try:
                        # call maybe_broadcast on every sample to allow interval-driven TX
                        self._ble_broadcast.maybe_broadcast(self._current_state, float(drowsiness), reason if 'reason' in locals() else "", vehicle_id=self.config.get("vehicle", {}).get("id", ""))
                    except Exception:
                        logger.exception("BLE maybe_broadcast failed")
            except Exception:
                pass
            # persist sample to fatigue logger if enabled (non-blocking)
            try:
                if getattr(self, "_fatigue_logger", None) is not None:
                    try:
                        self._fatigue_logger.log_sample(sample, vehicle_hash=self._vehicle_hash, trip_id=str(self.config.get("trip_id", "")))
                    except Exception:
                        logger.exception("FatigueLogger.log_sample failed")
            except Exception:
                logger.exception("Error while trying to persist fatigue sample")
            # profiling: record total
            if self._profiler:
                total_ms = (time.perf_counter() - t0) * 1000.0
                self._profiler.record_total(total_ms)
                self._profiler.log_warning_if_exceeded()
            return sample

    def get_health(self) -> Dict[str, Any]:
        status = self._health.status
        if self._camera is not None:
            cam_status = self._camera.get_status()
        else:
            cam_status = None
        # include grip monitor status if available
        grip_status = None
        try:
            gm = getattr(self, '_grip_monitor', None)
            if gm is not None:
                st = gm.get_state()
                grip_status = {'available': bool(st.get('available', False)), 'last_updated': st.get('last_updated')}
        except Exception:
            grip_status = None

        return {
            "status": status,
            "last_successful_read": self._health.last_successful_read.isoformat() if self._health.last_successful_read else None,
            "error_count": self._health.error_count,
            "availability": self._health.availability_percent,
            "camera_status": cam_status,
            "latency_stats": self._profiler.to_dict() if self._profiler is not None else None,
            "grip_monitor_status": grip_status,
        }

    def _classify_state(self, score: float) -> str:
        """Map weighted score to discrete state."""
        # Ranges: NORMAL (< fatigue_low), TIRED [fatigue_low, drowsy),
        # DROWSY [drowsy, micro), MICROSLEEP [micro, 1.0]
        if score < self._threshold_fatigue_low:
            return "NORMAL"
        if score >= self._threshold_fatigue_low and score < self._threshold_drowsy:
            return "TIRED"
        if score >= self._threshold_drowsy and score < self._threshold_micro:
            return "DROWSY"
        return "MICROSLEEP"

    def _should_transition(self, prev: str, target: str, score: float, ts: float) -> bool:
        """Hysteresis: require dwell time and margin to avoid flapping.

        Escalation requires score >= threshold + margin; de-escalation requires
        score <= threshold - margin. Also ensure at least dwell_s seconds since
        last change.
        """
        # if same, nothing to do
        if prev == target:
            return False

        # enforce dwell time
        if (ts - self._last_state_change_ts) < self._hysteresis_dwell_s:
            return False

        # determine thresholds mapping
        # map each target to its lower-bound threshold (used for escalation)
        thresh_map = {
            "NORMAL": self._threshold_fatigue_low,
            "TIRED": self._threshold_fatigue_low,
            "DROWSY": self._threshold_drowsy,
            "MICROSLEEP": self._threshold_micro,
        }

        target_thresh = thresh_map.get(target, self._threshold_drowsy)

        # escalation: need score >= target_threshold + margin
        if prev in ("NORMAL", "TIRED") and target in ("TIRED", "DROWSY", "MICROSLEEP"):
            return score >= (target_thresh + self._hysteresis_margin)

        # de-escalation: need score <= target_threshold - margin
        return score <= (target_thresh - self._hysteresis_margin)

    def _emit_state_event(self, prev_state: str, new_state: str, score: float, reason: str) -> None:
        # increment event id
        try:
            self._event_counter += 1
        except Exception:
            self._event_counter = getattr(self, "_event_counter", 0) + 1

        ev = {
            "timestamp": int(time.time()),
            "event_id": int(self._event_counter),
            "state": new_state,
            "prev_state": prev_state,
            "score": float(score),
            "reason": reason,
            "vehicle_hash": getattr(self, "_vehicle_hash", ""),
            "trip_id": str(self.config.get("trip_id", "")),
        }
        # include available metrics when present in local scope
        try:
            if 'ear' in locals():
                ev['ear'] = float(locals().get('ear'))
        except Exception:
            pass
        try:
            if 'mar' in locals():
                ev['mar'] = float(locals().get('mar'))
        except Exception:
            pass
        try:
            if 'head_pose' in locals():
                hp = locals().get('head_pose') or {}
                ev['head_pitch'] = float(hp.get('pitch', 0.0)) if isinstance(hp, dict) else None
        except Exception:
            pass
        try:
            self._events.append(ev)
            logger.info("Emitted fatigue state event: %s", ev)
        except Exception:
            logger.exception("Failed to append fatigue state event")
        # BLE broadcast on DROWSY/MICROSLEEP (use maybe_broadcast to honor interval)
        try:
            if getattr(self, "_ble_broadcast", None) is not None and new_state in ("DROWSY", "MICROSLEEP"):
                try:
                    self._ble_broadcast.maybe_broadcast(new_state, score, reason, vehicle_id=self.config.get("vehicle", {}).get("id", ""))
                except Exception:
                    logger.exception("BLE maybe_broadcast failed")
        except Exception:
            logger.exception("Error during BLE broadcast step")

        # update API state cache
        try:
            if getattr(self, "_api_state", None) is not None:
                try:
                    self._api_state.update_status(score, new_state, ev)
                except Exception:
                    logger.exception("API state update failed")
        except Exception:
            logger.exception("Error updating API state")

        # push to central endpoint asynchronously
        try:
            if self._central_enabled and requests is not None:
                def _push():
                    attempt = 0
                    while attempt <= max(0, int(self._central_retries)):
                        try:
                            requests.post(self._central_url, json=ev, timeout=self._central_timeout)
                            return
                        except Exception:
                            logger.exception("Central push attempt %s failed", attempt)
                        attempt += 1
                try:
                    t = threading.Thread(target=_push, daemon=True)
                    t.start()
                except Exception:
                    logger.exception("Failed to start central push thread")
        except Exception:
            logger.exception("Error scheduling central push")

    def get_events(self, clear: bool = True):
        try:
            evs = list(self._events)
            if clear:
                self._events.clear()
            return evs
        except Exception:
            return []

    def is_available(self) -> bool:
        return self._health.status != "error"

    def close(self) -> None:
        try:
            if self._camera is not None:
                self._camera.release()
        except Exception:
            pass
        try:
            if getattr(self, "_alert_mgr", None) is not None:
                try:
                    self._alert_mgr.close()
                except Exception:
                    pass
        except Exception:
            pass
        # stop grip monitor if running
        try:
            gm = getattr(self, '_grip_monitor', None)
            if gm is not None:
                try:
                    gm.stop()
                except Exception:
                    logger.exception('Failed to stop GripMonitor')
        except Exception:
            pass
