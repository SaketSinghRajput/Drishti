"""Labeling subsystem: detectors and temporal labeling helpers.

This module implements AccidentDetector, NearMissDetector, TemporalLabeler,
and a lightweight ManualAnnotator. Implementations are intentionally
simulation-friendly and defensive: they operate on the window dicts produced by
the data collection pipeline and use `ConfigLoader` for thresholds.

The goal is to provide deterministic, testable logic that can be extended
later with heavier dependencies (e.g., visualization or GUI-driven
annotation).
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from config.config_loader import ConfigLoader
from utils.time_utils import parse_timestamp, get_utc_timestamp
from utils import stats_utils

logger = logging.getLogger(__name__)


"""
Note: the module-level helper `_compute_deceleration_magnitude` was removed
to avoid duplicating logic — `AccidentDetector` implements the
deceleration computation using the configured axis and gravity subtraction.
"""


def _detect_collision_overlap(camera_samples: List[Dict[str, Any]], distance_threshold_m: float) -> Optional[Dict[str, Any]]:
    """Detect a potential collision based on camera object distances and in-path flags.

    Returns a dict with object info if collision-like sample found, otherwise None.
    Expects camera_samples to be list of detection dicts with distance_m and in_path boolean.
    """
    if not camera_samples:
        return None
    for det in camera_samples:
        try:
            dist = float(det.get("distance_m", float("inf")))
            in_path = bool(det.get("in_path", False))
            if dist <= float(distance_threshold_m) and in_path:
                return {"distance_m": dist, "label": det.get("label"), "in_path": True}
        except Exception:
            continue
    return None


class AccidentDetector:
    """Detect accidents using IMU deceleration spikes and camera collision overlap.

    Configuration keys (from `sensors.labeling` path or top-level `labeling`):
      - decel_threshold_g
      - collision_distance_m
      - collision_time_window_s
    """

    def __init__(self, config: ConfigLoader):
        self.config = config
        cfg = config.get("labeling", {})
        self.decel_threshold_g = float(cfg.get("decel_threshold_g", 3.0))
        self.collision_distance_m = float(cfg.get("collision_distance_m", 2.0))
        self.collision_time_window_s = float(cfg.get("collision_time_window_s", 0.5))
        # allow decel-only accidents (looser behavior) when explicitly enabled
        self.allow_decel_only_accidents = bool(cfg.get("allow_decel_only_accidents", False))
        # how to interpret deceleration: 'norm' uses vector magnitude, 'ax' uses longitudinal axis
        self.decel_axis = str(cfg.get("decel_axis", "norm"))
        # subtract a 1g baseline from norm if sensors include gravity
        self.subtract_gravity = bool(cfg.get("subtract_gravity", False))

    def detect(self, window: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return an accident dict if detected in the supplied window, else None.

        The window is expected to be the structure written by the logger (it may
        contain keys like `imu`, `camera`, `vehicle_controls`, `metadata`).
        """
        try:
            imu = window.get("imu", [])
            camera = window.get("camera", [])
            metadata = window.get("metadata", {})

            # restrict IMU samples considered to the configured collision time window
            # (best-effort: treat the window as recent N samples based on a nominal 10Hz rate)
            try:
                n_samples = max(1, int(self.collision_time_window_s * 10))
            except Exception:
                n_samples = 1
            imu_recent = imu[-n_samples:]

            # compute max deceleration in g using configured interpretation
            max_dec_g = self._compute_deceleration_magnitude(imu_recent)

            collision = _detect_collision_overlap(camera, self.collision_distance_m)

            # system-reported crash in metadata takes precedence
            if metadata and metadata.get("system_crash"):
                typ = metadata.get("crash_type", "collision")
                severity = self._classify_severity(max_dec_g)
                return {
                    "timestamp": window.get("timestamp") or get_utc_timestamp(),
                    "type": typ,
                    "severity": severity,
                    "max_deceleration_g": float(max_dec_g),
                    "collision_info": collision,
                }

            # stricter criteria: require BOTH a deceleration spike AND a collision
            if max_dec_g >= self.decel_threshold_g and collision:
                typ = self._infer_accident_type(collision, window)
                severity = self._classify_severity(max_dec_g)
                return {
                    "timestamp": window.get("timestamp") or get_utc_timestamp(),
                    "type": typ,
                    "severity": severity,
                    "max_deceleration_g": float(max_dec_g),
                    "collision_info": collision,
                }

            # legacy/looser behavior: allow decel-only accidents when explicitly enabled
            if self.allow_decel_only_accidents and max_dec_g >= self.decel_threshold_g:
                typ = self._infer_accident_type(collision, window)
                severity = self._classify_severity(max_dec_g)
                return {
                    "timestamp": window.get("timestamp") or get_utc_timestamp(),
                    "type": typ,
                    "severity": severity,
                    "max_deceleration_g": float(max_dec_g),
                    "collision_info": collision,
                }

            return None
        except Exception as e:
            logger.exception("AccidentDetector failed: %s", e)
            return None

    def _classify_severity(self, decel_g: float) -> str:
        cfg = self.config.get("labeling", {})
        thr = cfg.get("severity_thresholds", {})
        try:
            minor = float(thr.get("minor", 3.0))
            moderate = float(thr.get("moderate", 5.0))
            severe = float(thr.get("severe", 8.0))
        except Exception:
            minor, moderate, severe = 3.0, 5.0, 8.0
        if decel_g >= severe:
            return "severe"
        if decel_g >= moderate:
            return "moderate"
        if decel_g >= minor:
            return "minor"
        return "unknown"

    def _compute_deceleration_magnitude(self, imu_samples: List[Dict[str, Any]]) -> float:
        """Compute deceleration in g according to configured axis/interpretation.

        If `decel_axis=='norm'`, compute vector magnitude and optionally subtract 1g baseline.
        If `decel_axis` is a key like 'ax' compute longitudinal axis only.
        """
        if not imu_samples:
            return 0.0
        vals = []
        for s in imu_samples:
            try:
                if self.decel_axis == "norm":
                    ax = float(s.get("ax", 0.0))
                    ay = float(s.get("ay", 0.0))
                    az = float(s.get("az", 0.0))
                    mag = math.sqrt(ax * ax + ay * ay + az * az)
                    if self.subtract_gravity:
                        mag = max(0.0, mag - 9.81)
                    vals.append(mag / 9.81)
                else:
                    v = float(s.get(self.decel_axis, 0.0))
                    vals.append(abs(v) / 9.81)
            except Exception:
                continue
        return float(max(vals) if vals else 0.0)

    def _infer_accident_type(self, collision_info: Optional[Dict[str, Any]], window: Dict[str, Any]) -> str:
        # basic heuristic: camera collision => collision; wet road + decel => slip
        if collision_info:
            return "collision"
        # check inferred road surface if present
        weather = window.get("weather", [])
        surfaces = [s.get("road_surface") for s in weather if s.get("road_surface")]
        if surfaces and any(s in ("wet", "icy") for s in surfaces):
            return "slip"
        return "off-road"


class NearMissDetector:
    """Detect near-miss events using evasive maneuvers and contextual risk.

    Uses vehicle_controls and camera/environment context to decide.
    """

    def __init__(self, config: ConfigLoader):
        self.config = config
        cfg = config.get("labeling", {})
        self.brake_spike_threshold = float(cfg.get("brake_spike_threshold", 0.7))
        self.steering_rate_threshold = float(cfg.get("steering_rate_threshold", 180.0))
        self.evasive_window_s = float(cfg.get("evasive_maneuver_window_s", 2.0))
        self.min_risk_score = float(cfg.get("near_miss_min_risk_score", 0.6))
        # optional risk weights for weather/proximity components
        rw = cfg.get("risk_weights", {}) or {}
        self.risk_weights = {
            "surface": float(rw.get("surface", 0.4)),
            "visibility": float(rw.get("visibility", 0.3)),
            "precip": float(rw.get("precip", 0.2)),
            "proximity": float(rw.get("proximity", 0.3)),
        }

    def detect(self, window: Dict[str, Any], accident_detected: bool = False) -> Optional[Dict[str, Any]]:
        if accident_detected:
            return None
        try:
            vc = window.get("vehicle_controls", [])
            camera = window.get("camera", [])
            weather = window.get("weather", [])

            # consider only recent vehicle control samples within evasive window
            try:
                n_evasive = max(1, int(self.evasive_window_s * 10))
            except Exception:
                n_evasive = 1
            vc_recent = vc[-n_evasive:]

            # detect braking spike
            max_brake = 0.0
            for s in vc_recent:
                try:
                    max_brake = max(max_brake, float(s.get("brake_pressure", 0.0)))
                except Exception:
                    continue

            # steering rate detection: compute simple diffs over samples
            steering_vals = [float(s.get("steering_angle", 0.0)) for s in vc_recent if s.get("steering_angle") is not None]
            steering_rate = 0.0
            if len(steering_vals) >= 2:
                rates = [abs(steering_vals[i] - steering_vals[i - 1]) for i in range(1, len(steering_vals))]
                steering_rate = max(rates) if rates else 0.0

            evasive = None
            if max_brake >= self.brake_spike_threshold:
                evasive = "braking"
            elif steering_rate >= self.steering_rate_threshold:
                evasive = "swerving"

            # simple contextual check: close object increases chance
            close_object = any((float(d.get("distance_m", 9999)) < 3.0) for d in camera)

            # compute a quantitative risk score from weather/context and proximity
            risk_score = self._high_risk_context(weather)
            proximity_bonus = float(self.risk_weights.get("proximity", 0.3)) if close_object else 0.0
            total_risk = min(1.0, float(risk_score) + float(proximity_bonus))

            if evasive and total_risk >= self.min_risk_score:
                res = {
                    "timestamp": window.get("timestamp"),
                    "evasive_action": evasive,
                    "max_brake": float(max_brake),
                    "steering_rate": float(steering_rate),
                    "contributing_factors": ["close_vehicle"] if close_object else [],
                    "risk_score": float(total_risk),
                }
                return res
            return None
        except Exception as e:
            logger.exception("NearMissDetector failed: %s", e)
            return None

    def _high_risk_context(self, weather_samples: List[Dict[str, Any]]) -> float:
        """Compute a risk score in [0,1] from weather samples.

        Uses configured `risk_weights` (surface, visibility, precip). Returns
        a float score that will be combined with proximity to decide near-miss.
        """
        if not weather_samples:
            return 0.0
        total = 0.0
        count = 0
        for w in weather_samples:
            try:
                score = 0.0
                vis = float(w.get("visibility_m", 1000))
                if vis < 200:
                    score += self.risk_weights.get("visibility", 0.3)
                surface = w.get("road_surface")
                if surface in ("wet", "icy"):
                    score += self.risk_weights.get("surface", 0.4)
                precip = w.get("precip_type")
                if precip in ("rain", "snow"):
                    score += self.risk_weights.get("precip", 0.2)
                total += min(1.0, score)
                count += 1
            except Exception:
                continue
        if count == 0:
            return 0.0
        return min(1.0, float(total) / float(count))


class TemporalLabeler:
    """Labels windows preceding detected events within configurable horizons.

    Usage: labeler.label_windows(windows, events) -> list of labeled windows
    where `windows` is a list of window dicts (must include `timestamp` and `window_id`)
    and `events` is list of event dicts with `timestamp` and `type` ('accident'|'near_miss').
    """

    def __init__(self, config: ConfigLoader, horizons: Optional[List[int]] = None):
        self.config = config
        cfg = config.get("labeling", {})
        if horizons is not None:
            self.prediction_horizons = list(horizons)
        else:
            ph = cfg.get("prediction_horizons", [1, 5, 10])
            # robust parsing: allow list or comma-separated string
            if isinstance(ph, list):
                self.prediction_horizons = [int(x) for x in ph]
            elif isinstance(ph, str):
                s = ph.strip()
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                parts = [p.strip() for p in s.split(",") if p.strip()]
                try:
                    self.prediction_horizons = [int(p) for p in parts]
                except Exception:
                    self.prediction_horizons = [1, 5, 10]
            else:
                try:
                    self.prediction_horizons = [int(ph)]
                except Exception:
                    self.prediction_horizons = [1, 5, 10]

    def label_windows(self, windows: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # prepare map from timestamps to windows (parse timestamps to seconds)
        windows_sorted = sorted(windows, key=lambda w: w.get("timestamp"))
        # convert timestamps to float seconds for comparison
        win_times = [self._ts_to_epoch(w.get("timestamp")) for w in windows_sorted]
        labeled: List[Dict[str, Any]] = []

        for i, w in enumerate(windows_sorted):
            labeled_win = dict(w)
            labeled_win.setdefault("labels", {})
            # base labels
            labeled_win["labels"]["is_accident"] = False
            labeled_win["labels"]["is_near_miss"] = False
            labeled_win["labels"]["time_to_accident_s"] = None
            labeled_win["labels"]["accident_type"] = None
            # per-horizon labels (is_accident_h{h}, is_near_miss_h{h})
            for h in self.prediction_horizons:
                labeled_win["labels"][f"is_accident_h{h}"] = False
                labeled_win["labels"][f"is_near_miss_h{h}"] = False
            # find nearest upcoming event within any horizon
            t = win_times[i]
            nearest_event = None
            nearest_dt = None
            for ev in events:
                # respect trip boundaries: skip events from other trips
                if ev.get("trip_id") is not None and ev.get("trip_id") != w.get("trip_id"):
                    continue
                try:
                    ev_t = self._ts_to_epoch(ev.get("timestamp"))
                except Exception:
                    continue
                dt = ev_t - t
                if dt >= 0 and (nearest_dt is None or dt < nearest_dt):
                    nearest_dt = dt
                    nearest_event = ev

            if nearest_event and nearest_dt is not None:
                # for each configured horizon, set per-horizon booleans. A window
                # may be positive for multiple horizons (e.g., within 10s and 5s)
                for h in self.prediction_horizons:
                    if nearest_dt <= float(h):
                        if nearest_event.get("type") != "near_miss":
                            labeled_win["labels"][f"is_accident_h{h}"] = True
                            labeled_win["labels"][f"is_near_miss_h{h}"] = False
                            labeled_win["labels"]["is_accident"] = True
                            labeled_win["labels"]["is_near_miss"] = False
                            labeled_win["labels"]["time_to_accident_s"] = float(nearest_dt)
                            labeled_win["labels"]["accident_type"] = nearest_event.get("subtype") or nearest_event.get("type")
                        else:
                            labeled_win["labels"][f"is_near_miss_h{h}"] = True
                            labeled_win["labels"][f"is_accident_h{h}"] = False
                            labeled_win["labels"]["is_near_miss"] = True
                            labeled_win["labels"]["is_accident"] = False
                            labeled_win["labels"]["near_miss_type"] = nearest_event.get("evasive_action")
                            labeled_win["labels"]["time_to_accident_s"] = float(nearest_dt)
                            labeled_win["labels"]["accident_type"] = None

            labeled.append(labeled_win)

        return labeled

    def _ts_to_epoch(self, ts: Optional[str]) -> float:
        try:
            dt = parse_timestamp(ts)
            return dt.timestamp()
        except Exception:
            return 0.0


class ManualAnnotator:
    """Command-line helper to review and save annotations.

    This is intentionally minimal; tests exercise programmatic methods rather
    than interactive I/O. The `review_window` method accepts an input function
    to facilitate testing/mocking.
    """

    def __init__(self, config: ConfigLoader):
        self.config = config

    def review_window(self, window: Dict[str, Any], auto_labels: Optional[Dict[str, Any]] = None, input_fn=input) -> Dict[str, Any]:
        summary = {
            "window_id": window.get("window_id"),
            "timestamp": window.get("timestamp"),
        }
        # show a succinct summary
        print("Window:", summary)
        if auto_labels:
            print("Auto labels:", json.dumps(auto_labels, indent=2))

        resp = input_fn("Is this an accident? (y/n): ")
        is_acc = resp.strip().lower() == "y"
        acc_type = None
        severity = None
        is_near = False
        confidence = None
        requires_review = False
        if is_acc:
            acc_type = input_fn("Accident type (collision/slip/off-road) [collision]: ") or "collision"
            severity = input_fn("Severity (minor/moderate/severe) [minor]: ") or "minor"
            confidence = input_fn("Confidence (0-1) [0.9]: ") or "0.9"
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.9
        else:
            # ask if near-miss
            resp2 = input_fn("Is this a near-miss? (y/n): ")
            is_near = resp2.strip().lower() == "y"
            if is_near:
                acc_type = None
                severity = None
                evasive = input_fn("Evasive action (braking/swerving): ") or "braking"
                confidence = input_fn("Confidence (0-1) [0.7]: ") or "0.7"
                try:
                    confidence = float(confidence)
                except Exception:
                    confidence = 0.7
            else:
                evasive = None

        req = input_fn("Mark for review? (y/n): ")
        requires_review = req.strip().lower() == "y"

        label = {
            "window_id": window.get("window_id"),
            "timestamp": window.get("timestamp"),
            "is_accident": bool(is_acc),
            "accident_type": acc_type,
            "severity": severity,
            "is_near_miss": bool(is_near),
            "evasive_action": locals().get("evasive", None),
            "confidence": float(confidence) if confidence is not None else None,
            "requires_review": bool(requires_review),
            "annotator_id": "cli",
            "annotation_timestamp": get_utc_timestamp(),
            "notes": "",
        }
        return label


__all__ = ["AccidentDetector", "NearMissDetector", "TemporalLabeler", "ManualAnnotator"]
