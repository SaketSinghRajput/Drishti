"""DriverFatigueAdapter: Compute higher-level features from driver_fatigue samples.

This adapter ingests a list of driver_fatigue samples (timestamped dicts produced by
the `DriverFatigueHandler`) and computes rolling statistics, sudden-change detection,
trend analysis, head-pose aggregation, and event counting.

It is intentionally defensive: missing fields are skipped, empty sample lists return
an empty dict (caller may choose to drop it), and exceptions during analysis are
logged by callers (the adapter itself avoids raising for common data issues).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.config_loader import ConfigLoader
from utils.time_utils import parse_timestamp, timestamp_to_epoch
from utils.stats_utils import (
    safe_mean,
    safe_std,
    safe_min,
    safe_max,
    linear_trend,
)
from data_collection.driver_fatigue_adapter_hand import compute_grip_features

logger = logging.getLogger(__name__)


class DriverFatigueAdapter:
    """Adapter to compute driver fatigue features from sample arrays.

    Config parameters read from `sensors.driver_fatigue.*`:
      - sudden_drowsiness_threshold (float)
      - sudden_change_window_s (float)
      - trend_slope_threshold (float)
      - enable_rolling_stats (bool)
      - enable_sudden_change_detection (bool)
      - enable_trend_analysis (bool)
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        sect = "sensors.driver_fatigue"
        self.sudden_threshold = float(self.config.get(f"{sect}.sudden_drowsiness_threshold", 0.3))
        self.sudden_window_s = float(self.config.get(f"{sect}.sudden_change_window_s", 3.0))
        self.trend_slope_threshold = float(self.config.get(f"{sect}.trend_slope_threshold", 0.01))
        self.enable_rolling = bool(self.config.get(f"{sect}.enable_rolling_stats", True))
        self.enable_sudden = bool(self.config.get(f"{sect}.enable_sudden_change_detection", True))
        self.enable_trend = bool(self.config.get(f"{sect}.enable_trend_analysis", True))
        # grip features toggle
        try:
            self.enable_grip_features = bool(self.config.get(f"{sect}.hand_landmark.enable_grip_features", False))
        except Exception:
            self.enable_grip_features = False

    def analyze(self, samples: List[Dict[str, Any]], events: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Main entry point. Returns features dict or None for empty input.
        """
        if not samples:
            return None

        # defensive copy of values we will read
        try:
            # compute per-metric stats
            features: Dict[str, Any] = {}

            if self.enable_rolling:
                features.update(self._compute_rolling_statistics(samples, "drowsiness_score"))
                features.update(self._compute_rolling_statistics(samples, "blink_rate"))
                features.update(self._compute_rolling_statistics(samples, "perclos"))

            # sudden change detection
            if self.enable_sudden:
                features.update(self._detect_sudden_changes(samples))

            # trends
            if self.enable_trend:
                features["drowsiness_trend"] = self._compute_trend(samples, "drowsiness_score")
                features["blink_rate_trend"] = self._compute_trend(samples, "blink_rate")

            # head pose aggregation
            features.update(self._aggregate_head_pose(samples))

            # event counting
            features.update(self._count_events(samples))

            features["sample_count"] = len(samples)

            # State/event-derived features (optional)
            try:
                state_features = self._compute_state_features(samples, events)
                features.update(state_features)
            except Exception:
                logger.exception("Failed to compute state-derived features")

            # Grip features (optional)
            if self.enable_grip_features:
                try:
                    sample_rate = float(self.config.get(f"{sect}.sample_rate", 1.0))
                    gf = compute_grip_features(samples, sample_rate_hz=sample_rate)
                    if gf:
                        features.update(gf)
                except Exception:
                    logger.exception("Failed to compute grip features")

            return features
        except Exception as e:
            logger.exception("DriverFatigueAdapter analysis failed: %s", e)
            return None

    def _compute_rolling_statistics(self, samples: List[Dict], metric_key: str) -> Dict[str, float]:
        vals: List[float] = []
        for s in samples:
            v = s.get(metric_key)
            if isinstance(v, (int, float)):
                vals.append(float(v))

        return {
            f"{metric_key}_mean": safe_mean(vals),
            f"{metric_key}_std": safe_std(vals),
            f"{metric_key}_min": safe_min(vals),
            f"{metric_key}_max": safe_max(vals),
        }

    def _detect_sudden_changes(self, samples: List[Dict]) -> Dict[str, Any]:
        # Sort samples by timestamp
        items = []
        for s in samples:
            ts = s.get("timestamp")
            if not ts:
                continue
            try:
                epoch = timestamp_to_epoch(ts)
            except Exception:
                try:
                    epoch = parse_timestamp(ts).timestamp()
                except Exception:
                    continue
            items.append((epoch, s.get("drowsiness_score", 0.0), s))

        items.sort(key=lambda x: x[0])

        max_delta = 0.0
        max_abs_delta = 0.0
        spike_ts = None
        sudden_flag = False

        for i in range(1, len(items)):
            prev_t, prev_v, _ = items[i - 1]
            cur_t, cur_v, cur_s = items[i]
            try:
                # signed delta: positive means an increase in drowsiness
                delta = float(cur_v) - float(prev_v)
            except Exception:
                continue
            dt = cur_t - prev_t
            if dt <= 0:
                continue
            # track largest signed increase (positive) and largest magnitude
            if delta > max_delta:
                max_delta = delta
            if abs(delta) > max_abs_delta:
                max_abs_delta = abs(delta)
            # consider only upward changes as drowsiness increases
            if delta >= self.sudden_threshold and dt <= self.sudden_window_s:
                sudden_flag = True
                spike_ts = cur_s.get("timestamp")
                # break early on first spike
                break
        result: Dict[str, Any] = {
            "sudden_drowsiness_increase": sudden_flag,
            # signed maximum increase (positive = increase)
            "max_drowsiness_delta": max_delta,
            # magnitude-only maximum delta for consumers who want absolute size
            "max_drowsiness_abs_delta": max_abs_delta,
        }

        # only include the spike timestamp if we actually detected a spike
        if spike_ts:
            result["drowsiness_spike_timestamp"] = spike_ts

        return result

    def _compute_trend(self, samples: List[Dict], metric_key: str) -> str:
        vals = []
        for s in samples:
            v = s.get(metric_key)
            if isinstance(v, (int, float)):
                vals.append(float(v))

        # use linear_trend util which returns 'increasing'|'decreasing'|'stable'
        try:
            return linear_trend(vals, slope_threshold=self.trend_slope_threshold)
        except TypeError:
            # older signature without slope_threshold
            return linear_trend(vals)
        except Exception:
            return "stable"

    def _aggregate_head_pose(self, samples: List[Dict]) -> Dict[str, float]:
        pitch = []
        yaw = []
        roll = []
        for s in samples:
            hp = s.get("head_pose") or {}
            if isinstance(hp, dict):
                p = hp.get("pitch")
                y = hp.get("yaw")
                r = hp.get("roll")
                if isinstance(p, (int, float)):
                    pitch.append(float(p))
                if isinstance(y, (int, float)):
                    yaw.append(float(y))
                if isinstance(r, (int, float)):
                    roll.append(float(r))

        return {
            "head_pose_mean_pitch": safe_mean(pitch),
            "head_pose_mean_yaw": safe_mean(yaw),
            "head_pose_mean_roll": safe_mean(roll),
        }

    def _count_events(self, samples: List[Dict]) -> Dict[str, int]:
        total_yawns = 0
        micro_sleep_events = 0
        for s in samples:
            try:
                total_yawns += int(s.get("yawning_count", 0) or 0)
            except Exception:
                pass
            if s.get("micro_sleep_flag"):
                micro_sleep_events += 1

        return {"total_yawns": total_yawns, "micro_sleep_events": micro_sleep_events}

    def _compute_state_features(self, samples: List[Dict], events: Optional[List[Dict]]) -> Dict[str, Any]:
        """Compute features derived from handler state events and sample state tags.

        Returns a dict of optional features; defensive and returns empty dict on errors.
        """
        try:
            out: Dict[str, Any] = {}
            if events:
                out["state_transitions_count"] = len(events)
                # map states to numeric levels
                level_map = {"NORMAL": 0, "TIRED": 1, "DROWSY": 2, "MICROSLEEP": 3}
                max_level = 0
                drowsy_time = 0.0
                micro_time = 0.0
                drowsy_count = 0
                micro_count = 0
                # compute durations between events
                events_sorted = sorted(events, key=lambda e: e.get("timestamp", 0))
                prev_ts = None
                prev_state = None
                for ev in events_sorted:
                    st = ev.get("state")
                    t = ev.get("timestamp", 0)
                    lvl = level_map.get(st, 0)
                    if lvl > max_level:
                        max_level = lvl
                    if st == "DROWSY":
                        drowsy_count += 1
                    if st == "MICROSLEEP":
                        micro_count += 1
                    if prev_ts is not None and prev_state is not None:
                        dur = float(t) - float(prev_ts)
                        if prev_state == "DROWSY":
                            drowsy_time += max(0.0, dur)
                        if prev_state == "MICROSLEEP":
                            micro_time += max(0.0, dur)
                    prev_ts = t
                    prev_state = st

                out["max_state_level"] = int(max_level)
                out["time_in_drowsy_s"] = float(drowsy_time)
                out["time_in_microsleep_s"] = float(micro_time)
                out["drowsy_event_count"] = int(drowsy_count)
                out["microsleep_event_count"] = int(micro_count)

            # derive current state from sample tags if present
            try:
                states = [s.get("fatigue_state") for s in samples if s.get("fatigue_state")]
                if states:
                    # mode (most common)
                    from collections import Counter

                    c = Counter(states)
                    out["state_mode"] = c.most_common(1)[0][0]
                    out["current_state"] = states[-1]
            except Exception:
                pass

            return out
        except Exception:
            return {}
