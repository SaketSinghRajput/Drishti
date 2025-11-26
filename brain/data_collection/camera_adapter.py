"""CameraAdapter: compute window-level camera features from frame detections.

Follows the same pattern as BLEMeshAdapter and DriverFatigueAdapter: analyze a list of
frame dicts and return an aggregate features dict or None when no features are computed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from utils import stats_utils
try:
    from data_collection.camera_adapter_intent import compute_pedestrian_intent_features, compute_cyclist_features
except Exception:
    try:
        from .camera_adapter_intent import compute_pedestrian_intent_features, compute_cyclist_features
    except Exception:
        compute_pedestrian_intent_features = None
        compute_cyclist_features = None

logger = logging.getLogger(__name__)


class CameraAdapter:
    def __init__(self, config):
        self.config = config
        self.cfg = config.get("sensors.camera", {})
        self.enable = bool(self.cfg.get("enable_aggregate_features", True))
        # optional intent/cyclist aggregation toggles
        self.enable_pedestrian_intent_features = bool(self.cfg.get("enable_pedestrian_intent_features", False))
        self.enable_cyclist_features = bool(self.cfg.get("enable_cyclist_features", False))
        self.classes_of_interest = list(self.cfg.get("object_classes_of_interest", ["car", "person", "bicycle", "truck"]))
        # coerce critical thresholds to numeric list and sort ascending
        raw_thresh = self.cfg.get("critical_distance_thresholds_m", [5, 10, 20])
        try:
            self.critical_thresholds = sorted([float(x) for x in list(raw_thresh)])
        except Exception:
            # fallback defaults
            self.critical_thresholds = [5.0, 10.0, 20.0]
        # Enforce or warn about schema-backed default thresholds
        strict_schema = bool(self.config.get("data_collection.strict_schema", False))
        if self.critical_thresholds != [5.0, 10.0, 20.0]:
            if strict_schema:
                # Fail fast if strict schema enforcement is enabled
                raise ValueError(
                    "Configured critical_distance_thresholds_m %s does not match required schema-backed default [5,10,20] while data_collection.strict_schema=True"
                    % (self.critical_thresholds,)
                )
            else:
                logger.warning(
                    "Configured critical_distance_thresholds_m %s differs from schema-backed default [5,10,20]. "
                    "Changing thresholds requires updating config/schema/sensor_window_schema.json to avoid validation issues.",
                    self.critical_thresholds,
                )

    def analyze(self, samples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            if not self.enable or not samples:
                return None

            sample_count = len(samples)
            per_frame_counts = []
            closest_per_frame = []
            # dynamic bin keys: objects_within_<int(t)>m
            objects_within = {t: 0 for t in self.critical_thresholds}
            objects_in_path = []
            lane_offsets = []
            flagged_frames = 0
            frames_with_detections = 0
            detection_classes = {}
            rel_speeds = []
            pedestrian_frames = 0
            pedestrian_near_crossing = 0
            visibility = []

            for f in samples:
                objs = f.get("objects", []) or []
                per_frame_counts.append(len(objs))
                if objs:
                    frames_with_detections += 1
                # filter by classes of interest
                filtered_objs = [o for o in objs if isinstance(o, dict) and (not self.classes_of_interest or o.get("class") in self.classes_of_interest)]
                # closest
                dists = [o.get("distance_m", 999.0) for o in filtered_objs]
                closest = min(dists) if dists else 999.0
                closest_per_frame.append(closest)
                # counts in thresholds
                for o in filtered_objs:
                    dist = float(o.get("distance_m", 999.0))
                    for t in self.critical_thresholds:
                        if dist <= t:
                            objects_within[t] += 1
                    cls = o.get("class")
                    if cls:
                        detection_classes[cls] = detection_classes.get(cls, 0) + 1
                    rs = o.get("rel_speed_mps")
                    if isinstance(rs, (int, float)):
                        rel_speeds.append(rs)
                    if cls == "person":
                        pedestrian_frames += 1
                objects_in_path.append(int(f.get("objects_in_path", 0)))
                lane_offsets.append(float(f.get("lane_center_offset_m", 0.0)))
                if f.get("frame_flagged_for_retention"):
                    flagged_frames += 1
                if f.get("visibility_estimate_m") is not None:
                    visibility.append(float(f.get("visibility_estimate_m", 0.0)))

            total_detections = sum(per_frame_counts)

            approaching = sum(1 for s in rel_speeds if float(s) < 0)
            receding = sum(1 for s in rel_speeds if float(s) > 0)

            features = {
                "total_detections": int(total_detections),
                "avg_detections_per_frame": float(stats_utils.safe_mean(per_frame_counts)),
                "detection_classes": detection_classes,
                "unique_classes": list(detection_classes.keys()),
                # If closest_per_frame is empty, keep 999.0 (no detections)
                "min_distance_m": float(stats_utils.safe_min(closest_per_frame) or 999.0),
                "avg_closest_distance_m": float(stats_utils.safe_mean(closest_per_frame) or 999.0),
                "distance_std": float(stats_utils.safe_std(closest_per_frame) or 0.0),
                # dynamic distance bins
                **{f"objects_within_{int(t)}m": int(objects_within.get(t, 0)) for t in self.critical_thresholds},
                "avg_objects_in_path": float(stats_utils.safe_mean(objects_in_path) or 0.0),
                "max_objects_in_path": int(stats_utils.safe_max(objects_in_path) or 0),
                "obstacle_in_path_frames": int(sum(1 for v in objects_in_path if v > 0)),
                "obstacle_in_path_ratio": float((sum(1 for v in objects_in_path if v > 0) / sample_count) if sample_count else 0.0),
                "avg_lane_offset_m": float(stats_utils.safe_mean(lane_offsets) or 0.0),
                "lane_offset_std": float(stats_utils.safe_std(lane_offsets) or 0.0),
                "left_lane_clear_ratio": float(sum(1 for f in samples if f.get("is_left_lane_clear", True)) / sample_count if sample_count else 0.0),
                "right_lane_clear_ratio": float(sum(1 for f in samples if f.get("is_right_lane_clear", True)) / sample_count if sample_count else 0.0),
                "pedestrian_frames": int(pedestrian_frames),
                "pedestrian_near_crossing_frames": int(sum(1 for f in samples if f.get("pedestrian_near_crossing_flag"))),
                "pedestrian_risk_ratio": float((sum(1 for f in samples if f.get("pedestrian_near_crossing_flag")) / sample_count) if sample_count else 0.0),
                "approaching_objects_count": int(approaching),
                "receding_objects_count": int(receding),
                "max_approach_speed_mps": float(min(rel_speeds) if rel_speeds else 0.0),
                "avg_rel_speed_mps": float(stats_utils.safe_mean(rel_speeds) or 0.0),
                "avg_visibility_estimate_m": float(stats_utils.safe_mean(visibility) or 0.0),
                "min_visibility_m": float(stats_utils.safe_min(visibility) or 0.0),
                "flagged_frames_count": int(flagged_frames),
                "flagged_frames_ratio": float(flagged_frames / sample_count if sample_count else 0.0),
                "sample_count": int(sample_count),
                "frames_with_detections": int(frames_with_detections),
                "detection_rate": float(frames_with_detections / sample_count if sample_count else 0.0),
            }

            # Merge pedestrian intent & cyclist features if available and configured
            try:
                if self.enable_pedestrian_intent_features and compute_pedestrian_intent_features is not None:
                    intent_feats = compute_pedestrian_intent_features(samples)
                    if intent_feats:
                        features.update(intent_feats)
            except Exception:
                logger.exception("Failed to compute pedestrian intent features")
            try:
                if self.enable_cyclist_features and compute_cyclist_features is not None:
                    c_feats = compute_cyclist_features(samples)
                    if c_feats:
                        features.update(c_feats)
            except Exception:
                logger.exception("Failed to compute cyclist features")

            return features
        except Exception as e:
            logger.exception("CameraAdapter analyze failed: %s", e)
            return None
