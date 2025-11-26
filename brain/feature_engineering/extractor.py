"""Feature extraction utilities.

Provides FeatureExtractor which converts a sensor window dict into a flat
feature dictionary or a pandas.DataFrame row. Designed to be defensive and
work with or without numpy/pandas available.
"""
from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

from config.config_loader import ConfigLoader
from utils import stats_utils

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

from feature_engineering import feature_utils
try:
    from feature_engineering.extractor_interactions import grip_loss_speed_interaction, pedestrian_intent_proximity_interaction
except Exception:
    grip_loss_speed_interaction = None
    pedestrian_intent_proximity_interaction = None

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts raw, derived and interaction features from a window dict.

    Basic usage:
        fe = FeatureExtractor(ConfigLoader('config/default_config.yaml'))
        feat = fe.extract(window)
        df = fe.extract_to_dataframe([window, ...])
    """

    def __init__(self, config: ConfigLoader):
        self.config = config
        cfg = config.get("feature_engineering", {}) or {}
        # high-level feature toggles
        self.enable_raw = bool(cfg.get("enable_raw_sensor_stats", True))
        self.enable_derived = bool(cfg.get("enable_derived_features", True))
        self.enable_interactions = bool(cfg.get("enable_interaction_features", True))
        # fine-grained toggles
        self.extract_gps = bool(cfg.get("extract_gps_features", True))
        self.extract_imu = bool(cfg.get("extract_imu_features", True))
        self.extract_vc = bool(cfg.get("extract_vehicle_control_features", True))
        self.compute_jerk = bool(cfg.get("compute_jerk", True))
        self.compute_steering_rate = bool(cfg.get("compute_steering_rate", True))
        self.compute_speed_deltas = bool(cfg.get("compute_speed_deltas", True))
        self.compute_lane_offset_trend = bool(cfg.get("compute_lane_offset_trend", True))
        self.compute_drowsiness_interactions = bool(cfg.get("compute_drowsiness_interactions", True))
        self.compute_visibility_interactions = bool(cfg.get("compute_visibility_interactions", True))
        self.compute_proximity_interactions = bool(cfg.get("compute_proximity_interactions", True))
        self.compute_road_condition_interactions = bool(cfg.get("compute_road_condition_interactions", True))
        # new optional interaction toggles
        self.enable_grip_interactions = bool(cfg.get("enable_grip_interactions", False))
        self.enable_pedestrian_intent_interactions = bool(cfg.get("enable_pedestrian_intent_interactions", False))

        self.flatten_nested_features = bool(cfg.get("flatten_nested_features", True))
        self.include_metadata = bool(cfg.get("include_metadata", True))
        self.include_labels = bool(cfg.get("include_labels", True))

        self.missing_sentinel = float(cfg.get("missing_value_sentinel", -999.0))
        self.include_availability = bool(cfg.get("include_availability_flags", True))
        self.use_prefixes = bool(cfg.get("use_prefixes", False))
        self.prefix_raw = cfg.get("prefix_raw_sensors", "raw_") if self.use_prefixes else ""
        self.prefix_derived = cfg.get("prefix_derived", "derived_") if self.use_prefixes else ""
        self.prefix_inter = cfg.get("prefix_interaction", "interaction_") if self.use_prefixes else ""

    def extract(self, window: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # metadata and labels
        if self.include_metadata:
            out.update(self._extract_metadata_features(window))

        # availability flags
        gps = window.get("gps", [])
        imu = window.get("imu", [])
        vc = window.get("vehicle_controls", [])
        camera = window.get("camera", [])
        if self.include_availability:
            out["gps_available"] = bool(gps)
            out["imu_available"] = bool(imu)
            out["vc_available"] = bool(vc)
            out["camera_available"] = bool(camera)

        # Raw sensor aggregations
        if self.enable_raw:
            raw_feats = self._extract_raw_sensor_features(gps, imu, vc)
            out.update(raw_feats)

        # Derived features
        if self.enable_derived:
            derived = self._extract_derived_features(gps, imu, vc, camera)
            out.update(derived)

        # Consolidate adapter features (flatten nested dicts)
        adapter_feats = self._consolidate_adapter_features(window)
        out.update(adapter_feats)

        # Interaction features
        if self.enable_interactions:
            inter = self._extract_interaction_features(out)
            out.update(inter)

        # labels are emitted by _extract_metadata_features when include_metadata and include_labels are enabled

        return out

    def extract_to_dataframe(self, windows: List[Dict[str, Any]]):
        rows = [self.extract(w) for w in windows]
        if pd is not None:
            return pd.DataFrame(rows)
        # minimal fallback: try to import pandas at runtime or return list
        try:
            import pandas as _pd

            return _pd.DataFrame(rows)
        except Exception:
            return rows

    def _extract_raw_sensor_features(self, gps: List[Dict[str, Any]], imu: List[Dict[str, Any]], vc: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # GPS: speed_kph and heading stats (if enabled)
        if self.extract_gps:
            speeds = feature_utils.extract_time_series_values(gps, "speed_kph")
            out.update(feature_utils._agg_stats(speeds, prefix=self.prefix_raw + "speed_", sentinel=self.missing_sentinel))
            # heading (optional, may not exist in some windows)
            headings = feature_utils.extract_time_series_values(gps, "heading")
            if headings:
                out.update(feature_utils._agg_stats(headings, prefix=self.prefix_raw + "heading_", sentinel=self.missing_sentinel))

        # IMU: ax, ay, az, yaw_rate (if enabled)
        if self.extract_imu:
            ax = feature_utils.extract_time_series_values(imu, "ax")
            ay = feature_utils.extract_time_series_values(imu, "ay")
            az = feature_utils.extract_time_series_values(imu, "az")
            yaw = feature_utils.extract_time_series_values(imu, "yaw_rate")
            out.update(feature_utils._agg_stats(ax, prefix=self.prefix_raw + "ax_", sentinel=self.missing_sentinel))
            out.update(feature_utils._agg_stats(ay, prefix=self.prefix_raw + "ay_", sentinel=self.missing_sentinel))
            out.update(feature_utils._agg_stats(az, prefix=self.prefix_raw + "az_", sentinel=self.missing_sentinel))
            out.update(feature_utils._agg_stats(yaw, prefix=self.prefix_raw + "yaw_rate_", sentinel=self.missing_sentinel))
            # total acceleration magnitude
            mag = feature_utils.compute_magnitude(imu, ["ax", "ay", "az"])
            if mag:
                out[self.prefix_raw + "total_acceleration_mean"] = float(sum(mag) / len(mag))
                out[self.prefix_raw + "total_acceleration_max"] = float(max(mag))
            else:
                out[self.prefix_raw + "total_acceleration_mean"] = self.missing_sentinel
                out[self.prefix_raw + "total_acceleration_max"] = self.missing_sentinel

        # vehicle controls
        if self.extract_vc:
            steer = feature_utils.extract_time_series_values(vc, "steering_angle")
            brake = feature_utils.extract_time_series_values(vc, "brake_pressure")
            throttle = feature_utils.extract_time_series_values(vc, "throttle_position")
            out.update(feature_utils._agg_stats(steer, prefix=self.prefix_raw + "steer_", sentinel=self.missing_sentinel))
            out.update(feature_utils._agg_stats(brake, prefix=self.prefix_raw + "brake_", sentinel=self.missing_sentinel))
            out.update(feature_utils._agg_stats(throttle, prefix=self.prefix_raw + "throttle_", sentinel=self.missing_sentinel))
        else:
            # when vehicle control extraction disabled, set sentinels for common keys
            out.update({
                self.prefix_raw + "steer_mean": self.missing_sentinel,
                self.prefix_raw + "brake_mean": self.missing_sentinel,
                self.prefix_raw + "throttle_mean": self.missing_sentinel,
                self.prefix_raw + "brake_flag_ratio": self.missing_sentinel,
            })

        # brake flag ratio
        if self.extract_vc:
            if vc:
                flags = [1.0 for s in vc if s.get("brake_flag")]
                out[self.prefix_raw + "brake_flag_ratio"] = float(len(flags)) / max(1.0, float(len(vc)))
            else:
                out[self.prefix_raw + "brake_flag_ratio"] = self.missing_sentinel

        return out

    def _extract_derived_features(self, gps, imu, vc, camera) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # jerk from acceleration
        if self.compute_jerk:
            try:
                jerk = feature_utils.compute_jerk_from_acceleration(imu)
                for k, v in jerk.items():
                    out[self.prefix_derived + k] = v
            except Exception:
                logger.exception("Jerk computation failed")
                out[self.prefix_derived + "jerk_magnitude_mean"] = self.missing_sentinel
        else:
            out[self.prefix_derived + "jerk_magnitude_mean"] = self.missing_sentinel

        # steering rate
        if self.compute_steering_rate:
            try:
                sr = feature_utils.compute_rate_of_change(vc, "steering_angle")
                out[self.prefix_derived + "steering_rate_mean"] = float(sr.get("mean", self.missing_sentinel))
            except Exception:
                out[self.prefix_derived + "steering_rate_mean"] = self.missing_sentinel
        else:
            out[self.prefix_derived + "steering_rate_mean"] = self.missing_sentinel

        # speed deltas
        if self.compute_speed_deltas:
            try:
                speeds = feature_utils.extract_time_series_values(gps, "speed_kph")
                delta = feature_utils.compute_delta_features(speeds, "speed_kph")
                for k, v in delta.items():
                    out[self.prefix_derived + k] = v
            except Exception:
                out[self.prefix_derived + "speed_kph_delta_total"] = self.missing_sentinel
        else:
            out[self.prefix_derived + "speed_kph_delta_total"] = self.missing_sentinel

        # lane offset trend from camera features (if present)
        if self.compute_lane_offset_trend:
            try:
                slope = feature_utils.compute_trend_slope(camera, "avg_lane_offset_m")
                out[self.prefix_derived + "lane_offset_trend_slope"] = float(slope)
            except Exception:
                out[self.prefix_derived + "lane_offset_trend_slope"] = self.missing_sentinel
        else:
            out[self.prefix_derived + "lane_offset_trend_slope"] = self.missing_sentinel

        return out

    def _consolidate_adapter_features(self, window: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        adapters = ["driver_fatigue_features", "ble_mesh_features", "camera_features", "environment_features"]
        for a in adapters:
            src = window.get(a) or {}
            if self.flatten_nested_features:
                flat = feature_utils.flatten_dict(src, prefix=(a + "_") if src else "")
                out.update(flat)
            else:
                # include nested dict as-is under adapter key
                out[a] = src
        return out

    def _extract_metadata_features(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata and identifiers from a window into a flat dict.

        Includes `window_id`, `trip_id`, `window_timestamp`, and optional
        fields like `vehicle_hash`, `data_quality_score`, and missing_sensors.
        Also emits labels prefixed with `label_` if `include_labels` is True.
        """
        out: Dict[str, Any] = {}
        out["window_id"] = window.get("window_id")
        out["trip_id"] = window.get("trip_id")
        out["window_timestamp"] = window.get("timestamp")

        # optional richer metadata keys
        meta = window.get("metadata") or {}
        if meta:
            out["vehicle_hash"] = meta.get("vehicle_hash")
            out["data_quality_score"] = meta.get("data_quality_score")
            # If schema provides missing_sensors list, expose count and list
            missing = meta.get("missing_sensors")
            if isinstance(missing, list):
                out["missing_sensors_count"] = len(missing)
                out["missing_sensors_list"] = ",".join(missing) if missing else ""

        # labels (optionally included)
        if self.include_labels and window.get("labels"):
            for k, v in window.get("labels", {}).items():
                out[f"label_{k}"] = v

        return out

    def _extract_interaction_features(self, current_features: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # helper to safely fetch
        def _g(k):
            v = current_features.get(k)
            try:
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        speed_mean = _g(self.prefix_raw + "speed_mean") or _g("speed_mean")
        drows = _g("driver_fatigue_features_drowsiness_score_mean") or _g("drowsiness_score_mean")
        visibility = _g("environment_features_visibility_mean_m") or 0.0
        curvature = _g("environment_features_curvature_mean_m") or 0.0
        wet_flag = 1.0 if (current_features.get("environment_features_road_surface") in ("wet", "icy")) else 0.0

        if self.compute_drowsiness_interactions:
            out[self.prefix_inter + "drowsiness_speed_interaction"] = drows * speed_mean
        else:
            out[self.prefix_inter + "drowsiness_speed_interaction"] = 0.0

        if self.compute_visibility_interactions:
            out[self.prefix_inter + "visibility_speed_interaction"] = visibility * speed_mean
        else:
            out[self.prefix_inter + "visibility_speed_interaction"] = 0.0

        if self.compute_road_condition_interactions:
            out[self.prefix_inter + "curvature_speed_interaction"] = curvature * speed_mean
            out[self.prefix_inter + "wet_road_speed_interaction"] = wet_flag * speed_mean
        else:
            out[self.prefix_inter + "curvature_speed_interaction"] = 0.0
            out[self.prefix_inter + "wet_road_speed_interaction"] = 0.0

        if self.compute_proximity_interactions:
            out[self.prefix_inter + "brake_drowsiness_interaction"] = _g(self.prefix_raw + "brake_mean") * drows
            # nearest object inverse distance
            min_dist = current_features.get("camera_features_min_distance_m") or 0.0
            out[self.prefix_inter + "nearest_object_speed_interaction"] = (1.0 / max(1e-6, float(min_dist))) * speed_mean if min_dist else 0.0
        else:
            out[self.prefix_inter + "brake_drowsiness_interaction"] = 0.0
            out[self.prefix_inter + "nearest_object_speed_interaction"] = 0.0

        # Additional interactions: grip × speed and pedestrian intent × proximity
        try:
            if self.enable_grip_interactions and grip_loss_speed_interaction is not None:
                # build driver and camera feature views
                driver_feats = {k.replace('driver_fatigue_features_', ''): v for k, v in current_features.items() if k.startswith('driver_fatigue_features_')}
                camera_feats = {k.replace('camera_features_', ''): v for k, v in current_features.items() if k.startswith('camera_features_')}
                grip_out = grip_loss_speed_interaction(driver_feats, camera_feats)
                if grip_out:
                    out.update(grip_out)
        except Exception:
            logger.exception('Failed to compute grip interactions')

        try:
            if self.enable_pedestrian_intent_interactions and pedestrian_intent_proximity_interaction is not None:
                camera_feats = {k.replace('camera_features_', ''): v for k, v in current_features.items() if k.startswith('camera_features_')}
                misc = {'min_distance_m': current_features.get('camera_features_min_distance_m') or current_features.get('min_distance_m')}
                pi_out = pedestrian_intent_proximity_interaction(camera_feats, misc)
                if pi_out:
                    out.update(pi_out)
        except Exception:
            logger.exception('Failed to compute pedestrian intent interactions')

        return out


__all__ = ["FeatureExtractor"]
