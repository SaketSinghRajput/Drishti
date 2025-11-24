"""EnvironmentAdapter: aggregates weather and road samples into window-level features.

Follows pattern of other adapters: __init__(config) and analyze(samples) -> features dict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from config.config_loader import ConfigLoader
from utils import stats_utils
from utils.environment_utils import (
    compute_weather_risk_score,
    compute_road_geometry_risk_score,
    categorize_time_of_day,
    infer_road_surface_from_weather,
)

logger = logging.getLogger(__name__)


class EnvironmentAdapter:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.cfg = config.get("environment_features", {})
        self.enable_weather = bool(self.cfg.get("enable_weather_risk", True))
        self.enable_road = bool(self.cfg.get("enable_road_risk", True))
        self.enable_interaction = bool(self.cfg.get("enable_interaction_features", True))
        self.enable_time_of_day = bool(self.cfg.get("enable_time_of_day", True))
        # thresholds
        self.low_visibility_threshold = float(self.cfg.get("low_visibility_threshold_m", 500.0))
        self.high_risk_curvature_threshold = float(self.cfg.get("high_risk_curvature_threshold_m", 200.0))
        # risk weights & thresholds (exposed via config.environment_features)
        self.weather_risk_weights = dict(self.cfg.get("weather_risk_weights", {}))
        self.road_risk_weights = dict(self.cfg.get("road_risk_weights", {}))
        self.steep_slope_threshold_percent = float(self.cfg.get("steep_slope_threshold_percent", 5.0))

    def analyze(self, weather_samples: List[Dict[str, Any]] = None, road_samples: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            weather_samples = weather_samples or []
            road_samples = road_samples or []
            if not weather_samples and not road_samples:
                return None

            # Weather features
            precip_types = [s.get("precip_type", "none") for s in weather_samples]
            precip_intensities = [float(s.get("precip_intensity_mm_hr", 0.0)) for s in weather_samples]
            visibilities = [float(s.get("visibility_m", 0.0)) for s in weather_samples]
            ambient_modes = [s.get("ambient_light_level") for s in weather_samples if s.get("ambient_light_level") is not None]
            road_surfaces = [s.get("road_surface") for s in weather_samples if s.get("road_surface") is not None]

            weather_risk = 0.0
            if self.enable_weather and weather_samples:
                mode_precip = max(set(precip_types), key=precip_types.count) if precip_types else "none"
                avg_intensity = float(stats_utils.safe_mean(precip_intensities)) if precip_intensities else 0.0
                max_intensity = float(stats_utils.safe_max(precip_intensities)) if precip_intensities else 0.0
                vis_mean = float(stats_utils.safe_mean(visibilities)) if visibilities else 0.0
                vis_min = float(stats_utils.safe_min(visibilities)) if visibilities else 0.0
                vis_std = float(stats_utils.safe_std(visibilities)) if visibilities else 0.0
                ambient = None
                if ambient_modes:
                    ambient = max(set(ambient_modes), key=ambient_modes.count)
                surface = None
                if road_surfaces:
                    surface = max(set(road_surfaces), key=road_surfaces.count)
                else:
                    # If weather samples exist but no road_surface provided by sensors,
                    # infer a likely surface from the dominant precipitation type and
                    # optional ambient temperature if available.
                    if weather_samples:
                        # try to find an ambient temperature field if present
                        ambient_temps = [s.get("ambient_temp_c") for s in weather_samples if s.get("ambient_temp_c") is not None]
                        ambient_temp = None
                        if ambient_temps:
                            try:
                                ambient_temp = float(ambient_temps[0])
                            except Exception:
                                ambient_temp = None
                        surface = infer_road_surface_from_weather(mode_precip, ambient_temp)
                weather_risk = compute_weather_risk_score(mode_precip, avg_intensity, vis_mean, surface, weights=self.weather_risk_weights)
            else:
                mode_precip = "none"
                avg_intensity = 0.0
                max_intensity = 0.0
                vis_mean = vis_min = vis_std = 0.0
                ambient = None
                surface = None

            # Road features
            curvature_vals = [float(s.get("curvature_radius_m", 0.0)) for s in road_samples]
            speed_limits = [float(s.get("speed_limit_kph", 0.0)) for s in road_samples]
            turns = [s.get("upcoming_turn", "none") for s in road_samples]
            intersections = [bool(s.get("intersection_flag", False)) for s in road_samples]
            slopes = [float(s.get("slope_percent", 0.0)) for s in road_samples]

            road_risk = 0.0
            if self.enable_road and road_samples:
                cur_mean = float(stats_utils.safe_mean(curvature_vals)) if curvature_vals else 0.0
                cur_min = float(stats_utils.safe_min(curvature_vals)) if curvature_vals else 0.0
                cur_std = float(stats_utils.safe_std(curvature_vals)) if curvature_vals else 0.0
                speed_mode = int(stats_utils.safe_mean(speed_limits)) if speed_limits else 0
                turn_freq = float(sum(1 for t in turns if t and t != "none") / max(1, len(turns)))
                intersection_ratio = float(sum(1 for i in intersections if i) / max(1, len(intersections)))
                slope_mean = float(stats_utils.safe_mean(slopes)) if slopes else 0.0
                slope_max_abs = float(max((abs(s) for s in slopes), default=0.0))
                # compute road risk (higher when curvature small => sharper)
                # use inverse of radius for risk
                # aggregate upcoming_turn: choose the most frequent non-'none' turn if available
                non_none_turns = [t for t in turns if t and t != "none"]
                agg_turn = None
                if non_none_turns:
                    agg_turn = max(set(non_none_turns), key=non_none_turns.count)
                road_risk = compute_road_geometry_risk_score(
                    cur_mean,
                    agg_turn,
                    bool(intersection_ratio),
                    slope_max_abs,
                    speed_mode,
                    weights=self.road_risk_weights,
                    steep_slope_threshold_percent=self.steep_slope_threshold_percent,
                )
            else:
                cur_mean = cur_min = cur_std = 0.0
                speed_mode = 0
                turn_freq = 0.0
                intersection_ratio = 0.0
                slope_mean = slope_max_abs = 0.0

            # time of day
            tod = None
            if self.enable_time_of_day:
                # pick a timestamp from samples, prefer weather then road
                ts = None
                if weather_samples:
                    ts = weather_samples[0].get("timestamp")
                elif road_samples:
                    ts = road_samples[0].get("timestamp")
                if ts:
                    tod = categorize_time_of_day(ts)

            # interaction features
            wet_road_flag = (surface in ("wet", "icy")) if surface else False
            high_curvature_flag = (cur_mean > 0 and cur_mean < self.high_risk_curvature_threshold)
            combined_risk = float(min(1.0, weather_risk * road_risk)) if (weather_risk and road_risk) else max(weather_risk, road_risk)

            features: Dict[str, Any] = {
                "weather_risk_score": float(weather_risk),
                "road_geometry_risk_score": float(road_risk),
                "combined_risk_score": float(combined_risk),
                "time_of_day": tod,
                "precip_type_mode": mode_precip,
                "avg_precip_intensity_mm_hr": float(avg_intensity),
                "max_precip_intensity_mm_hr": float(max_intensity),
                "visibility_mean_m": float(vis_mean),
                "visibility_min_m": float(vis_min),
                "visibility_std": float(vis_std),
                "visibility_category": ("low" if vis_mean <= self.low_visibility_threshold else ("medium" if vis_mean <= self.low_visibility_threshold * 4 else "high")),
                "ambient_light_mode": ambient,
                "road_surface_mode": surface,
                "wet_road_flag": bool(wet_road_flag),
                "curvature_mean_m": float(cur_mean),
                "curvature_min_m": float(cur_min),
                "curvature_std": float(cur_std),
                "high_curvature_flag": bool(high_curvature_flag),
                "speed_limit_mode_kph": int(speed_mode),
                "turn_frequency": float(turn_freq),
                "intersection_ratio": float(intersection_ratio),
                "slope_mean_percent": float(slope_mean),
                "slope_max_abs_percent": float(slope_max_abs),
                "weather_sample_count": int(len(weather_samples)),
                "road_sample_count": int(len(road_samples)),
            }

            return features
        except Exception as e:
            logger.exception("EnvironmentAdapter analyze failed: %s", e)
            return None
