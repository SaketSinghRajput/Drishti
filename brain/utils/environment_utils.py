"""Environment-specific helper utilities: risk scoring and categorization."""
from __future__ import annotations

from typing import Optional
from datetime import datetime
import math

from utils.time_utils import parse_timestamp


def _normalize(v: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0


def compute_weather_risk_score(
    precip_type: str,
    precip_intensity: float,
    visibility_m: float,
    road_surface: Optional[str] = None,
    weights: Optional[dict] = None,
) -> float:
    """Return a 0-1 risk score for weather conditions.

    Components (each normalized to 0-1) are combined using `weights` which is a
    mapping like {precip:0.3, visibility:0.4, surface:0.3}. If weights is None a
    sensible default is used. The precip component reflects presence of rain/snow
    and the intensity; visibility maps lower distances to higher risk; surface
    increases risk for 'wet' and 'icy'. The returned value is clamped to [0,1].
    """
    # defaults sum to 1.0
    w = dict(precip=0.3, visibility=0.4, surface=0.3)
    if isinstance(weights, dict):
        w.update(weights)

    pt = (precip_type or "none").lower()
    # precip component: none=0, rain=0.7, snow=1.0 (base), scaled by intensity
    base_precip = 0.0
    if pt == "rain":
        base_precip = 0.7
    elif pt == "snow":
        base_precip = 1.0
    try:
        intensity = max(0.0, float(precip_intensity))
        intensity_norm = _normalize(intensity / 25.0)
    except Exception:
        intensity_norm = 0.0
    precip_score = _normalize(base_precip * intensity_norm)

    # visibility component: lower visibility -> higher risk
    try:
        vis = float(visibility_m) if visibility_m is not None else 1000.0
    except Exception:
        vis = 1000.0
    if vis <= 200:
        visibility_score = 1.0
    elif vis <= 1000:
        visibility_score = 0.5
    else:
        visibility_score = 0.0

    # surface component
    surface_score = 1.0 if (road_surface in ("wet", "icy")) else 0.0

    # weighted combination (normalize by sum of weights to be tolerant)
    total_w = sum(float(v) for v in w.values()) if w else 1.0
    if total_w <= 0:
        total_w = 1.0
    score = (
        w.get("precip", 0.0) * precip_score
        + w.get("visibility", 0.0) * visibility_score
        + w.get("surface", 0.0) * surface_score
    )
    score = float(score / total_w)
    return _normalize(score)


def compute_road_geometry_risk_score(
    curvature_radius_m: float,
    upcoming_turn: Optional[str],
    intersection_flag: bool,
    slope_percent: float,
    speed_kph: float,
    weights: Optional[dict] = None,
    steep_slope_threshold_percent: float = 5.0,
) -> float:
    """Return a 0-1 risk score for road geometry.

    The function computes component scores for curvature (smaller radius ->
    higher risk), turn severity ('sharp' > left/right > none), intersection,
    slope (larger absolute slope > threshold), and speed. Components are
    combined using `weights` (defaults to a reasonable mapping). The
    `steep_slope_threshold_percent` replaces any hard-coded slope boundary.
    """
    # default weights
    w = dict(curvature=0.4, turn=0.2, intersection=0.2, slope=0.2)
    if isinstance(weights, dict):
        w.update(weights)

    # curvature score: map radius -> [0,1] (smaller -> larger)
    try:
        r = float(curvature_radius_m) if curvature_radius_m else float("inf")
        if r <= 20:
            curvature_score = 1.0
        elif r <= 50:
            curvature_score = 0.9
        elif r <= 200:
            curvature_score = 0.6
        elif r <= 500:
            curvature_score = 0.3
        else:
            curvature_score = 0.0
    except Exception:
        curvature_score = 0.0

    # turn score: sharp highest, left/right medium
    if upcoming_turn == "sharp":
        turn_score = 1.0
    elif upcoming_turn in ("left", "right"):
        turn_score = 0.5
    else:
        turn_score = 0.0

    intersection_score = 1.0 if bool(intersection_flag) else 0.0

    # slope score: relative to configured steep threshold
    try:
        s = abs(float(slope_percent)) if slope_percent is not None else 0.0
        if steep_slope_threshold_percent <= 0:
            slope_score = 0.0
        else:
            slope_score = _normalize(min(1.0, s / float(steep_slope_threshold_percent)))
    except Exception:
        slope_score = 0.0

    # speed score (simple mapping)
    try:
        sp = float(speed_kph) if speed_kph else 0.0
        if sp >= 120:
            speed_score = 1.0
        elif sp >= 80:
            speed_score = 0.8
        elif sp >= 50:
            speed_score = 0.4
        else:
            speed_score = 0.0
    except Exception:
        speed_score = 0.0

    total_w = sum(float(v) for v in w.values()) if w else 1.0
    if total_w <= 0:
        total_w = 1.0

    score = (
        w.get("curvature", 0.0) * curvature_score
        + w.get("turn", 0.0) * turn_score
        + w.get("intersection", 0.0) * intersection_score
        + w.get("slope", 0.0) * slope_score
    )
    score = float(score / total_w)
    # optionally consider speed as a modifier (blend with final score)
    score = _normalize((score + speed_score * 0.1))
    return score


def categorize_time_of_day(timestamp: str) -> str:
    """Return one of 'night','morning','afternoon','evening' based on hour in UTC.

    Expects ISO timestamp parseable by utils.time_utils.parse_timestamp.
    """
    try:
        dt = parse_timestamp(timestamp)
        hour = dt.hour
        if 0 <= hour < 6:
            return "night"
        if 6 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        return "evening"
    except Exception:
        return "unknown"


def infer_road_surface_from_weather(precip_type: str, ambient_temp_c: Optional[float] = None) -> str:
    pt = (precip_type or "none").lower()
    if pt == "none":
        return "dry"
    if pt == "rain":
        return "wet"
    if pt == "snow":
        try:
            if ambient_temp_c is not None and ambient_temp_c < 0:
                return "icy"
        except Exception:
            pass
        return "wet"
    return "dry"


def categorize_visibility(visibility_m: float, thresholds: dict) -> str:
    try:
        v = float(visibility_m)
    except Exception:
        return "high"
    low = float(thresholds.get("low", 200))
    medium = float(thresholds.get("medium", 1000))
    if v <= low:
        return "low"
    if v <= medium:
        return "medium"
    return "high"
