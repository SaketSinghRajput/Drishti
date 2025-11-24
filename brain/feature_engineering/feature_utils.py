"""Feature engineering helper utilities.

Provides derivative, jerk, magnitude, trend, and small aggregation helpers.
"""
from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except Exception:
    np = None

from utils.time_utils import parse_timestamp

logger = logging.getLogger(__name__)


def extract_time_series_values(samples: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    if not samples:
        return out
    for s in samples:
        try:
            v = s.get(key)
            if v is None:
                continue
            out.append(float(v))
        except Exception:
            continue
    return out


def _agg_stats(values: List[float], prefix: str = "", sentinel: float = -999.0) -> Dict[str, float]:
    if not values:
        return {
            prefix + "mean": sentinel,
            prefix + "std": sentinel,
            prefix + "min": sentinel,
            prefix + "max": sentinel,
        }
    try:
        if np is not None:
            arr = np.array(values, dtype=float)
            return {
                prefix + "mean": float(arr.mean()),
                prefix + "std": float(arr.std()),
                prefix + "min": float(arr.min()),
                prefix + "max": float(arr.max()),
            }
        else:
            n = len(values)
            mean = sum(values) / n
            var = sum((x - mean) ** 2 for x in values) / n
            return {
                prefix + "mean": float(mean),
                prefix + "std": float(math.sqrt(var)),
                prefix + "min": float(min(values)),
                prefix + "max": float(max(values)),
            }
    except Exception:
        logger.exception("_agg_stats failed")
        return {
            prefix + "mean": sentinel,
            prefix + "std": sentinel,
            prefix + "min": sentinel,
            prefix + "max": sentinel,
        }


def compute_derivative(samples: List[Dict[str, Any]], value_key: str, time_key: str = "timestamp") -> List[float]:
    """Compute derivative values (delta_value / delta_time) for consecutive samples."""
    if not samples or len(samples) < 2:
        return []
    # sort by timestamp if available
    try:
        sorted_samples = sorted(samples, key=lambda s: parse_timestamp(s.get(time_key)))
    except Exception:
        sorted_samples = list(samples)
    out: List[float] = []
    prev_t = None
    prev_v = None
    for s in sorted_samples:
        try:
            t = parse_timestamp(s.get(time_key)).timestamp()
            v = float(s.get(value_key))
            if prev_t is not None and (t - prev_t) > 0:
                out.append((v - prev_v) / (t - prev_t))
            prev_t = t
            prev_v = v
        except Exception:
            continue
    return out


def compute_rate_of_change(samples: List[Dict[str, Any]], value_key: str) -> Dict[str, float]:
    vals = extract_time_series_values(samples, value_key)
    if not vals or len(vals) < 2:
        return {"mean": 0.0, "max": 0.0}
    rates: List[float] = []
    for i in range(1, len(vals)):
        rates.append(abs(vals[i] - vals[i - 1]))
    return {"mean": float(sum(rates) / len(rates)), "max": float(max(rates))}


def compute_jerk_from_acceleration(accel_samples: List[Dict[str, Any]], accel_keys: List[str] = ["ax", "ay", "az"]) -> Dict[str, float]:
    # Build per-axis lists for derivative
    if not accel_samples or len(accel_samples) < 2:
        return {"jerk_x_mean": 0.0, "jerk_y_mean": 0.0, "jerk_z_mean": 0.0, "jerk_magnitude_mean": 0.0, "jerk_magnitude_max": 0.0}
    axes_vals = {k: extract_time_series_values(accel_samples, k) for k in accel_keys}
    # compute derivative per axis using simple diffs (assume uniform sample spacing if no timestamps)
    jerks = {k: [] for k in accel_keys}
    n = max(len(v) for v in axes_vals.values())
    # safe approach: compute pairwise diffs where possible
    for k in accel_keys:
        vals = axes_vals.get(k, [])
        for i in range(1, len(vals)):
            try:
                jerks[k].append((vals[i] - vals[i - 1]))
            except Exception:
                continue
    mag_list: List[float] = []
    for i in range(max(1, max((len(j) for j in jerks.values()), default=1))):
        try:
            x = jerks.get("ax", [0])[i] if i < len(jerks.get("ax", [])) else 0.0
            y = jerks.get("ay", [0])[i] if i < len(jerks.get("ay", [])) else 0.0
            z = jerks.get("az", [0])[i] if i < len(jerks.get("az", [])) else 0.0
            mag = math.sqrt(x * x + y * y + z * z)
            mag_list.append(abs(mag))
        except Exception:
            continue
    def _mean(lst: List[float]) -> float:
        return float(sum(lst) / len(lst)) if lst else 0.0

    return {
        "jerk_x_mean": _mean(jerks.get("ax", [])),
        "jerk_y_mean": _mean(jerks.get("ay", [])),
        "jerk_z_mean": _mean(jerks.get("az", [])),
        "jerk_magnitude_mean": _mean(mag_list),
        "jerk_magnitude_max": float(max(mag_list) if mag_list else 0.0),
    }


def compute_magnitude(samples: List[Dict[str, Any]], component_keys: List[str]) -> List[float]:
    out: List[float] = []
    for s in samples:
        try:
            vals = [float(s.get(k, 0.0)) for k in component_keys]
            out.append(math.sqrt(sum(v * v for v in vals)))
        except Exception:
            continue
    return out


def compute_trend_slope(samples: List[Dict[str, Any]], value_key: str, time_key: str = "timestamp") -> float:
    # samples may be list of dicts or pre-extracted lists
    vals = []
    times = []
    for s in samples:
        try:
            t = parse_timestamp(s.get(time_key)).timestamp()
            v = float(s.get(value_key))
            times.append(t)
            vals.append(v)
        except Exception:
            continue
    if len(times) < 2:
        return 0.0
    try:
        if np is not None:
            coeffs = np.polyfit(times, vals, 1)
            return float(coeffs[0])
        # simple least squares
        n = len(times)
        xm = sum(times) / n
        ym = sum(vals) / n
        num = sum((times[i] - xm) * (vals[i] - ym) for i in range(n))
        den = sum((times[i] - xm) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        return float(num / den)
    except Exception:
        logger.exception("compute_trend_slope failed")
        return 0.0


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except Exception:
        return default


def compute_delta_features(values: List[float], key_name: str) -> Dict[str, float]:
    if not values:
        return {f"{key_name}_delta_total": 0.0, f"{key_name}_delta_max": 0.0, f"{key_name}_delta_mean": 0.0}
    try:
        total = float(values[-1] - values[0]) if len(values) >= 2 else 0.0
        diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))] if len(values) >= 2 else [0.0]
        return {f"{key_name}_delta_total": total, f"{key_name}_delta_max": float(max(diffs)), f"{key_name}_delta_mean": float(sum(diffs) / len(diffs))}
    except Exception:
        logger.exception("compute_delta_features failed")
        return {f"{key_name}_delta_total": 0.0, f"{key_name}_delta_max": 0.0, f"{key_name}_delta_mean": 0.0}


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=prefix + k + "_"))
        else:
            out[prefix + k] = v
    return out
