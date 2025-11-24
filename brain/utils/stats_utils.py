"""Statistical helper functions with small, safe fallbacks.

These functions prefer numpy when available for speed and precision but provide
pure-Python fallbacks so the project can run in environments without numpy.
"""
from __future__ import annotations

import math
from typing import List

try:
    import numpy as _np
except Exception:
    _np = None


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    if _np is not None:
        return float(_np.mean(_np.array(values, dtype=float)))
    return float(sum(values) / len(values))


def safe_std(values: List[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    if _np is not None:
        return float(_np.std(_np.array(values, dtype=float), ddof=0))
    # population std
    m = safe_mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def safe_min(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(min(values))


def safe_max(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(max(values))


def compute_percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    if _np is not None:
        return float(_np.percentile(_np.array(values, dtype=float), percentile))
    # simple pure-Python percentile (nearest-rank)
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return float(d0 + d1)


def linear_trend(values: List[float], slope_threshold: float = 0.01) -> str:
    """Return 'increasing', 'decreasing', or 'stable' based on linear slope.

    Uses numpy.polyfit when available, otherwise computes slope via least-squares.
    """
    if not values or len(values) < 2:
        return "stable"

    xs = list(range(len(values)))
    ys = [float(v) for v in values]

    if _np is not None:
        try:
            slope = float(_np.polyfit(xs, ys, 1)[0])
        except Exception:
            slope = 0.0
    else:
        # compute slope = covariance(x,y)/variance(x)
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        slope = float(num / den) if den != 0 else 0.0

    if abs(slope) < slope_threshold:
        return "stable"
    return "increasing" if slope > 0 else "decreasing"


def rolling_window_stats(values: List[float], window_size: int) -> List[dict]:
    if not values or window_size <= 0:
        return []
    out = []
    for i in range(len(values) - window_size + 1):
        w = values[i : i + window_size]
        out.append({"mean": safe_mean(w), "std": safe_std(w)})
    return out
