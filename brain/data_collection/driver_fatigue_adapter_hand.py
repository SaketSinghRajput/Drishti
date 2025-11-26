"""
Grip feature extraction for DriverFatigueAdapter.

This module provides `_compute_grip_features(samples)` which extracts rolling
statistics from `grip_confidence` present in per-frame samples and detects
sudden grip loss events.

It is intended to be invoked from the adapter's `analyze()` and merged into the
final feature dict returned to the feature pipeline.
"""

from typing import List, Dict, Any
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _extract_grip_series(samples: List[Dict[str, Any]]) -> List[float]:
    vals = []
    for s in samples:
        v = s.get('grip_confidence')
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    return vals


def _compute_grip_features(samples: List[Dict[str, Any]], sample_rate_hz: float = 10.0) -> Dict[str, Any]:
    """Compute grip statistics and sudden loss detection.

    Returns:
      dict with keys: grip_confidence_mean, grip_confidence_std, grip_confidence_min,
                     sudden_grip_loss_flag, no_grip_duration_s
    """
    vals = _extract_grip_series(samples)
    if not vals:
        return {}

    arr = np.array(vals, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    mn = float(np.min(arr))

    # sudden grip loss: any drop >0.3 within 2 seconds
    window = int(max(1, int(2.0 * sample_rate_hz)))
    sudden = False
    for i in range(len(arr) - window):
        if arr[i] - arr[i + window] > 0.3:
            sudden = True
            break

    # no_grip_duration_s: consecutive time where confidence <0.3
    threshold = 0.3
    max_dur = 0.0
    cur = 0
    for v in arr:
        if v < threshold:
            cur += 1
        else:
            if cur > 0:
                dur = cur / sample_rate_hz
                if dur > max_dur:
                    max_dur = dur
            cur = 0
    if cur > 0:
        dur = cur / sample_rate_hz
        if dur > max_dur:
            max_dur = dur

    return {
        'grip_confidence_mean': mean,
        'grip_confidence_std': std,
        'grip_confidence_min': mn,
        'sudden_grip_loss_flag': bool(sudden),
        'no_grip_duration_s': float(max_dur),
    }


# Export name used by adapter
compute_grip_features = _compute_grip_features
