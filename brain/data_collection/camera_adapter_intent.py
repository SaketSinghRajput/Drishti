"""
Pedestrian intent and cyclist feature extraction for CameraAdapter.

Exposes helpers used by `CameraAdapter.analyze()` to compute aggregated
pedestrian intent scores and cyclist lean statistics across samples/windows.
"""
from typing import List, Dict, Any
import numpy as np


def _safe_get(obj: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = obj
    try:
        for p in path:
            cur = cur.get(p, {})
        return cur if cur != {} else default
    except Exception:
        return default


def _compute_pedestrian_intent_features(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = []
    for s in samples:
        objs = s.get('objects', [])
        for o in objs:
            pd = o.get('pose_data') or {}
            v = pd.get('pedestrian_intent_score')
            if v is None:
                continue
            try:
                scores.append(float(v))
            except Exception:
                continue
    if not scores:
        return {}
    arr = np.array(scores, dtype=np.float32)
    return {
        'pedestrian_intent_score_mean': float(np.mean(arr)),
        'pedestrian_intent_score_max': float(np.max(arr)),
        'pedestrians_walking_towards_count': int((arr > 0.7).sum()),
    }


def _compute_cyclist_features(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    angles = []
    maneuver_count = 0
    for s in samples:
        objs = s.get('objects', [])
        for o in objs:
            if o.get('class') not in ('bicycle', 'bike', 'cyclist'):
                continue
            pd = o.get('pose_data') or {}
            a = pd.get('cyclist_lean_angle')
            if a is None:
                continue
            try:
                ang = float(a)
            except Exception:
                continue
            angles.append(ang)
            if ang > 15.0:
                maneuver_count += 1
    if not angles:
        return {}
    arr = np.array(angles, dtype=np.float32)
    return {
        'avg_cyclist_lean_angle': float(np.mean(arr)),
        'max_cyclist_lean_angle': float(np.max(arr)),
        'cyclists_maneuvering_count': int(maneuver_count),
    }


# Exported interface
compute_pedestrian_intent_features = _compute_pedestrian_intent_features
compute_cyclist_features = _compute_cyclist_features
