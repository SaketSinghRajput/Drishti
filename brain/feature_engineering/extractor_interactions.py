"""
Additional interaction feature computations for grip & pedestrian intent.

These functions can be called by the main `extractor` to add interaction terms
when enabled by config.
"""
from typing import Dict, Any


def grip_loss_speed_interaction(driver_feats: Dict[str, Any], camera_feats: Dict[str, Any]) -> Dict[str, float]:
    """(1 - grip_confidence_mean) * speed_mean -> higher when no grip at high speed."""
    grip = driver_feats.get('grip_confidence_mean')
    speed = camera_feats.get('speed_mean') or camera_feats.get('vehicle_speed') or 0.0
    if grip is None or speed is None:
        return {}
    try:
        val = (1.0 - float(grip)) * float(speed)
        return {'interaction_grip_loss_speed_interaction': float(val)}
    except Exception:
        return {}


def pedestrian_intent_proximity_interaction(camera_feats: Dict[str, Any], misc: Dict[str, Any]) -> Dict[str, float]:
    """pedestrian_intent_score_max * (1 / min_distance_m)."""
    intent = camera_feats.get('pedestrian_intent_score_max')
    min_dist = misc.get('min_distance_m')
    if intent is None or not min_dist:
        return {}
    try:
        val = float(intent) * (1.0 / float(min_dist))
        return {'interaction_pedestrian_intent_proximity_interaction': float(val)}
    except Exception:
        return {}
