"""
Pose enrichment helpers for CameraHandler: pedestrian intent scoring and cyclist lean detection.

These helpers are designed to be used by the existing `CameraHandler`'s
_perfer_frame/_infer_frame flow. They do not replace the pose detector; they
consume pose keypoints (normalized coordinates) and return added fields for
`pose_data`.

Functions exposed:
  - compute_pedestrian_intent(prev_kps, curr_kps, window_frames) -> float
  - compute_cyclist_lean_angle(curr_kps) -> float (degrees)

Expectations for keypoints: list of (x,y) normalized values according to
MediaPipe pose model ordering. For pedestrian intent we use mid-hip / mid-shoulder
as robust anchors.
"""

from typing import List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)


def _midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def compute_pedestrian_intent(prev_kps: Optional[List[Tuple[float, float]]],
                               curr_kps: List[Tuple[float, float]],
                               camera_velocity_window: int = 5) -> float:
    """Compute a heuristic pedestrian intent score in [0,1].

    Approach:
      - Use shoulder midpoint and hip midpoint as anchors.
      - Compute vertical (y) delta between prev and curr: decreasing y -> moving towards camera
      - Normalize delta magnitude into a score where:
           >0.7 = approaching; ~0.5 stationary; <0.3 receding

    prev_kps: keypoints from previous frame (same person) or None
    curr_kps: keypoints from current frame
    camera_velocity_window: window size used by caller to average deltas (present for API compatibility)
    """
    try:
        # required indices: left_shoulder=11, right_shoulder=12, left_hip=23, right_hip=24 (MediaPipe typical)
        # Some implementations use different indices; caller must map accordingly. We defensively check bounds.
        if not curr_kps or len(curr_kps) < 25:
            return 0.5

        ls = curr_kps[11]
        rs = curr_kps[12]
        lh = curr_kps[23]
        rh = curr_kps[24]
        shoulder_mid = _midpoint(ls, rs)
        hip_mid = _midpoint(lh, rh)

        curr_val = (shoulder_mid[1] + hip_mid[1]) / 2.0

        if prev_kps and len(prev_kps) >= 25:
            pls = prev_kps[11]
            prs = prev_kps[12]
            plh = prev_kps[23]
            prh = prev_kps[24]
            prev_sh_mid = _midpoint(pls, prs)
            prev_hip_mid = _midpoint(plh, prh)
            prev_val = (prev_sh_mid[1] + prev_hip_mid[1]) / 2.0
            # delta negative means y decreased -> moving up on image -> towards camera
            delta = prev_val - curr_val
        else:
            # no prev frame -> assume stationary neutral
            delta = 0.0

        # Scale delta to score. Empirical scaling: delta ~ 0.02 -> noticeable movement.
        # We'll use a sigmoid-like mapping
        score = 1.0 / (1.0 + math.exp(-20.0 * (delta - 0.01)))
        # Clamp
        score = max(0.0, min(1.0, score))
        return score
    except Exception:
        logger.exception("Failed to compute pedestrian intent")
        return 0.5


def compute_cyclist_lean_angle(curr_kps: List[Tuple[float, float]]) -> float:
    """Compute shoulder-to-hip tilt angle in degrees for cyclist maneuver detection.

    Approach:
      - Compute shoulder midpoint and hip midpoint as above.
      - Compute angle between vertical axis and vector (shoulder_mid -> hip_mid)
      - Return absolute angle in degrees. Typical maneuver threshold: >15 degrees.
    """
    try:
        if not curr_kps or len(curr_kps) < 25:
            return 0.0
        ls = curr_kps[11]
        rs = curr_kps[12]
        lh = curr_kps[23]
        rh = curr_kps[24]
        shoulder_mid = _midpoint(ls, rs)
        hip_mid = _midpoint(lh, rh)
        vx = hip_mid[0] - shoulder_mid[0]
        vy = hip_mid[1] - shoulder_mid[1]
        # angle from vertical (0 when vertical). atan2(vx, vy)
        if vy == 0:
            return 0.0
        angle_rad = math.atan2(vx, vy)
        angle_deg = abs(math.degrees(angle_rad))
        return angle_deg
    except Exception:
        logger.exception("Failed to compute cyclist lean")
        return 0.0
