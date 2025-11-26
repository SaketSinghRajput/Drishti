"""Utility functions for driver drowsiness detection.

Contains EAR/MAR computations, head-pose helpers, and IR preprocessing.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Dict


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return float(np.hypot(x2 - x1, y2 - y1))


def compute_ear(eye: List[Tuple[float, float]]) -> float:
    """Compute Eye Aspect Ratio for 6 points: p1..p6.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if len(eye) < 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye[:6]
    a = euclidean_distance(p2, p6)
    b = euclidean_distance(p3, p5)
    c = euclidean_distance(p1, p4)
    if c == 0:
        return 0.0
    return float((a + b) / (2.0 * c))


def compute_mar(mouth: List[Tuple[float, float]]) -> float:
    """Compute a simple MAR (mouth aspect ratio).

    Use vertical/horizontal distances to estimate opening.
    """
    if len(mouth) < 8:
        return 0.0
    # choose upper/lower and left/right reference points
    left = mouth[0]
    right = mouth[6]
    top = mouth[3]
    bottom = mouth[7]
    hor = euclidean_distance(left, right)
    ver = euclidean_distance(top, bottom)
    if hor == 0:
        return 0.0
    return float(ver / hor)


def preprocess_ir_frame(frame: np.ndarray, enable_hist_eq: bool = True, enable_clahe: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if enable_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    elif enable_hist_eq:
        gray = cv2.equalizeHist(gray)
    return gray


def normalize_landmarks(landmarks: List[Tuple[float, float]], width: int, height: int) -> List[Tuple[float, float]]:
    return [(x / width, y / height) for x, y in landmarks]


def get_3d_face_model() -> List[Tuple[float, float, float]]:
    # A minimal 3D face model (nose, eyes, mouth, chin) for solvePnP
    return [
        (0.0, 0.0, 0.0),  # nose tip
        (-30.0, 30.0, -30.0),  # left eye
        (30.0, 30.0, -30.0),  # right eye
        (-20.0, -30.0, -30.0),  # left mouth
        (20.0, -30.0, -30.0),  # right mouth
        (0.0, -60.0, -30.0),  # chin
    ]


# NOTE: keep only a single robust get_head_pose implementation below that
# relies on MediaPipe-style landmark indices. The older variant was removed
# to avoid duplicated behavior and subtle divergence.


def get_head_pose(landmarks_2d: List[Tuple[float, float]], frame_width: int, frame_height: int) -> Dict[str, float]:
    """Estimate head pose (pitch, yaw, roll) from 2D landmarks using solvePnP.

    Expects landmarks_2d to contain at least indices for nose, left eye,
    right eye, left mouth corner, right mouth corner. If insufficient
    landmarks are provided, returns zeros.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}

    model_pts = np.array(get_3d_face_model(), dtype=np.float64)
    # Prefer MediaPipe indices and include chin (199) to make 6 points
    indices = [1, 33, 263, 61, 291, 199]
    img_pts = []
    for idx in indices:
        if idx < len(landmarks_2d):
            x, y = landmarks_2d[idx]
            img_pts.append([float(x), float(y)])
        else:
            # not enough landmarks available
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}

    img_pts = np.array(img_pts, dtype=np.float64)
    focal_length = frame_width
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(model_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}

    # convert rotation vector to Euler angles
    R, _ = cv2.Rodrigues(rvec)
    sy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    # convert radians to degrees
    pitch = float(np.degrees(x))
    yaw = float(np.degrees(y))
    roll = float(np.degrees(z))
    return {"pitch": pitch, "yaw": yaw, "roll": roll, "confidence": 1.0}


def is_blink(ear: float, threshold: float, duration_ms: int) -> bool:
    return ear < threshold and duration_ms <= 500


def is_yawn(mar: float, threshold: float, duration_ms: int) -> bool:
    return mar > threshold and duration_ms >= 500


def smooth_value(current: float, previous: float, alpha: float = 0.6) -> float:
    return alpha * current + (1 - alpha) * previous


def estimate_ambient_light(frame: np.ndarray) -> float:
    """Estimate ambient light level as mean grayscale intensity (0-255).

    Defensive: returns 255 for None or empty frames to avoid altering thresholds.
    """
    try:
        if frame is None:
            return 255.0
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if gray.size == 0:
            return 255.0
        return float(gray.mean())
    except Exception:
        return 255.0


def adjust_thresholds_for_ir(ear_threshold: float, mar_threshold: float, ambient_light_level: float, ir_ear_offset: float = 0.05, ir_mar_offset: float = -0.05, ambient_light_low_threshold: float = 30.0) -> tuple[float, float]:
    """Adjust thresholds for IR/low-light conditions.

    If ambient light is below `ambient_light_low_threshold` we adjust ear/mar
    thresholds by the provided offsets. Returns (ear_threshold_adj, mar_threshold_adj).
    """
    try:
        if ambient_light_level < ambient_light_low_threshold:
            return max(0.01, ear_threshold + float(ir_ear_offset)), min(0.99, mar_threshold + float(ir_mar_offset))
        return ear_threshold, mar_threshold
    except Exception:
        return ear_threshold, mar_threshold


def detect_motion(frame_current: np.ndarray, frame_previous: np.ndarray, threshold: float = 5.0) -> bool:
    """Detect motion between two frames by mean absolute pixel difference.

    Returns True if mean absolute difference exceeds `threshold`.
    """
    try:
        if frame_current is None or frame_previous is None:
            return False
        import numpy as _np

        fc = frame_current
        fp = frame_previous
        if fc.shape != fp.shape:
            # resize previous to match current if possible
            try:
                import cv2

                fp = cv2.resize(fp, (fc.shape[1], fc.shape[0]))
            except Exception:
                return True
        diff = _np.abs(_np.asarray(fc, dtype=_np.float32) - _np.asarray(fp, dtype=_np.float32))
        mean_diff = float(diff.mean())
        return mean_diff > float(threshold)
    except Exception:
        return True
