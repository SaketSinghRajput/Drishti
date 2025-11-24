"""Camera helper utilities for distance, speed, lane and path estimation.

Lightweight helpers used by camera handlers and model backends. Types are
annotated for better static analysis while remaining runtime-friendly.
"""
from typing import List, Dict, Any, Optional, Tuple
import math

DEFAULT_SIZES: Dict[str, float] = {
    "car": 4.5,
    "person": 1.7,
    "bicycle": 1.8,
    "motorcycle": 2.0,
    "bus": 12.0,
    "truck": 8.0,
}


def estimate_distance_from_bbox(
    bbox_pixels: List[int],
    object_class: str,
    focal_length_px: float = 500.0,
    known_sizes: Optional[Dict[str, float]] = None,
) -> float:
    """Estimate object distance (meters) from bbox using a pinhole camera model.

    Parameters
    - bbox_pixels: [x1,y1,x2,y2]
    - object_class: class name to lookup approximate real-world height
    - focal_length_px: focal length in pixels (approx)

    Returns estimated distance in meters, clamped to [0.5, 200].
    """
    if not bbox_pixels or len(bbox_pixels) < 4:
        return 999.0
    x1, y1, x2, y2 = bbox_pixels
    h = max(1, abs(int(y2) - int(y1)))
    sizes = known_sizes if isinstance(known_sizes, dict) else DEFAULT_SIZES
    real_h = sizes.get(object_class, sizes.get("person", 1.7))
    try:
        dist = (real_h * float(focal_length_px)) / float(h)
    except Exception:
        return 999.0
    dist = max(0.5, min(dist, 200.0))
    return float(dist)


def estimate_relative_speed(
    current_distance: float,
    previous_distance: float,
    time_delta_s: float,
    ego_speed_mps: float = 0.0,
) -> float:
    """Estimate relative speed (m/s). Negative = approaching, Positive = receding.

    Returns 0.0 for invalid inputs or zero time delta.
    """
    if time_delta_s <= 0 or current_distance is None or previous_distance is None:
        return 0.0
    rel = (previous_distance - current_distance) / float(time_delta_s)
    try:
        adj = rel - ego_speed_mps
    except Exception:
        adj = rel
    return float(adj)


def compute_lane_offset(bbox_center_x: int, image_width: int, lane_width_m: float = 3.5) -> float:
    """Estimate lateral offset from lane center (meters).

    Negative = left, Positive = right.
    """
    if image_width <= 0:
        return 0.0
    offset_ratio = (float(bbox_center_x) - float(image_width) / 2.0) / float(image_width)
    return float(offset_ratio * lane_width_m)


def is_in_path(
    bbox_pixels: List[int],
    image_width: int,
    image_height: int,
    path_center_tolerance: float = 0.3,
) -> bool:
    """Return True if the bbox center falls within the center path region and is low enough vertically.

    The vertical check biases toward objects in the lower half of the image as more relevant.
    """
    if not bbox_pixels or len(bbox_pixels) < 4:
        return False
    x1, y1, x2, y2 = bbox_pixels
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if image_width <= 0 or image_height <= 0:
        return False
    center_norm = (cx / float(image_width))
    tol = path_center_tolerance
    in_center = abs(center_norm - 0.5) <= tol * 0.5
    lower_half = cy >= (image_height * 0.4)
    return bool(in_center and lower_half)


def filter_low_confidence_detections(detections: List[Dict[str, Any]], min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """Filter detections by confidence/score key.
    Supports keys 'confidence' and 'score'."""
    if not detections:
        return []
    out: List[Dict[str, Any]] = []
    for d in detections:
        c = None
        if isinstance(d, dict):
            c = d.get("confidence") if d.get("confidence") is not None else d.get("score")
        try:
            if c is None:
                continue
            if float(c) >= float(min_confidence):
                out.append(d)
        except Exception:
            continue
    return out


def _iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, float(xB - xA))
    interH = max(0.0, float(yB - yA))
    interArea = interW * interH
    boxAArea = max(0.0, float(boxA[2] - boxA[0])) * max(0.0, float(boxA[3] - boxA[1]))
    boxBArea = max(0.0, float(boxB[2] - boxB[0])) * max(0.0, float(boxB[3] - boxB[1]))
    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0.0:
        return 0.0
    return interArea / denom


def nms_detections(detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Simple NMS implementation. Each detection must have 'bbox_pixels' and 'confidence'/'score'."""
    if not detections:
        return []
    dets: List[Dict[str, Any]] = []
    for d in detections:
        if not isinstance(d, dict):
            continue
        bbox = d.get("bbox_pixels")
        if not bbox or len(bbox) < 4:
            continue
        try:
            conf = d.get("confidence") if d.get("confidence") is not None else d.get("score", 0.0)
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        dets.append({"d": d, "bbox": tuple(float(x) for x in bbox[:4]), "conf": conf_f})
    dets.sort(key=lambda x: x["conf"], reverse=True)
    keep: List[Dict[str, Any]] = []
    while dets:
        current = dets.pop(0)
        keep.append(current["d"])
        rem: List[Dict[str, Any]] = []
        for other in dets:
            iouv = _iou(current["bbox"], other["bbox"])
            if iouv <= iou_threshold:
                rem.append(other)
        dets = rem
    return keep
