import pytest
from utils import camera_utils


def test_estimate_distance_from_bbox_basic():
    bbox_small = [100, 100, 140, 200]  # taller box -> closer
    bbox_large = [100, 100, 120, 120]  # small height -> far
    d1 = camera_utils.estimate_distance_from_bbox(bbox_small, "person")
    d2 = camera_utils.estimate_distance_from_bbox(bbox_large, "person")
    assert 0.5 <= d1 <= 200.0
    assert 0.5 <= d2 <= 200.0
    assert d1 != d2


def test_estimate_distance_edge_cases():
    assert camera_utils.estimate_distance_from_bbox([], "person") == 999.0
    assert camera_utils.estimate_distance_from_bbox([0, 0, 0, 0], "unknown") >= 0.5


def test_estimate_distance_unknown_and_extremes():
    # Unknown class should still return a reasonable distance using defaults
    d = camera_utils.estimate_distance_from_bbox([10, 10, 100, 300], "unicorn")
    assert 0.5 <= d <= 200.0

    # Extremely large bbox (close object) -> small distance
    close = camera_utils.estimate_distance_from_bbox([0, 0, 640, 480], "car")
    assert close < 10.0

    # Tiny bbox (far away) -> large distance or the upper bound
    far = camera_utils.estimate_distance_from_bbox([300, 300, 305, 305], "car")
    assert far >= close


def test_estimate_relative_speed():
    s = camera_utils.estimate_relative_speed(10.0, 12.0, 1.0)
    assert s == pytest.approx(2.0)
    s2 = camera_utils.estimate_relative_speed(12.0, 10.0, 1.0)
    assert s2 == pytest.approx(-2.0)
    # zero time delta
    assert camera_utils.estimate_relative_speed(10.0, 9.0, 0.0) == 0.0


def test_compute_lane_offset_and_path():
    img_w = 640
    center = img_w // 2
    assert pytest.approx(0.0, abs=0.1) == camera_utils.compute_lane_offset(center, img_w)
    left = 0
    right = img_w
    assert camera_utils.compute_lane_offset(left, img_w) < 0
    assert camera_utils.compute_lane_offset(right, img_w) > 0
    # is_in_path
    bbox_center = [300, 300, 340, 420]
    assert camera_utils.is_in_path(bbox_center, 640, 480)
    bbox_edge = [0, 0, 10, 10]
    assert not camera_utils.is_in_path(bbox_edge, 640, 480)


def test_filter_and_nms():
    detections = [
        {"confidence": 0.9, "bbox_pixels": [10, 10, 50, 50]},
        {"confidence": 0.2, "bbox_pixels": [12, 12, 48, 48]},
        {"confidence": 0.8, "bbox_pixels": [200, 200, 260, 300]},
    ]
    filtered = camera_utils.filter_low_confidence_detections(detections, min_confidence=0.5)
    assert len(filtered) == 2
    nmsed = camera_utils.nms_detections(detections, iou_threshold=0.5)
    assert isinstance(nmsed, list)
    assert all("bbox_pixels" in d for d in nmsed)
