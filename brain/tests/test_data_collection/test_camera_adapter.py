from config.config_loader import ConfigLoader
from data_collection.camera_adapter import CameraAdapter


def make_sample_frame(num_objs=1):
    objs = []
    for i in range(num_objs):
        objs.append({
            "class": "car",
            "confidence": 0.9,
            "bbox_pixels": [100, 200, 200, 400],
            "distance_m": 15.0,
            "rel_speed_mps": -1.2,
        })
    return {
        "timestamp": "2025-01-01T00:00:00Z",
        "objects": objs,
        "lane_center_offset_m": 0.1,
        "objects_in_path": num_objs,
        "frame_flagged_for_retention": False,
        "visibility_estimate_m": 1000.0,
    }


def test_camera_adapter_basic_analysis():
    cfg = ConfigLoader("config/default_config.yaml")
    ca = CameraAdapter(cfg)
    samples = [make_sample_frame(2) for _ in range(5)]
    feats = ca.analyze(samples)
    assert isinstance(feats, dict)
    assert feats["sample_count"] == 5
    assert feats["total_detections"] == 10


def test_camera_adapter_thresholds_affect_bins(tmp_path):
    # Write a temporary config that changes critical thresholds to [10,20,30]
    cfg_path = tmp_path / "tmp_config.yaml"
    cfg_path.write_text(
        """
sensors:
  camera:
    critical_distance_thresholds_m: [10, 20, 30]
    enable_aggregate_features: true
"""
    )
    cfg = ConfigLoader(str(cfg_path))
    ca = CameraAdapter(cfg)
    # one object at distance 15 should be counted in 20 and 30 bins but not 10
    sample = make_sample_frame(1)
    sample["objects"][0]["distance_m"] = 15.0
    feats = ca.analyze([sample])
    assert feats is not None
    # ensure bin keys exist for the configured thresholds
    assert "objects_within_10m" in feats
    assert "objects_within_20m" in feats
    assert feats["objects_within_10m"] == 0
    assert feats["objects_within_20m"] == 1
