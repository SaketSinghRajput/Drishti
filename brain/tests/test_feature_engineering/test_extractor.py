import pytest

from config.config_loader import ConfigLoader
from feature_engineering import FeatureExtractor


def make_window_basic():
    return {
        "window_id": "w1",
        "timestamp": "2025-01-01T00:00:00Z",
        "trip_id": "t1",
        "gps": [{"speed_kph": 50.0}, {"speed_kph": 55.0}, {"speed_kph": 52.0}],
        "imu": [{"ax": -0.1, "ay": 0.0, "az": 9.7}, {"ax": -0.5, "ay": 0.0, "az": 9.6}],
        "vehicle_controls": [{"steering_angle": 0.0, "brake_pressure": 0.0}, {"steering_angle": 5.0, "brake_pressure": 0.2}],
        "driver_fatigue_features": {"drowsiness_score_mean": 0.2},
        "environment_features": {"visibility_mean_m": 100.0, "curvature_mean_m": 0.01, "road_surface": "dry"},
        "camera_features": {"min_distance_m": 5.0},
    }


def test_extract_basic():
    cfg = ConfigLoader('config/default_config.yaml')
    fe = FeatureExtractor(cfg)
    w = make_window_basic()
    out = fe.extract(w)
    assert out["window_id"] == "w1"
    assert "speed_mean" in out or "raw_speed_mean" in out
    # derived and interaction features present
    assert any(k.endswith("jerk_magnitude_mean") for k in out.keys())
    assert any("interaction" in k or "drowsiness_speed_interaction" in k for k in out.keys())


def test_disabled_feature_groups(monkeypatch):
    # create a config-like object that disables derived and interaction features
    class Cfg:
        def get(self, key, default=None):
            if key == 'feature_engineering':
                return {
                    'enable_raw_sensor_stats': True,
                    'enable_derived_features': False,
                    'enable_interaction_features': False,
                    'extract_gps_features': True,
                    'extract_imu_features': True,
                    'extract_vehicle_control_features': True,
                    'compute_jerk': False,
                    'compute_steering_rate': False,
                    'compute_speed_deltas': False,
                    'compute_lane_offset_trend': False,
                    'compute_drowsiness_interactions': False,
                    'compute_visibility_interactions': False,
                    'compute_proximity_interactions': False,
                    'compute_road_condition_interactions': False,
                    'flatten_nested_features': True,
                    'include_metadata': True,
                    'include_labels': True,
                }
            return default

    fe = FeatureExtractor(Cfg())
    w = make_window_basic()
    out = fe.extract(w)
    # derived features should be sentinel / missing
    assert out.get('derived_jerk_magnitude_mean') == -999.0 or out.get('jerk_magnitude_mean') == -999.0
    # interactions should be absent or zero
    assert out.get('interaction_drowsiness_speed_interaction', 0.0) == 0.0
    # metadata keys present
    assert 'window_id' in out and out['window_id'] == 'w1'


def test_extract_to_dataframe():
    cfg = ConfigLoader('config/default_config.yaml')
    fe = FeatureExtractor(cfg)
    windows = [make_window_basic(), make_window_basic()]
    df = fe.extract_to_dataframe(windows)
    # either pandas DataFrame or list returned
    if hasattr(df, "shape"):
        assert df.shape[0] == 2
    else:
        assert isinstance(df, list) and len(df) == 2
