"""Example demonstrating FeatureExtractor usage."""
import json

from config.config_loader import ConfigLoader
from feature_engineering import FeatureExtractor


def main():
    cfg = ConfigLoader('config/default_config.yaml')
    fe = FeatureExtractor(cfg)

    # build a synthetic window
    window = {
        "window_id": "example1",
        "timestamp": "2025-01-01T00:00:00Z",
        "trip_id": "trip_example",
        "gps": [{"speed_kph": 60.0}, {"speed_kph": 62.0}, {"speed_kph": 58.0}],
        "imu": [{"ax": 0.1, "ay": 0.0, "az": 9.7}, {"ax": -0.5, "ay": 0.1, "az": 9.6}],
        "vehicle_controls": [{"steering_angle": 0.0, "brake_pressure": 0.0}, {"steering_angle": 10.0, "brake_pressure": 0.5}],
        "driver_fatigue_features": {"drowsiness_score_mean": 0.4},
        "environment_features": {"visibility_mean_m": 50.0, "curvature_mean_m": 0.02, "road_surface": "wet"},
    }

    feats = fe.extract(window)
    print("Extracted features:\n")
    for k in sorted(feats.keys()):
        print(f"{k}: {feats[k]}")


if __name__ == "__main__":
    main()
