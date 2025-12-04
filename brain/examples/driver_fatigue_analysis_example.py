"""Example usage of DriverFatigueAdapter.

Generates synthetic driver_fatigue samples and runs the adapter to show
computed features.
"""
import json
import sys
import pathlib

# Ensure repository root (two levels up) is on sys.path so top-level
# packages like `config` and `data_collection` are importable when
# running this example from `brain/`.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.config_loader import ConfigLoader
from data_collection.driver_fatigue_adapter import DriverFatigueAdapter
from utils.time_utils import get_utc_timestamp


def make_samples():
    samples = []
    # first half low drowsiness
    for _ in range(5):
        samples.append({
            "drowsiness_score": 0.2,
            "blink_rate": 15,
            "perclos": 0.1,
            "yawning_count": 0,
            "micro_sleep_flag": False,
            "head_pose": {"pitch": 0, "yaw": 0, "roll": 0, "confidence": 1.0},
            "timestamp": get_utc_timestamp(),
        })
    # second half spike
    for _ in range(5):
        samples.append({
            "drowsiness_score": 0.8,
            "blink_rate": 5,
            "perclos": 0.6,
            "yawning_count": 1,
            "micro_sleep_flag": False,
            "head_pose": {"pitch": 1, "yaw": -1, "roll": 0, "confidence": 0.9},
            "timestamp": get_utc_timestamp(),
        })
    return samples


def main():
    cfg = ConfigLoader()
    ad = DriverFatigueAdapter(cfg)
    samples = make_samples()
    features = ad.analyze(samples)
    print(json.dumps(features, indent=2))


if __name__ == "__main__":
    main()
