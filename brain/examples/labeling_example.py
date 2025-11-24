"""Small end-to-end example demonstrating detection, labeling, balancing, and export.

Run as: python examples/labeling_example.py
"""
import logging
import shutil
from pathlib import Path

from config.config_loader import ConfigLoader
from data_collection.labeling import AccidentDetector, NearMissDetector, TemporalLabeler, ManualAnnotator
from utils import dataset_utils


def make_synthetic_windows():
    wins = []
    for i in range(30):
        w = {
            "window_id": f"win{i}",
            "trip_id": "demo_trip",
            "timestamp": f"2025-01-01T00:00:{i:02d}Z",
            "imu": [{"ax": 0.0, "ay": 0.0, "az": 9.81}] ,
            "camera": [],
            "vehicle_controls": [],
            "weather": [],
        }
        wins.append(w)
    # inject a near-miss: braking spike at t=10
    wins[10]["vehicle_controls"] = [{"brake_pressure": 0.9, "steering_angle": 0.0}]
    wins[10]["camera"] = [{"distance_m": 2.0, "in_path": True}]
    # inject an accident: high decel at t=25
    wins[25]["imu"] = [{"ax": -50.0, "ay": 0.0, "az": 9.81}]
    wins[25]["camera"] = [{"distance_m": 1.0, "in_path": True}]
    return wins


def main():
    logging.basicConfig(level=logging.INFO)
    cfg = ConfigLoader("config/default_config.yaml")
    wins = make_synthetic_windows()

    ad = AccidentDetector(cfg)
    nm = NearMissDetector(cfg)
    events = []
    for w in wins:
        a = ad.detect(w)
        if a:
            events.append({**a, "type": "accident", "subtype": a.get("type")})
        nm_d = nm.detect(w, accident_detected=bool(a))
        if nm_d:
            events.append({**nm_d, "type": "near_miss"})

    tl = TemporalLabeler(cfg)
    labeled = tl.label_windows(wins, events)
    n_acc = sum(1 for w in labeled if w.get("labels", {}).get("is_accident"))
    n_near = sum(1 for w in labeled if w.get("labels", {}).get("is_near_miss"))
    print("Labeled windows: accidents=", n_acc, "near_misses=", n_near)
    outdir = Path("./example_labels")
    if outdir.exists():
        shutil.rmtree(outdir)
    dataset_utils.export_labeled_dataset(labeled, str(outdir))
    print("Exported to", outdir)

    # demonstrate balancing and splitting
    balanced = dataset_utils.balance_dataset_smote(labeled)
    train, val, test = dataset_utils.temporal_train_test_split(balanced)
    print(f"Train/Val/Test sizes: {len(train)}/{len(val)}/{len(test)}")
    # round-trip load
    loaded = dataset_utils.load_labeled_dataset(str(outdir))
    print("Loaded back", len(loaded), "labels")
    # cleanup
    shutil.rmtree(outdir)


if __name__ == "__main__":
    main()
