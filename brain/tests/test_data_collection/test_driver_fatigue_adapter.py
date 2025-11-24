from config.config_loader import ConfigLoader
from data_collection.driver_fatigue_adapter import DriverFatigueAdapter
from utils.time_utils import get_utc_timestamp, timestamp_to_epoch
import time
import pytest
from math import isclose


def make_sample(drowsiness, blink_rate=10, perclos=0.1, yawning=0, micro=False, ts=None):
    return {
        "drowsiness_score": drowsiness,
        "blink_rate": blink_rate,
        "perclos": perclos,
        "yawning_count": yawning,
        "micro_sleep_flag": micro,
        "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 1.0},
        "timestamp": ts or get_utc_timestamp(),
    }


def test_adapter_rolling_and_counts(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("sensors:\n  driver_fatigue:\n    sudden_drowsiness_threshold: 0.3\n")
    cfg = ConfigLoader(str(p))
    ad = DriverFatigueAdapter(cfg)

    samples = [make_sample(v) for v in [0.1, 0.2, 0.3, 0.4, 0.5]]
    feats = ad.analyze(samples)
    assert isclose(feats["drowsiness_score_mean"], 0.3, rel_tol=1e-3)
    assert feats["drowsiness_score_min"] == 0.1
    assert feats["drowsiness_score_max"] == 0.5
    assert feats["total_yawns"] == 0


def test_sudden_drowsiness_detection_positive(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("sensors:\n  driver_fatigue:\n    sudden_drowsiness_threshold: 0.3\n    sudden_change_window_s: 3.0\n")
    cfg = ConfigLoader(str(p))
    ad = DriverFatigueAdapter(cfg)

    from datetime import datetime, timezone, timedelta
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(seconds=2)
    s1 = make_sample(0.2, ts=t0.isoformat(timespec="milliseconds").replace("+00:00", "Z"))
    # simulate later sample with big jump within sudden change window
    s2 = make_sample(0.8, ts=t1.isoformat(timespec="milliseconds").replace("+00:00", "Z"))
    feats = ad.analyze([s1, s2])
    assert feats["sudden_drowsiness_increase"] is True
    assert feats["max_drowsiness_delta"] >= 0.5


def test_trend_and_empty_and_single(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("sensors:\n  driver_fatigue:\n    trend_slope_threshold: 0.01\n")
    cfg = ConfigLoader(str(p))
    ad = DriverFatigueAdapter(cfg)

    # increasing trend
    inc = [make_sample(v) for v in [0.1, 0.3, 0.5, 0.7]]
    f1 = ad.analyze(inc)
    assert f1["drowsiness_trend"] == "increasing"

    # single sample
    single = [make_sample(0.25)]
    f2 = ad.analyze(single)
    assert f2["drowsiness_score_mean"] == 0.25

    # empty
    assert ad.analyze([]) is None


def test_blink_and_perclos_statistics(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("sensors:\n  driver_fatigue:\n    sudden_drowsiness_threshold: 0.3\n")
    cfg = ConfigLoader(str(p))
    ad = DriverFatigueAdapter(cfg)

    samples = [make_sample(0.1, blink_rate=5, perclos=0.05), make_sample(0.2, blink_rate=7, perclos=0.1), make_sample(0.3, blink_rate=9, perclos=0.15)]
    feats = ad.analyze(samples)
    assert feats["blink_rate_mean"] == 7.0
    assert feats["perclos_min"] == 0.05
    assert feats["perclos_max"] == 0.15


def test_head_pose_aggregation_and_event_counts(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("sensors:\n  driver_fatigue:\n    sudden_drowsiness_threshold: 0.3\n")
    cfg = ConfigLoader(str(p))
    ad = DriverFatigueAdapter(cfg)

    from datetime import datetime, timezone
    t = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    samples = [
        {"drowsiness_score": 0.1, "head_pose": {"pitch": 1.0, "yaw": 2.0, "roll": 3.0}, "yawning_count": 1, "micro_sleep_flag": True, "timestamp": t},
        {"drowsiness_score": 0.2, "head_pose": {"pitch": 2.0, "yaw": 3.0, "roll": 4.0}, "yawning_count": 2, "micro_sleep_flag": False, "timestamp": t},
    ]

    feats = ad.analyze(samples)
    # head pose means
    assert feats["head_pose_mean_pitch"] == 1.5
    assert feats["head_pose_mean_yaw"] == 2.5
    assert feats["head_pose_mean_roll"] == 3.5
    # event counts
    assert feats["total_yawns"] == 3
    assert feats["micro_sleep_events"] == 1


def test_missing_fields_and_custom_thresholds(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("sensors:\n  driver_fatigue:\n    sudden_drowsiness_threshold: 0.2\n    sudden_change_window_s: 5.0\n    trend_slope_threshold: 0.005\n")
    cfg = ConfigLoader(str(p))
    ad = DriverFatigueAdapter(cfg)

    # samples with some missing fields (no head_pose, no yawning_count)
    from datetime import datetime, timezone, timedelta
    t0 = datetime.now(timezone.utc)
    s1 = {"drowsiness_score": 0.1, "timestamp": t0.isoformat().replace("+00:00", "Z")}
    s2 = {"drowsiness_score": 0.4, "timestamp": (t0 + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")}
    feats = ad.analyze([s1, s2])
    # sudden increase should be detected with configured threshold 0.2
    assert feats["sudden_drowsiness_increase"] is True
    # ensure missing keys didn't break head pose or counts (they should be defaults)
    assert "head_pose_mean_pitch" in feats
    assert "total_yawns" in feats
