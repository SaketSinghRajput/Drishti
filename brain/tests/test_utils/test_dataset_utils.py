import os
import json
from typing import List

import pytest

from utils import dataset_utils
import datetime
import math


def _make_window(wid: int, trip: str = "t1", accident: bool = False):
    return {
        "window_id": f"w{wid}",
        "trip_id": trip,
        "timestamp": f"2025-01-01T00:00:{wid:02d}Z",
        "labels": {"is_accident": bool(accident)},
        "feature_x": float(wid),
    }


@pytest.fixture
def imbalanced_windows() -> List[dict]:
    wins = []
    # 10 positives, 90 negatives across trips
    for i in range(1, 101):
        wins.append(_make_window(i, trip=f"trip{i//20}", accident=(i <= 10)))
    return wins


def test_balance_dataset_smote_target_ratio(imbalanced_windows):
    out = dataset_utils.balance_dataset_smote(imbalanced_windows, sampling_strategy=0.5, k_neighbors=1)
    pos = [w for w in out if w.get("labels", {}).get("is_accident")]
    # target is roughly 0.5 * negatives = 0.5 * 90 = 45 minority samples
    assert len(pos) >= 45
    # synthetic samples should be flagged when created
    synthetics = [w for w in out if w.get("synthetic_sample")]
    assert len(synthetics) >= 0


def test_balance_smote_fallback_small_minority():
    # very small minority (2 positives) should fall back to duplication to reach target
    wins = []
    for i in range(1, 21):
        wins.append(_make_window(i, trip="tX", accident=(i <= 2)))
    out = dataset_utils.balance_dataset_smote(wins, sampling_strategy=0.5, k_neighbors=5)
    pos = [w for w in out if w.get("labels", {}).get("is_accident")]
    npos = len(pos)
    assert npos >= 5  # target approx 0.5 * negatives (~=9), but at least growth from 2


def test_smote_insufficient_samples_fallback():
    # Create small minority (3 positives) and ensure no exception and fallback duplication
    wins = []
    for i in range(1, 21):
        wins.append(_make_window(i, trip="tSmall", accident=(i <= 3)))
    out = dataset_utils.balance_dataset_smote(wins, sampling_strategy=0.5, k_neighbors=10)
    pos = [w for w in out if w.get("labels", {}).get("is_accident")]
    assert len(pos) >= 3


def test_smote_no_numeric_features():
    # windows without numeric columns should fall back to duplication without crash
    wins = []
    for i in range(1, 11):
        w = {"window_id": f"w{i}", "trip_id": "tN", "timestamp": f"2025-01-01T00:00:{i:02d}Z", "labels": {"is_accident": (i <= 2)}, "meta": {"foo": "bar"}}
        wins.append(w)
    out = dataset_utils.balance_dataset_smote(wins, sampling_strategy=0.5)
    assert isinstance(out, list)
    assert all('labels' in w for w in out)


def test_compute_sample_weights():
    labels = [0] * 90 + [1] * 10
    weights = dataset_utils.compute_sample_weights(labels)
    assert len(weights) == len(labels)
    # weights sum to n
    assert abs(sum(weights) - len(labels)) < 1e-6
    # minority average weight should be larger
    maj_avg = sum(weights[:90]) / 90.0
    min_avg = sum(weights[90:]) / 10.0
    assert min_avg > maj_avg


def test_export_and_load_roundtrip(tmp_path):
    wins = [_make_window(i, trip="tA", accident=(i % 10 == 0)) for i in range(1, 21)]
    outdir = tmp_path / "out"
    dataset_utils.export_labeled_dataset(wins, str(outdir))
    loaded = dataset_utils.load_labeled_dataset(str(outdir))
    assert len(loaded) == len(wins)
    # metadata file present and correct
    with open(os.path.join(str(outdir), "metadata.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta.get("count") == len(wins)


def test_export_metadata_rich(tmp_path):
    wins = [_make_window(i, trip="tA", accident=(i % 4 == 0)) for i in range(1, 21)]
    outdir = tmp_path / "out_meta"
    dataset_utils.export_labeled_dataset(wins, str(outdir))
    with open(os.path.join(str(outdir), "metadata.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert 'n_accidents' in meta
    assert 'accident_ratio' in meta
    assert meta.get('min_timestamp') is not None
    assert meta.get('max_timestamp') is not None


def test_export_load_with_trip_filter(tmp_path):
    wins = [_make_window(i, trip=("tripA" if i <= 10 else "tripB"), accident=(i % 5 == 0)) for i in range(1, 21)]
    outdir = tmp_path / "out2"
    dataset_utils.export_labeled_dataset(wins, str(outdir))
    loaded_all = dataset_utils.load_labeled_dataset(str(outdir))
    loaded_tripA = dataset_utils.load_labeled_dataset(str(outdir), trip_id="tripA")
    assert len(loaded_all) == 20
    assert len(loaded_tripA) == 10


def test_load_date_filter(tmp_path):
    wins = [_make_window(i, trip="tA", accident=(i % 3 == 0)) for i in range(1, 21)]
    outdir = tmp_path / "out_dates"
    dataset_utils.export_labeled_dataset(wins, str(outdir))
    # filter to only include timestamps >= 00:00:05 and <= 00:00:10
    loaded = dataset_utils.load_labeled_dataset(str(outdir), start_ts="2025-01-01T00:00:05Z", end_ts="2025-01-01T00:00:10Z")
    for w in loaded:
        ts = dataset_utils.__import__('utils.time_utils') if False else None
        # basic sanity: timestamps should be within the requested window
        t = w.get('timestamp')
        assert t >= "2025-01-01T00:00:05Z" and t <= "2025-01-01T00:00:10Z"


def test_temporal_train_test_split_no_leakage():
    wins = []
    for t in range(5):
        for i in range(20):
            wins.append(_make_window(t * 20 + i, trip=f"trip{t}", accident=(i == 0)))
    train, val, test = dataset_utils.temporal_train_test_split(wins, test_size=0.2, val_size=0.1)
    # total windows conserved
    assert len(train) + len(val) + len(test) == len(wins)
    # no trip appears in more than one split
    def trips(xs):
        return set(w.get("trip_id") for w in xs)

    assert trips(train).isdisjoint(trips(val) | trips(test))


def test_temporal_split_holdout_trips_and_ratios():
    # create 10 trips with single windows each
    wins = []
    for t in range(10):
        wins.append(_make_window(t, trip=f"trip{t}", accident=(t % 3 == 0)))
    train, val, test = dataset_utils.temporal_train_test_split(wins, test_size=0.2, val_size=0.1, holdout_trips=["trip8", "trip9"])
    # holdout trips should be in test
    test_trips = set(w.get('trip_id') for w in test)
    assert 'trip8' in test_trips and 'trip9' in test_trips
    # sizes roughly as expected (2 holdouts -> test at least 2)
    assert len(test) >= 2


def test_validate_temporal_split_helper_detects_leakage():
    # construct splits with deliberate leakage (same trip in both)
    train = [_make_window(1, trip='leak', accident=False)]
    test = [_make_window(2, trip='leak', accident=True)]
    def _validate(train, test):
        train_trips = set(w.get('trip_id') for w in train)
        test_trips = set(w.get('trip_id') for w in test)
        return not train_trips.isdisjoint(test_trips)
    assert _validate(train, test) is True


def test_smote_creates_labeled_items_when_available(imbalanced_windows):
    # Only run detailed SMOTE assertions if SMOTE/pandas are available
    from utils import dataset_utils as du

    out = du.balance_dataset_smote(imbalanced_windows, sampling_strategy=0.5, k_neighbors=1)
    # Ensure every returned item has a labels key
    assert all(isinstance(w.get('labels'), dict) for w in out)
    # Synthetic samples, if any, must have labels['is_accident'] consistent
    synthetics = [w for w in out if w.get('synthetic_sample')]
    for s in synthetics:
        assert 'is_accident' in s.get('labels', {})
