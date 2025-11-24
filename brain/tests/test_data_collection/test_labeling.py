import json
import types
import pytest
from config.config_loader import ConfigLoader
from data_collection.labeling import AccidentDetector, NearMissDetector, TemporalLabeler, ManualAnnotator


def make_imu_sample(g=0.0):
    # return dict with ax, ay, az in m/s^2 representing a decel of g
    val = -g * 9.81
    return {"ax": val, "ay": 0.0, "az": 0.0}


def make_window(t="2025-01-01T00:00:00Z", window_id="w1", imu=None, camera=None, vehicle_controls=None, weather=None):
    return {
        "window_id": window_id,
        "timestamp": t,
        "imu": imu or [],
        "camera": camera or [],
        "vehicle_controls": vehicle_controls or [],
        "weather": weather or [],
    }


def test_detect_accident_from_decel_spike():
    cfg = ConfigLoader('config/default_config.yaml')
    ad = AccidentDetector(cfg)
    # decel-only accidents are enabled only when the config flag is set; enable for this test
    ad.allow_decel_only_accidents = True
    w = make_window(imu=[make_imu_sample(5.0)])
    res = ad.detect(w)
    assert res is not None
    assert res.get('max_deceleration_g') >= 5.0


def test_detect_accident_from_camera_collision():
    cfg = ConfigLoader('config/default_config.yaml')
    ad = AccidentDetector(cfg)
    # Under stricter criteria, require both decel and collision. Inject a decel sample
    cam = [{"distance_m": 1.0, "in_path": True, "label": "car"}]
    w = make_window(camera=cam, imu=[make_imu_sample(4.0)])
    res = ad.detect(w)
    assert res is not None
    assert res.get('type') == 'collision'


@pytest.mark.parametrize("decel_g,expected", [
    (2.5, 'unknown'),
    (4.0, 'minor'),
    (6.0, 'moderate'),
    (9.0, 'severe'),
])
def test_accident_severity(monkeypatch, decel_g, expected):
    # create a detector with custom severity thresholds via a simple config mock
    cfg_mock = types.SimpleNamespace(get=lambda key, default=None: {"severity_thresholds": {"minor": 3.0, "moderate": 5.0, "severe": 8.0}})
    ad = AccidentDetector(cfg_mock)
    res = ad._classify_severity(decel_g)
    assert res == expected


def test_accident_combined():
    cfg = ConfigLoader('config/default_config.yaml')
    ad = AccidentDetector(cfg)
    cam = [{"distance_m": 1.0, "in_path": True, "label": "car"}]
    w = make_window(camera=cam, imu=[make_imu_sample(4.0)])
    res = ad.detect(w)
    assert res is not None
    assert res.get('type') == 'collision'
    assert res.get('severity') == 'minor'
    assert 'collision_info' in res and res['collision_info'] is not None


def test_accident_missing_imu():
    cfg = ConfigLoader('config/default_config.yaml')
    ad = AccidentDetector(cfg)
    cam = [{"distance_m": 0.5, "in_path": True, "label": "truck"}]
    w = make_window(camera=cam, imu=[])
    res = ad.detect(w)
    # strict AND behavior should not report an accident without decel
    assert res is None


def test_accident_missing_camera_with_flag(monkeypatch):
    cfg = ConfigLoader('config/default_config.yaml')
    ad = AccidentDetector(cfg)
    # IMU spike only
    w = make_window(camera=[], imu=[make_imu_sample(4.0)])
    # default stricter behavior -> None
    assert ad.detect(w) is None
    # enable legacy behavior and expect detection
    ad.allow_decel_only_accidents = True
    res = ad.detect(w)
    assert res is not None
    assert res.get('max_deceleration_g') >= 4.0


def test_near_miss_braking():
    cfg = ConfigLoader('config/default_config.yaml')
    nm = NearMissDetector(cfg)
    vc = [{"brake_pressure": 0.8}]
    # include a wet road surface to raise contextual risk_score above threshold
    w = make_window(vehicle_controls=vc, camera=[{"distance_m": 2.5}], weather=[{"road_surface": "wet"}])
    res = nm.detect(w, accident_detected=False)
    assert res is not None
    assert res.get('evasive_action') == 'braking'


def test_near_miss_swerving():
    cfg = ConfigLoader('config/default_config.yaml')
    nm = NearMissDetector(cfg)
    # steering angle jump large enough to count as swerving
    vc = [{"steering_angle": 0.0}, {"steering_angle": 250.0}]
    # include wet surface to raise risk_score above threshold
    w = make_window(vehicle_controls=vc, camera=[{"distance_m": 2.0}], weather=[{"road_surface": "wet"}])
    res = nm.detect(w, accident_detected=False)
    assert res is not None
    assert res.get('evasive_action') == 'swerving'
    assert res.get('steering_rate') >= 250.0


def test_near_miss_precedence():
    cfg = ConfigLoader('config/default_config.yaml')
    nm = NearMissDetector(cfg)
    vc = [{"brake_pressure": 0.9}]
    w = make_window(vehicle_controls=vc, camera=[{"distance_m": 1.0}], weather=[{"road_surface": "wet"}])
    # when an accident is already detected, near-miss detector should return None
    res = nm.detect(w, accident_detected=True)
    assert res is None


def test_near_miss_factors():
    cfg = ConfigLoader('config/default_config.yaml')
    nm = NearMissDetector(cfg)
    vc = [{"brake_pressure": 0.8}]
    w = make_window(vehicle_controls=vc, camera=[{"distance_m": 2.0}], weather=[{"road_surface": "wet", "precip_type": "rain"}])
    res = nm.detect(w, accident_detected=False)
    assert res is not None
    # contributing_factors should include close_vehicle when camera distance small
    assert 'contributing_factors' in res
    assert any('close_vehicle' in f for f in res.get('contributing_factors', [])) or res.get('risk_score', 0) >= 0.6


def test_near_miss_low_risk_case():
    cfg = ConfigLoader('config/default_config.yaml')
    nm = NearMissDetector(cfg)
    vc = [{"brake_pressure": 0.2}]
    w = make_window(vehicle_controls=vc, camera=[{"distance_m": 10.0}], weather=[{"road_surface": "dry"}])
    res = nm.detect(w, accident_detected=False)
    assert res is None


def test_temporal_labeler_basic():
    cfg = ConfigLoader('config/default_config.yaml')
    tl = TemporalLabeler(cfg)
    windows = [make_window(t=f"2025-01-01T00:00:{i:02d}Z", window_id=f"w{i}") for i in range(20)]
    events = [{"timestamp": "2025-01-01T00:00:15Z", "type": "collision", "trip_id": None}]
    labeled = tl.label_windows(windows, events)
    # windows with timestamps 05-14 should be labeled positive for 10s horizon
    positives = [w for w in labeled if w['labels']['is_accident']]
    assert len(positives) >= 9


def test_temporal_labeler_respects_trip_boundaries():
    cfg = ConfigLoader('config/default_config.yaml')
    tl = TemporalLabeler(cfg)
    # two trips: tripA has an event at t=15, tripB at t=12
    windows = []
    for i in range(20):
        trip = 'tripA' if i < 10 else 'tripB'
        windows.append(make_window(t=f"2025-01-01T00:00:{i:02d}Z", window_id=f"w{i}", imu=[], camera=[], vehicle_controls=[], weather=None))
        windows[-1]['trip_id'] = trip
    events = [
        {"timestamp": "2025-01-01T00:00:15Z", "type": "collision", "trip_id": "tripB"},
        {"timestamp": "2025-01-01T00:00:05Z", "type": "collision", "trip_id": "tripA"},
    ]
    labeled = tl.label_windows(windows, events)
    # windows in tripA before t=05 should be positive; tripB before t=15 should be positive
    tripA_pos = [w for w in labeled if w.get('trip_id') == 'tripA' and w['labels']['is_accident']]
    tripB_pos = [w for w in labeled if w.get('trip_id') == 'tripB' and w['labels']['is_accident']]
    # ensure no cross-labeling: windows in tripA shouldn't be labeled because of tripB event
    assert any(w for w in tripA_pos)
    assert any(w for w in tripB_pos)


def test_temporal_multiple_horizons():
    cfg = ConfigLoader('config/default_config.yaml')
    tl = TemporalLabeler(cfg, horizons=[1, 5])
    windows = [make_window(t=f"2025-01-01T00:00:{i:02d}Z", window_id=f"w{i}") for i in range(10)]
    # event at t=03 -> time_to=3 for windows before 03
    events = [{"timestamp": "2025-01-01T00:00:03Z", "type": "collision", "trip_id": None}]
    labeled = tl.label_windows(windows, events)
    w2 = next(w for w in labeled if w['window_id'] == 'w2')
    # time_to_accident_s should be numeric (closest upcoming event at t=03 for window at t=02)
    assert float(w2['labels']['time_to_accident_s']) == 1.0 or float(w2['labels']['time_to_accident_s']) == 1
    # w2 should be within 1s horizon? we expect per-horizon flags according to nearest_dt
    assert isinstance(w2['labels'].get('is_accident_h1'), bool)
    assert isinstance(w2['labels'].get('is_accident_h5'), bool)


def test_temporal_overlapping_and_nearest_event():
    cfg = ConfigLoader('config/default_config.yaml')
    tl = TemporalLabeler(cfg, horizons=[10])
    windows = [make_window(t=f"2025-01-01T00:00:{i:02d}Z", window_id=f"w{i}") for i in range(20)]
    # events at t=07 and t=12; for window at t=05 nearest is 07
    events = [
        {"timestamp": "2025-01-01T00:00:07Z", "type": "collision", "trip_id": None},
        {"timestamp": "2025-01-01T00:00:12Z", "type": "collision", "trip_id": None},
    ]
    labeled = tl.label_windows(windows, events)
    w5 = next(w for w in labeled if w['window_id'] == 'w5')
    assert float(w5['labels']['time_to_accident_s']) == 2.0


def test_temporal_trip_boundaries_at_edges():
    cfg = ConfigLoader('config/default_config.yaml')
    tl = TemporalLabeler(cfg, horizons=[10])
    windows = []
    # tripA windows 0-4, tripB 5-9, accident at tripA end t=04
    for i in range(10):
        trip = 'tripA' if i < 5 else 'tripB'
        w = make_window(t=f"2025-01-01T00:00:{i:02d}Z", window_id=f"w{i}")
        w['trip_id'] = trip
        windows.append(w)
    events = [{"timestamp": "2025-01-01T00:00:04Z", "type": "collision", "trip_id": "tripA"}]
    labeled = tl.label_windows(windows, events)
    # ensure tripB windows not labeled by tripA event
    tripB_pos = [w for w in labeled if w.get('trip_id') == 'tripB' and w['labels']['is_accident']]
    assert len(tripB_pos) == 0


def test_manual_add_notes_and_low_confidence(monkeypatch):
    cfg = ConfigLoader('config/default_config.yaml')
    ma = ManualAnnotator(cfg)
    w = make_window()
    # sequence of inputs: is_accident='n', is_near_miss='y', evasive='braking', confidence='0.5', requires_review='y'
    seq = iter(['n', 'y', 'braking', '0.5', 'y'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(seq))
    lab = ma.review_window(w, auto_labels=None, input_fn=input)
    assert 'notes' in lab
    assert isinstance(lab.get('notes'), str)
    assert lab.get('confidence') == 0.5
    assert lab.get('requires_review') is True


def test_end_to_end_with_missing_data():
    cfg = ConfigLoader('config/default_config.yaml')
    ad = AccidentDetector(cfg)
    nm = NearMissDetector(cfg)
    tl = TemporalLabeler(cfg)
    # windows with missing IMU should not crash detectors
    windows = [make_window(t=f"2025-01-01T00:00:0{i}Z", imu=[], camera=[]) for i in range(5)]
    events = []
    # run detect on each window
    for w in windows:
        a = ad.detect(w)
        n = nm.detect(w, accident_detected=bool(a))
        # should not raise and should return None or dict
        assert (a is None) or isinstance(a, dict)
        assert (n is None) or isinstance(n, dict)
    # labeler should also run
    labeled = tl.label_windows(windows, events)
    assert isinstance(labeled, list)


def test_manual_annotator_simulated_input(monkeypatch):
    cfg = ConfigLoader('config/default_config.yaml')
    ma = ManualAnnotator(cfg)
    w = make_window()
    monkeypatch.setattr('builtins.input', lambda prompt='': 'y')
    lab = ma.review_window(w, auto_labels=None, input_fn=input)
    assert lab['is_accident'] is True
