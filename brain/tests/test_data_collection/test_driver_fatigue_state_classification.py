import time
import types
import sys

import pytest

from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
from data_collection.driver_fatigue_adapter import DriverFatigueAdapter


class _FakeCamera:
    def __init__(self):
        self.is_ir = False

    def read_frame(self):
        return object()

    def mark_no_face_detected(self):
        return None

    def clear_no_face_counter(self):
        return None


class _FakeDetector:
    def __init__(self, ear=0.1, mar=0.1, pitch=0.0, yaw=0.0, confidence=0.9):
        self._ear = ear
        self._mar = mar
        self._pitch = pitch
        self._yaw = yaw
        self._confidence = confidence

    def process_frame(self, frame):
        return {
            "ear": self._ear,
            "mar": self._mar,
            "head_pose": {"pitch": self._pitch, "yaw": self._yaw, "roll": 0.0},
            "confidence": self._confidence,
            "no_face_detected": False,
            "blinks_detected": 0,
            "total_yawns": 0,
            "micro_sleep_flag": False,
        }


def test_weighted_score_and_state_mapping():
    cfg = {
        "sensors.driver_fatigue.sample_rate": 1,
        "sensors.driver_fatigue": {
            "ear_threshold": 0.25,
            "mar_threshold": 0.6,
            "head_tilt_threshold": 30.0,
            "state_classification": {"fatigue_low": 0.3, "drowsy": 0.6, "micro": 0.8, "enable": True, "state_hysteresis_dwell_s": 0.0, "state_hysteresis_margin": 0.0},
        },
    }

    h = DriverFatigueHandler(cfg, use_cv=False)
    # ensure handler thresholds and hysteresis are set for the test (plain dict config
    # used in tests doesn't support dot-lookup used by the handler)
    h._threshold_fatigue_low = 0.3
    h._threshold_drowsy = 0.6
    h._threshold_micro = 0.8
    h._hysteresis_dwell_s = 0.0
    h._hysteresis_margin = 0.0
    # inject fake CV resources
    h._camera = _FakeCamera()
    # set detector to produce a high eye-fatigue signal (very small EAR)
    h._detector = _FakeDetector(ear=0.01, mar=0.1)
    h._profiler = None

    sample = h.read()
    assert "drowsiness_score" in sample
    score = sample["drowsiness_score"]
    # with ear very small, eye_fatigue ~ 1.0 -> score should be > fatigue_low (>=0.3)
    assert score >= 0.3
    # because we configured immediate transitions, an event should have been emitted
    evs = h.get_events()
    assert isinstance(evs, list)
    assert len(evs) >= 1
    ev = evs[-1]
    assert set(ev.keys()) >= {"timestamp", "state", "prev_state", "score", "reason"}
    # reason should be descriptive (EAR dominant in our input)
    assert "EAR" in ev["reason"] or "Ear" in ev["reason"]


def test_hysteresis_prevents_immediate_deescalation():
    cfg = {
        "sensors.driver_fatigue.sample_rate": 1,
        "sensors.driver_fatigue": {
            "ear_threshold": 0.25,
            "mar_threshold": 0.6,
            "head_tilt_threshold": 30.0,
            "state_classification": {"fatigue_low": 0.3, "drowsy": 0.6, "micro": 0.8, "enable": True, "state_hysteresis_dwell_s": 2.0, "state_hysteresis_margin": 0.0},
        },
    }

    h = DriverFatigueHandler(cfg, use_cv=False)
    # ensure thresholds/hysteresis set
    h._threshold_fatigue_low = 0.3
    h._threshold_drowsy = 0.6
    h._threshold_micro = 0.8
    h._hysteresis_dwell_s = 2.0
    h._hysteresis_margin = 0.0
    h._camera = _FakeCamera()
    h._profiler = None
    # first produce a high-fatigue reading to force a transition
    h._detector = _FakeDetector(ear=0.01, mar=0.2)
    s1 = h.read()
    evs1 = h.get_events()
    assert len(evs1) >= 1
    prev_state = s1.get("fatigue_state")

    # simulate immediate previous change so dwell prevents de-escalation
    h._last_state_change_ts = time.time()
    h._detector = _FakeDetector(ear=0.3, mar=0.1)
    target = h._classify_state(0.0)
    allowed = h._should_transition(prev_state, target, 0.0, time.time())
    assert allowed is False


def test_adapter_integration_with_events():
    # build a small sample list and events list and ensure adapter returns state features
    samples = []
    now = time.time()
    samples.append({"timestamp": now - 2, "drowsiness_score": 0.1, "fatigue_state": "NORMAL"})
    samples.append({"timestamp": now - 1, "drowsiness_score": 0.7, "fatigue_state": "DROWSY"})
    events = [
        {"timestamp": now - 1.5, "state": "TIRED", "prev_state": "NORMAL", "score": 0.35, "reason": "EAR below threshold"},
        {"timestamp": now - 1.0, "state": "DROWSY", "prev_state": "TIRED", "score": 0.7, "reason": "Yawning detected"},
    ]

    from config.config_loader import ConfigLoader

    # adapter expects a config loader or dict-like; pass a simple dict
    adapter = DriverFatigueAdapter({"sensors.driver_fatigue": {}})
    features = adapter.analyze(samples, events=events)
    assert features is not None
    assert features.get("state_transitions_count") == len(events)
    assert features.get("max_state_level") is not None
    assert features.get("current_state") == "DROWSY"
