import time
import threading

from brain.data_collection.driver_fatigue_adapter_hand import compute_grip_features


def test_grip_detection_enabled():
    # simulate samples with grip confidence
    samples = [{'grip_confidence': 0.9} for _ in range(10)]
    feats = compute_grip_features(samples, sample_rate_hz=10.0)
    assert 'grip_confidence_mean' in feats
    assert feats['grip_confidence_mean'] > 0.8


def test_no_hands_detected():
    samples = [{'grip_confidence': 0.0} for _ in range(5)]
    feats = compute_grip_features(samples, sample_rate_hz=10.0)
    assert feats['grip_confidence_min'] == 0.0


def test_sudden_grip_loss():
    # simulate drop from 0.9 to 0.2 within 2s window (sample_rate 10 -> window 20 steps)
    samples = [ {'grip_confidence': 0.9} for _ in range(10)] + [ {'grip_confidence': 0.2} for _ in range(10)]
    feats = compute_grip_features(samples, sample_rate_hz=10.0)
    assert feats.get('sudden_grip_loss_flag') is True
