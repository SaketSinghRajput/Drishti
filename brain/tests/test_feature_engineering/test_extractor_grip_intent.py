import pytest
from brain.feature_engineering.extractor_interactions import grip_loss_speed_interaction, pedestrian_intent_proximity_interaction


def test_grip_loss_speed_interaction():
    driver = {'grip_confidence_mean': 0.2}
    camera = {'speed_mean': 80}
    out = grip_loss_speed_interaction(driver, camera)
    assert pytest.approx(out['interaction_grip_loss_speed_interaction'], rel=1e-5) == (1 - 0.2) * 80


def test_pedestrian_intent_proximity_interaction():
    camera = {'pedestrian_intent_score_max': 0.9}
    misc = {'min_distance_m': 5.0}
    out = pedestrian_intent_proximity_interaction(camera, misc)
    assert pytest.approx(out['interaction_pedestrian_intent_proximity_interaction'], rel=1e-5) == 0.9 * (1.0 / 5.0)


def test_interactions_disabled_missing():
    assert grip_loss_speed_interaction({}, {}) == {}
    assert pedestrian_intent_proximity_interaction({}, {}) == {}
