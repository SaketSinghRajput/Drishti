import math

from feature_engineering import feature_utils


def test_extract_time_series_values_and_agg():
    samples = [{"v": 1}, {"v": 3}, {"v": 2}]
    vals = feature_utils.extract_time_series_values(samples, "v")
    assert vals == [1.0, 3.0, 2.0]
    stats = feature_utils._agg_stats(vals, prefix="test_")
    assert math.isclose(stats["test_mean"], sum(vals) / len(vals))


def test_compute_derivative_and_jerk():
    samples = [
        {"timestamp": "2025-01-01T00:00:00Z", "ax": 0.0},
        {"timestamp": "2025-01-01T00:00:01Z", "ax": 1.0},
        {"timestamp": "2025-01-01T00:00:02Z", "ax": 2.0},
    ]
    deriv = feature_utils.compute_derivative(samples, "ax")
    assert len(deriv) == 2
    jerk = feature_utils.compute_jerk_from_acceleration(samples)
    assert "jerk_magnitude_mean" in jerk


def test_compute_trend_and_delta():
    samples = [
        {"timestamp": "2025-01-01T00:00:00Z", "v": 0},
        {"timestamp": "2025-01-01T00:00:01Z", "v": 2},
        {"timestamp": "2025-01-01T00:00:02Z", "v": 4},
    ]
    slope = feature_utils.compute_trend_slope(samples, "v")
    assert slope > 0
    vals = [0, 2, 4]
    delta = feature_utils.compute_delta_features(vals, "v")
    assert delta["v_delta_total"] == 4
