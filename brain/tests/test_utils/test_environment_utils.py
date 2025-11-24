import pytest

from utils.environment_utils import (
    categorize_time_of_day,
    infer_road_surface_from_weather,
    categorize_visibility,
    compute_weather_risk_score,
    compute_road_geometry_risk_score,
)


@pytest.mark.parametrize(
    "ts,expected",
    [
        ("2025-01-01T02:00:00Z", "night"),
        ("2025-01-01T08:00:00Z", "morning"),
        ("2025-01-01T14:00:00Z", "afternoon"),
        ("2025-01-01T19:00:00Z", "evening"),
        ("not-a-time", "unknown"),
    ],
)
def test_categorize_time_of_day(ts, expected):
    assert categorize_time_of_day(ts) == expected


@pytest.mark.parametrize(
    "precip, temp, expected",
    [
        ("none", None, "dry"),
        ("rain", None, "wet"),
        ("snow", -5.0, "icy"),
        ("snow", 2.0, "wet"),
    ],
)
def test_infer_road_surface_from_weather(precip, temp, expected):
    assert infer_road_surface_from_weather(precip, temp) == expected


@pytest.mark.parametrize(
    "vis,thresholds,expected",
    [
        (50, {"low": 200, "medium": 1000}, "low"),
        (500, {"low": 200, "medium": 1000}, "medium"),
        (2000, {"low": 200, "medium": 1000}, "high"),
        (None, {"low": 200, "medium": 1000}, "high"),
    ],
)
def test_categorize_visibility(vis, thresholds, expected):
    assert categorize_visibility(vis, thresholds) == expected


def test_compute_weather_risk_score_monotonic():
    # none -> low risk
    s_none = compute_weather_risk_score('none', 0.0, 2000, 'dry')
    # rain low intensity
    s_rain = compute_weather_risk_score('rain', 1.0, 1000, 'wet')
    # snow high intensity
    s_snow = compute_weather_risk_score('snow', 10.0, 50, 'icy')
    assert 0.0 <= s_none <= 1.0
    assert s_rain >= s_none
    assert s_snow >= s_rain
    assert s_snow <= 1.0


def test_compute_road_geometry_risk_score_monotonic():
    # gentle straight road
    s_straight = compute_road_geometry_risk_score(2000.0, None, False, 0.0, 40.0)
    # moderate curve
    s_curve = compute_road_geometry_risk_score(150.0, 'left', False, 2.0, 60.0)
    # sharp curve + intersection + slope
    s_sharp = compute_road_geometry_risk_score(20.0, 'sharp', True, 10.0, 100.0)
    assert 0.0 <= s_straight <= 1.0
    assert s_curve >= s_straight
    assert s_sharp >= s_curve
    assert s_sharp <= 1.0
