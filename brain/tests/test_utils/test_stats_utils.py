from utils.stats_utils import safe_mean, safe_std, safe_min, safe_max, compute_percentile, linear_trend, rolling_window_stats


def test_safe_mean_and_min_max():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert safe_mean(vals) == 3.0
    assert safe_min(vals) == 1.0
    assert safe_max(vals) == 5.0


def test_safe_std_and_percentile():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert round(safe_std(vals), 3) == round(1.41421356237, 3)
    assert compute_percentile(vals, 50) == 3.0


def test_linear_trend_and_rolling():
    inc = [1, 2, 3, 4, 5]
    assert linear_trend(inc) == "increasing"
    dec = [5, 4, 3, 2, 1]
    assert linear_trend(dec) == "decreasing"
    stable = [3.0, 3.01, 2.99, 3.0]
    assert linear_trend(stable, slope_threshold=0.02) == "stable"

    r = rolling_window_stats([1,2,3,4,5], 3)
    assert isinstance(r, list) and len(r) == 3


def test_stats_edge_cases_and_percentile():
    # empty sequences
    assert safe_mean([]) == 0.0
    # current implementation returns 0.0 for empty min/max
    assert safe_min([]) == 0.0
    assert safe_max([]) == 0.0
    # single-value std should be 0.0
    assert safe_std([42.0]) == 0.0
    # percentile 95 on known sequence
    vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p95 = compute_percentile(vals, 95)
    # 95th percentile of 0..9 should be close to 8.55 depending on method; assert within expected range
    assert 8.0 <= p95 <= 9.0
