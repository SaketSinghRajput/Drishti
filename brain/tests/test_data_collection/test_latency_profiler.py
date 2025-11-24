import time
from data_collection.sensors.driver_drowsiness.profiler import LatencyProfiler


def test_latency_profiler_basic():
    p = LatencyProfiler(budget_ms=50, window_size=10)
    with p.profile('stage1'):
        time.sleep(0.001)
    with p.profile('stage2'):
        time.sleep(0.002)
    p.record_total(10.0)
    d = p.to_dict()
    # check profiler produced expected aggregate keys
    assert 'budget_exceeded_count' in d and 'total_mean_ms' in d
    assert abs(d['total_mean_ms'] - 10.0) < 1e-6

