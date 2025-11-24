import time
import threading
from config.config_loader import ConfigLoader
from inference.decision_unit import DecisionUnit


def test_thresholds_and_hysteresis():
    cfg = ConfigLoader('config/default_config.yaml')
    du = DecisionUnit(cfg)

    # low
    d = du.update(0.05, {}, 1.0)
    assert d['alert_level'] == 'none'

    d = du.update(0.15, {"gps": {"stale": False}}, 1.0)
    # May not immediately escalate due to margin/dwell; force timestamp
    # Simulate enough time passed
    du._last_change_ts = time.time() - 10
    d = du.update(0.35, {"gps": {"stale": False}, "imu": {"stale": False}, "vehicle_controls": {"stale": False}}, 1.0)
    assert d['alert_level'] != 'none'

    # Reset
    du.reset()
    s = du.get_state()
    assert s['current_level'] == 'none'


def test_uncertainty_flags():
    cfg = ConfigLoader('config/default_config.yaml')
    du = DecisionUnit(cfg)
    # Missing critical sensor
    d = du.update(0.2, {"gps": {"stale": True}}, 1.0)
    assert d['uncertainty_flag']

    # Low quality
    d = du.update(0.2, {"gps": {"stale": False}, "imu": {"stale": False}, "vehicle_controls": {"stale": False}}, 0.3)
    assert d['uncertainty_flag']


def test_thread_safety_updates():
    cfg = ConfigLoader('config/default_config.yaml')
    du = DecisionUnit(cfg)

    def worker():
        for i in range(10):
            du.update(0.1 + i * 0.05, {"gps": {"stale": False}}, 1.0)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = du.get_state()
    assert 'current_level' in s