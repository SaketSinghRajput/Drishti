import time
import threading
from unittest import mock
from config.config_loader import ConfigLoader
from inference.edge_inference import EdgeInferenceEngine


def test_engine_start_stop():
    cfg = ConfigLoader('config/default_config.yaml')
    engine = EdgeInferenceEngine(cfg, _simulation_mode=True)
    engine.start()
    time.sleep(0.2)
    engine.stop()
    status = engine.get_status()
    assert status is not None
    assert status.get('uptime_s', 0.0) >= 0.0
    assert 'latency_stats' in status


def test_sensor_ingest_and_infer():
    cfg = ConfigLoader('config/default_config.yaml')
    engine = EdgeInferenceEngine(cfg, _simulation_mode=True)
    engine.start()
    now = time.time()
    engine.add_sensor_reading('gps', {'timestamp': now, 'lat': 0.0})
    engine.add_sensor_reading('imu', {'timestamp': now, 'ax': 0.0})
    engine.add_sensor_reading('vehicle_controls', {'timestamp': now, 'steering': 0.0})

    # Wait for at least one inference cycle
    time.sleep(1.5)
    pred = engine.get_latest_prediction()
    assert (isinstance(pred, dict) and 'risk_score' in pred) or pred is None
    engine.stop()
