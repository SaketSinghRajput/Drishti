import time
import threading
from config.config_loader import ConfigLoader
from inference.sliding_window_buffer import SlidingWindowBuffer


def test_sliding_window_basic(tmp_path):
    cfg = ConfigLoader('config/default_config.yaml')
    buf = SlidingWindowBuffer(cfg, vehicle_hash='v1', trip_id='t1')
    assert not buf.is_ready()

    t = time.time()
    gps = {'timestamp': t, 'lat': 0.0}
    imu = {'timestamp': t, 'ax': 0.1}
    vc = {'timestamp': t, 'steering': 0.0}

    buf.add_reading('gps', gps)
    assert not buf.is_ready()
    buf.add_reading('imu', imu)
    buf.add_reading('vehicle_controls', vc)
    # Add other expected sensors minimally so buffer meets readiness criteria
    buf.add_reading('driver_fatigue', {'timestamp': t, 'score': 0.5})
    buf.add_reading('ble_mesh', {'timestamp': t, 'rssi': -60})
    buf.add_reading('camera', {'timestamp': t, 'frame_id': 1})
    assert buf.is_ready()

    window = buf.get_current_window()
    # New schema places timestamp/window_id/trip_id and metadata at top-level
    assert 'timestamp' in window
    assert 'window_id' in window
    assert 'trip_id' in window
    assert 'metadata' in window and 'vehicle_hash' in window['metadata']

    # Test clear
    buf.clear()
    assert not buf.is_ready()


def test_thread_safety(tmp_path):
    cfg = ConfigLoader('config/default_config.yaml')
    buf = SlidingWindowBuffer(cfg)

    def writer(name):
        for i in range(100):
            buf.add_reading(name, {'timestamp': time.time(), 'v': i})

    threads = [threading.Thread(target=writer, args=("gps",)), threading.Thread(target=writer, args=("imu",))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Attempt to snapshot while writes happen
    w = buf.get_current_window()
    assert isinstance(w, dict)
