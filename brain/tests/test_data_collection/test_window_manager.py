from config.config_loader import ConfigLoader
from data_collection.window_manager import WindowManager
from utils.time_utils import get_utc_timestamp


def test_window_manager_basic(tmp_path):
    cfg = ConfigLoader()
    wm = WindowManager(cfg)
    # add a gps reading
    wm.add_reading("gps", {"lat": 1.0, "lon": 2.0, "timestamp": get_utc_timestamp()})
    assert wm.should_flush() is True or isinstance(wm.get_current_window(), dict)


def test_clear_window(tmp_path):
    cfg = ConfigLoader()
    wm = WindowManager(cfg)
    wm.add_reading("gps", {"lat": 1.0, "lon": 2.0, "timestamp": get_utc_timestamp()})
    wm.clear_window()
    w = wm.get_current_window()
    # after clear, window may still have metadata but no gps key
    assert "gps" not in w
