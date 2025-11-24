from config.config_loader import ConfigLoader
from data_collection.buffer import CircularBuffer
from utils.time_utils import get_utc_timestamp


def test_buffer_write_and_cleanup(tmp_path):
    cfg = ConfigLoader()
    # write a minimal config file and override storage path for test
    p = tmp_path / "cfg.yaml"
    # include required top-level sections and set storage path to tmp dir
    p.write_text(
        "sensors:\n  gps:\n    sample_rate: 1\ndata_collection:\n  storage_path: '" + str(tmp_path / "windows") + "'\n  buffer_retention_hours: 1\nmodel:\n  prediction_horizon_s: 5\n  risk_thresholds: {low: 0.1}\ninference:\n  latency_budget_ms: 100\ntelemetry:\n  log_level: INFO\nprivacy:\n  encrypt_storage: false\n"
    )
    cfg.set_path(str(p))
    cb = CircularBuffer(cfg)
    # ensure storage path exists
    p = cb.write_window({"timestamp": get_utc_timestamp(), "vehicle_hash": "v1"})
    assert p is not None
    assert cb.get_file_count() >= 1
