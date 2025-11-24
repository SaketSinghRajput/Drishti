from config.config_loader import ConfigLoader
from data_collection.sensors import GPSHandler, IMUHandler, VehicleControlsHandler, DriverFatigueHandler


def _write_minimal_cfg(p):
    p.write_text(
        "sensors:\n  gps:\n    sample_rate: 1\ndata_collection:\n  storage_path: '/tmp'\n  window_size_s: 1\n  window_interval_s: 1\n  buffer_retention_hours: 1\nmodel:\n  prediction_horizon_s: 5\n  risk_thresholds: {low: 0.1}\ninference:\n  latency_budget_ms: 100\ntelemetry:\n  log_level: INFO\nprivacy:\n  encrypt_storage: false\n"
    )


def test_sensor_handlers_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_minimal_cfg(p)
    cfg = ConfigLoader(str(p))

    handlers = [GPSHandler(cfg), IMUHandler(cfg), VehicleControlsHandler(cfg), DriverFatigueHandler(cfg)]
    for h in handlers:
        # perform several reads to exercise success and health tracking
        successes = 0
        for _ in range(5):
            r = h.read()
            if r:
                successes += 1

        health = h.get_health()
        # health must include availability key and it should be numeric
        assert "availability" in health
        assert isinstance(health["availability"], (int, float))
        # if at least one successful read happened, availability should be > 0
        if successes > 0:
            assert health["availability"] > 0
