import time
import json

from config.config_loader import ConfigLoader
from data_collection.logger import DataLogger


def _write_minimal_cfg(p, storage_path):
    p.write_text(
        "sensors:\n  gps:\n    sample_rate: 5\ndata_collection:\n  storage_path: '" + str(storage_path).replace("\\", "/") + "'\n  window_size_s: 1\n  window_interval_s: 0.5\n  buffer_retention_hours: 1\nmodel:\n  prediction_horizon_s: 5\n  risk_thresholds: {low: 0.1}\ninference:\n  latency_budget_ms: 100\ntelemetry:\n  log_level: INFO\nprivacy:\n  encrypt_storage: false\n"
    )


def test_datalogger_writes_windows(tmp_path):
    storage = tmp_path / "windows"
    storage.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_cfg(cfg_path, storage)

    cfg = ConfigLoader(str(cfg_path))
    dl = DataLogger(cfg)
    dl.start()
    # run briefly to collect and flush a couple windows
    time.sleep(2.0)
    dl.stop()

    files = list(storage.iterdir())
    assert len(files) >= 1

    # verify trip_id is consistent across written windows
    trip_ids = set()
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                j = json.load(fh)
                if "trip_id" in j:
                    trip_ids.add(j["trip_id"])
        except Exception:
            continue

    assert len(trip_ids) <= 1


    def test_logger_ble_mesh_integration(tmp_path):
        storage = tmp_path / "windows"
        storage.mkdir()
        cfg_path = tmp_path / "cfg.yaml"
        _write_minimal_cfg(cfg_path, storage)

        cfg = ConfigLoader(str(cfg_path))
        dl = DataLogger(cfg)
        dl.start()
        import time

        time.sleep(2.0)
        dl.stop()

        found_raw = False
        found_features = False
        for f in storage.iterdir():
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    j = json.load(fh)
                    if "ble_mesh" in j and isinstance(j.get("ble_mesh"), list):
                        found_raw = True
                    if "ble_mesh_features" in j and isinstance(j.get("ble_mesh_features"), dict):
                        found_features = True
            except Exception:
                continue

        assert found_raw and found_features, "Expected at least one window with ble_mesh and ble_mesh_features"


def test_datalogger_driver_fatigue_integration(tmp_path):
    # storage and config
    storage = tmp_path / "windows"
    storage.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_cfg(cfg_path, storage)

    cfg = ConfigLoader(str(cfg_path))
    dl = DataLogger(cfg)
    dl.start()
    # run briefly to collect and flush at least one window
    import time

    time.sleep(2.0)
    dl.stop()

    found = False
    for f in storage.iterdir():
        try:
            with open(f, "r", encoding="utf-8") as fh:
                j = json.load(fh)
                df = j.get("driver_fatigue_features")
                if isinstance(df, dict) and "drowsiness_score_mean" in df:
                    found = True
                    break
        except Exception:
            continue

    assert found, "Expected at least one written window to include driver_fatigue_features"


def test_logger_writes_when_no_driver_fatigue(monkeypatch, tmp_path):
    # Ensure logger writes windows even if WindowManager returns no driver_fatigue key
    storage = tmp_path / "windows"
    storage.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_cfg(cfg_path, storage)

    cfg = ConfigLoader(str(cfg_path))
    dl = DataLogger(cfg)

    # replace should_flush to trigger exactly one flush
    calls = {"n": 0}

    def should_flush_once():
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(dl.window_manager, "should_flush", should_flush_once)

    # return a window that intentionally lacks 'driver_fatigue'
    def fake_get_current_window(sensor_health_snapshot=None):
        return {"timestamp": "2020-01-01T00:00:00Z", "window_id": "w1"}

    monkeypatch.setattr(dl.window_manager, "get_current_window", fake_get_current_window)

    # run one loop iteration of flush loop in background
    t = __import__("threading").Thread(target=dl._window_flush_loop, daemon=True)
    t.start()
    import time

    time.sleep(1.0)
    dl._stop_event.set()
    t.join(timeout=1.0)

    files = list(storage.iterdir())
    assert len(files) >= 1
    # verify written windows do not include driver_fatigue_features
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            j = json.load(fh)
            assert "driver_fatigue_features" not in j


def test_driver_fatigue_adapter_exception_logged(monkeypatch, caplog, tmp_path):
    storage = tmp_path / "windows"
    storage.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_cfg(cfg_path, storage)

    cfg = ConfigLoader(str(cfg_path))
    dl = DataLogger(cfg)

    # monkeypatch adapter to raise
    def bad_analyze(_):
        raise RuntimeError("adapter failure")

    monkeypatch.setattr(dl.fatigue_adapter, "analyze", bad_analyze)

    # make flush happen once
    calls = {"n": 0}

    def should_flush_once():
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(dl.window_manager, "should_flush", should_flush_once)

    # provide a window with driver_fatigue samples so adapter would be called
    def fake_get_current_window(sensor_health_snapshot=None):
        return {"timestamp": "2020-01-01T00:00:00Z", "window_id": "w1", "driver_fatigue": [{"timestamp": "2020-01-01T00:00:00Z", "drowsiness_score": 0.1}]}

    monkeypatch.setattr(dl.window_manager, "get_current_window", fake_get_current_window)

    caplog.set_level("WARNING")
    t = __import__("threading").Thread(target=dl._window_flush_loop, daemon=True)
    t.start()
    import time

    time.sleep(1.0)
    dl._stop_event.set()
    t.join(timeout=1.0)

    # adapter raised; DataLogger should have logged a warning but still written a window
    assert any("Failed to compute driver fatigue features" in m.message for m in caplog.records)
    files = list(storage.iterdir())
    assert len(files) >= 1


def test_logger_environment_features_integration(tmp_path):
    storage = tmp_path / "windows"
    storage.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_cfg(cfg_path, storage)

    cfg = ConfigLoader(str(cfg_path))
    dl = DataLogger(cfg)
    dl.start()
    import time

    time.sleep(2.0)
    dl.stop()

    found = False
    for f in storage.iterdir():
        try:
            with open(f, "r", encoding="utf-8") as fh:
                j = json.load(fh)
                ef = j.get("environment_features")
                if isinstance(ef, dict) and all(k in ef for k in ("weather_risk_score", "road_geometry_risk_score", "time_of_day")):
                    found = True
                    break
        except Exception:
            continue

    assert found, "Expected at least one window to include environment_features with anchor fields"
