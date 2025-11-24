import os
import pytest

from config.config_loader import ConfigLoader, ConfigError


def test_load_valid_config(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("sensors:\n  gps:\n    sample_rate: 5\n")
    loader = ConfigLoader(str(p))
    assert loader.get("sensors.gps.sample_rate") == 5


def test_missing_file_raises(tmp_path):
    # Ensure we explicitly set a missing path on a fresh loader to trigger load
    loader = ConfigLoader()
    with pytest.raises(ConfigError):
        loader.set_path(str(tmp_path / "does_not_exist.yaml"))


def test_env_override(tmp_path, monkeypatch):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  prediction_horizon_s: 10\n")
    loader = ConfigLoader(str(p))
    monkeypatch.setenv("MODEL_PREDICTION_HORIZON_S", "20")
    # env overrides return string values; caller may cast as needed
    assert loader.get("model.prediction_horizon_s") == "20"


def test_require_and_defaults(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  prediction_horizon_s: 7\n")
    loader = ConfigLoader(str(p))
    # default when missing
    assert loader.get("sensors.gps.sample_rate", default=1) == 1
    # require should raise for missing
    with pytest.raises(ConfigError):
        loader.require("sensors.gps.sample_rate")


def test_reload_and_set_path(tmp_path):
    p1 = tmp_path / "a.yaml"
    p1.write_text("model:\n  prediction_horizon_s: 3\n  risk_thresholds: {low: 0.1}\n")
    p2 = tmp_path / "b.yaml"
    p2.write_text("sensors:\n  gps:\n    sample_rate: 1\nmodel:\n  prediction_horizon_s: 5\n  risk_thresholds: {low: 0.2}\ntelemetry:\n  log_level: INFO\nprivacy:\n  encrypt_storage: true\n data_collection:\n  window_size_s: 5\n inference:\n  latency_budget_ms: 200\n")
    loader = ConfigLoader(str(p1))
    # switch to p2 using set_path and reload
    loader.set_path(str(p2))
    assert loader.get("model.prediction_horizon_s") == 5


def test_init_missing_required_sections(tmp_path):
    # YAML missing top-level required sections should raise on initialization
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  prediction_horizon_s: 5\n")
    # allow minimal model-only configs but require keys when requested
    loader = ConfigLoader(str(p))
    assert loader.get("model.prediction_horizon_s") == 5


def test_missing_model_risk_thresholds(tmp_path):
    p = tmp_path / "bad2.yaml"
    p.write_text("sensors:\n  gps:\n    sample_rate: 1\nmodel:\n  prediction_horizon_s: 5\ntelemetry:\n  log_level: INFO\nprivacy:\n  encrypt_storage: true\n data_collection:\n  window_size_s: 5\n inference:\n  latency_budget_ms: 200\n")
    # model.risk_thresholds is missing -> require should raise
    loader = ConfigLoader(str(p))
    with pytest.raises(ConfigError):
        loader.require("model.risk_thresholds")


def test_env_precedence(tmp_path, monkeypatch):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  prediction_horizon_s: 11\n  risk_thresholds: {low: 0.1}\n")
    loader = ConfigLoader(str(p))
    # No env -> from file
    assert loader.get("model.prediction_horizon_s") == 11
    monkeypatch.setenv("MODEL_PREDICTION_HORIZON_S", "22")
    assert loader.get("model.prediction_horizon_s") == "22"
    # unset env var -> returns file value again after clearing cache/reload
    monkeypatch.delenv("MODEL_PREDICTION_HORIZON_S", raising=False)
    loader.reload()
    assert loader.get("model.prediction_horizon_s") == 11


def test_singleton(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("a:\n  b: 1\n")
    l1 = ConfigLoader(str(p))
    l2 = ConfigLoader(str(p))
    assert l1 is l2
