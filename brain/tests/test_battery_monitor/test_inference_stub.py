import pytest
from battery_monitor.ml.inference_stub import BatteryMLInference


class DummyConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_predict_anomaly_dummy():
    cfg = DummyConfig({"battery_monitor.ml.model_path": "nonexistent.pkl", "battery_monitor.ml.anomaly_threshold": 0.5})
    inf = BatteryMLInference(cfg)
    feats = {"voltage_imbalance": 0.3, "temp_gradient": 5.0}
    res = inf.predict_anomaly(feats)
    assert "anomaly_score" in res
    assert isinstance(res["anomaly_score"], float)


def test_predict_rul_dummy():
    cfg = DummyConfig({"battery_monitor.ml.model_path": "nonexistent.pkl"})
    inf = BatteryMLInference(cfg)
    feats = {"soh_mean": 75.0}
    r = inf.predict_rul(feats)
    assert r == pytest.approx(750.0)
