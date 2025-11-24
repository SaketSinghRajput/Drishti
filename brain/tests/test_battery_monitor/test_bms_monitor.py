import pytest
from unittest.mock import Mock

from battery_monitor.rules.bms_monitor import BMSMonitor


class DummyConfig(dict):
    def get(self, key, default=None):
        # simple dotted-key support
        if isinstance(key, str) and key.startswith("battery_monitor"):
            parts = key.split('.')
            cur = self
            for p in parts:
                if p in cur:
                    cur = cur[p]
                else:
                    return default
            return cur
        return super().get(key, default)


@pytest.fixture
def telemetry_collector():
    tc = Mock()
    tc.log_battery_alert = Mock()
    return tc


@pytest.fixture
def config():
    cfg = DummyConfig({
        "battery_monitor": {
            "ml": {"enable": True},
            "rules": {"enable_soh_tracking": True}
        }
    })
    return cfg


def make_monitor_with_mocks(config, telemetry_collector, rule_alerts=None, ml_result=None):
    # rule_alerts: list of dicts returned by RuleEngine.evaluate
    # ml_result: dict returned by predict_anomaly
    from battery_monitor.rules import RuleEngine
    from battery_monitor.ml import BatteryMLInference, BMSFeatureExtractor

    # instantiate monitor
    mon = BMSMonitor(config, telemetry_collector)

    # replace rule_engine
    mon.rule_engine = Mock()
    mon.rule_engine.evaluate = Mock(return_value=rule_alerts or [])

    # replace extractor
    mon.extractor = Mock()
    mon.extractor.extract_features = Mock(return_value={})

    # replace ml
    mon.ml = Mock()
    if ml_result is None:
        mon.ml.predict_anomaly = Mock(return_value={"is_anomaly": False, "anomaly_score": 0.0})
        mon.ml.predict_rul = Mock(return_value=None)
    else:
        mon.ml.predict_anomaly = Mock(return_value=ml_result)
        mon.ml.predict_rul = Mock(return_value=ml_result.get("rul_hours", None))

    return mon


def test_rule_only_alert_logs_and_returns(config, telemetry_collector):
    # rule triggers a high temp; provide a 60-sample snapshot with metadata to validate snapshot forwarding
    samples = []
    for i in range(60):
        samples.append({
            "cell_temps": [25, 75.0, 26],
            "cell_voltages": [3.7, 3.8, 3.75],
            "soh": 95,
            "metadata": {"vehicle_hash": "veh123", "trip_id": "trip1"},
        })

    rule_alert = {"rule_name": "max_cell_temp", "triggered": True, "severity": "HIGH", "reason": "Max cell temp 75.0C > 70.0C", "value": 75.0}

    mon = make_monitor_with_mocks(config, telemetry_collector, rule_alerts=[rule_alert])

    alerts = mon.process_telemetry(samples)

    assert any(a.get("source") == "rule" for a in alerts)
    # verify telemetry logged with snapshot (mock was called with telemetry_snapshot kw)
    telemetry_collector.log_battery_alert.assert_called()
    call_kwargs = telemetry_collector.log_battery_alert.call_args[1]
    assert "telemetry_snapshot" in call_kwargs
    tsnap = call_kwargs.get("telemetry_snapshot")
    assert isinstance(tsnap, list) and len(tsnap) == 60
    # verify rule alert content forwarded
    called_alert = telemetry_collector.log_battery_alert.call_args[0][0]
    assert called_alert["source"] == "rule"
    assert called_alert.get("rule") == "max_cell_temp"
    assert called_alert.get("value") == 75.0
    assert called_alert.get("anomaly_score") is None


def test_ml_only_alert_logs_and_returns(config, telemetry_collector):
    # ML flags anomaly
    sample = {"cell_temps": [25, 26, 26], "cell_voltages": [3.7, 3.71, 3.72], "soh": 95}
    ml_result = {"is_anomaly": True, "anomaly_score": 0.85, "rul_hours": 12}

    mon = make_monitor_with_mocks(config, telemetry_collector, rule_alerts=[], ml_result=ml_result)

    alerts = mon.process_telemetry([sample])

    assert any(a.get("source") == "ml" for a in alerts)
    telemetry_collector.log_battery_alert.assert_called()
    # find ml alert
    ml_alerts = [a for a in alerts if a.get("source") == "ml"]
    assert ml_alerts and ml_alerts[0]["anomaly_score"] == ml_result["anomaly_score"]
    assert ml_alerts[0]["rul_hours"] == 12


def test_combined_rule_and_ml_alerts(config, telemetry_collector):
    sample = {"cell_temps": [25, 80, 26], "cell_voltages": [3.7, 3.8, 3.75], "soh": 20}
    rule_alert = {"rule_name": "max_cell_temp", "triggered": True, "severity": "HIGH", "reason": "Max cell temp 80.0C > 70.0C", "value": 80}
    ml_result = {"is_anomaly": True, "anomaly_score": 0.9, "rul_hours": 1}

    mon = make_monitor_with_mocks(config, telemetry_collector, rule_alerts=[rule_alert], ml_result=ml_result)

    alerts = mon.process_telemetry([sample])

    assert any(a.get("source") == "rule" for a in alerts)
    assert any(a.get("source") == "ml" for a in alerts)
    # ensure two log calls (one for rule, one for ml)
    assert telemetry_collector.log_battery_alert.call_count >= 2


def test_no_alerts_returns_empty(config, telemetry_collector):
    sample = {"cell_temps": [25, 26, 26], "cell_voltages": [3.7, 3.71, 3.72], "soh": 95}
    mon = make_monitor_with_mocks(config, telemetry_collector, rule_alerts=[], ml_result={"is_anomaly": False, "anomaly_score": 0.1})

    alerts = mon.process_telemetry([sample])
    assert alerts == []
    telemetry_collector.log_battery_alert.assert_not_called()


def test_empty_samples_returns_no_alerts(config, telemetry_collector):
    mon = make_monitor_with_mocks(config, telemetry_collector, rule_alerts=[], ml_result={"is_anomaly": False, "anomaly_score": 0.0})
    alerts = mon.process_telemetry([])
    assert alerts == []


def test_ml_disabled_skips_ml(config, telemetry_collector):
    cfg = DummyConfig({"battery_monitor": {"ml": {"enable": False}}})
    mon = make_monitor_with_mocks(cfg, telemetry_collector, rule_alerts=[], ml_result={"is_anomaly": True, "anomaly_score": 0.9})

    alerts = mon.process_telemetry([{"cell_temps": [25], "cell_voltages": [3.7], "soh": 98}])
    # ML is disabled so no ml alerts
    assert not any(a.get("source") == "ml" for a in alerts)
    # no logs if no rule alerts
    telemetry_collector.log_battery_alert.assert_not_called()


def test_missing_fields_skips_rules():
    # Real RuleEngine should skip checks when fields are missing
    from battery_monitor.rules.rule_engine import RuleEngine

    cfg = DummyConfig({"battery_monitor": {"rules": {"enable_soh_tracking": False}}})
    engine = RuleEngine(cfg)
    # telemetry lacks cell_temps and cell_voltages
    sample = {"soh": 99}
    alerts = engine.evaluate(sample)
    assert alerts == []
