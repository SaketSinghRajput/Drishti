import os
from tempfile import TemporaryDirectory

from battery_monitor.rules.bms_monitor import BMSMonitor


class SimpleConfig:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        # simple dotted-key support
        parts = key.split(".")
        cur = self._data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur


def test_bms_monitor_triggers_end_to_end():
    # minimal config pointing storage and outbox to temp dir
    base = {
        "battery_monitor": {
            "alerts": {"log_path": "", "ble_outbox_path": ""},
            "storage": {"db_path": ""},
            "ml": {"enable": True},
        },
        "privacy": {"node_id_salt": "s"},
    }
    with TemporaryDirectory() as td:
        base["battery_monitor"]["alerts"]["log_path"] = os.path.join(td, "alerts.log")
        base["battery_monitor"]["alerts"]["ble_outbox_path"] = os.path.join(td, "ble.json")
        base["battery_monitor"]["storage"]["db_path"] = os.path.join(td, "events.db")
        cfg = SimpleConfig(base)
        # telemetry collector can be a simple object with log_battery_alert
        class DummyTel:
            def __init__(self):
                self.logged = []

            def log_battery_alert(self, alert, telemetry_snapshot=None):
                self.logged.append((alert, telemetry_snapshot))

        tel = DummyTel()
        monitor = BMSMonitor(cfg, tel)
        # replace internal rule engine and ml/feature extractor with simple fakes
        # make the rule critical so BLE broadcast is triggered by AlertManager
        monitor.rule_engine.evaluate = lambda latest: [{"severity": "CRITICAL", "reason": "vdrop", "rule_name": "r1", "value": 3.2}]
        monitor.extractor.extract_features = lambda samples: {"f1": 1.0}
        monitor.ml.predict_anomaly = lambda feats: {"is_anomaly": True, "anomaly_score": 0.9}
        monitor.ml.predict_rul = lambda feats: 12
        samples = [{"metadata": {"vehicle_hash": "v1"}, "cells": [3.7, 3.8]}]
        alerts = monitor.process_telemetry(samples)
        assert any(a["source"] == "rule" for a in alerts)
        assert any(a["source"] == "ml" for a in alerts)
        # check subsystems wrote files
        assert os.path.exists(base["battery_monitor"]["alerts"]["log_path"]) is True
        assert os.path.exists(base["battery_monitor"]["alerts"]["ble_outbox_path"]) is True
        monitor.stop()
