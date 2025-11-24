"""Simple demo that generates synthetic battery telemetry and runs BMSMonitor.

Run: python examples/battery_alert_demo.py
"""
from battery_monitor.rules.bms_monitor import BMSMonitor


class SimpleConfig:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        parts = key.split(".")
        cur = self._data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur


class DummyTelemetry:
    def log_battery_alert(self, alert, telemetry_snapshot=None):
        print("Telemetry logged:", alert)


def run_demo():
    cfg = SimpleConfig({
        "battery_monitor": {"alerts": {}, "storage": {}, "ml": {"enable": True}},
        "privacy": {"node_id_salt": "demo"},
    })
    tel = DummyTelemetry()
    monitor = BMSMonitor(cfg, tel)
    # inject simple fakes
    monitor.rule_engine.evaluate = lambda latest: [{"severity": "HIGH", "reason": "voltage_drop", "rule_name": "v1", "value": 3.1}]
    monitor.extractor.extract_features = lambda samples: {"avg_v": 3.1}
    monitor.ml.predict_anomaly = lambda feats: {"is_anomaly": True, "anomaly_score": 0.88}
    monitor.ml.predict_rul = lambda feats: 24
    samples = [{"metadata": {"vehicle_hash": "demo_v"}, "cells": [3.1, 3.0, 3.2]}]
    alerts = monitor.process_telemetry(samples)
    print("Generated alerts:", alerts)
    monitor.stop()


if __name__ == "__main__":
    run_demo()
