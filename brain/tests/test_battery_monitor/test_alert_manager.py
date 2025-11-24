import json
import os
from tempfile import TemporaryDirectory

from battery_monitor.alerts.alert_manager import AlertManager


class DummyBLE:
    def __init__(self):
        self.broadcasts = []

    def broadcast(self, alert, vehicle_id=None):
        self.broadcasts.append((alert, vehicle_id))


class DummyEventLogger:
    def __init__(self):
        self.logged = []

    def log_alert(self, alert, telemetry_snapshot=None, features=None, scores=None):
        self.logged.append({"alert": alert, "features": features, "scores": scores})


def test_alert_manager_triggers_and_logs():
    cfg = {"battery_monitor": {"alerts": {"log_path": ""}}, "privacy": {"node_id_salt": "s"}}
    with TemporaryDirectory() as td:
        lp = os.path.join(td, "alerts.log")
        cfg["battery_monitor"]["alerts"]["log_path"] = lp
        ble = DummyBLE()
        el = DummyEventLogger()
        am = AlertManager(cfg, telemetry=None, ble_stub=ble, event_logger=el)
        # trigger a medium alert
        medium = {"severity": "MEDIUM", "reason": "temp_high", "anomaly_score": 0.6}
        am.trigger_alert(medium, telemetry_snapshot=[{"metadata": {"vehicle_hash": "v1"}}])
        # trigger critical -> should call ble and event_logger and write shutdown log
        critical = {"severity": "CRITICAL", "reason": "cell_failure", "anomaly_score": 0.99}
        am.trigger_alert(critical, telemetry_snapshot=[{"metadata": {"vehicle_hash": "v1"}}])
        # check file was written
        with open(lp, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        assert len(lines) >= 2
        # event logger recorded alerts
        assert len(el.logged) >= 2
        # BLE got at least one broadcast for critical
        assert any(b[0]["severity"] == "CRITICAL" for b in ble.broadcasts)
        am.close()
