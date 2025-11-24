import os
import sqlite3
from tempfile import TemporaryDirectory

from battery_monitor.storage.event_logger import EventLogger


def test_event_logger_snapshot_and_alert_storage():
    cfg = {"battery_monitor": {"storage": {"db_path": ""}}}
    with TemporaryDirectory() as td:
        db = os.path.join(td, "events.db")
        cfg["battery_monitor"]["storage"]["db_path"] = db
        el = EventLogger(cfg)
        alert = {"severity": "HIGH", "reason": "voltage_diff"}
        telemetry = [{"metadata": {"vehicle_hash": "v1"}, "cells": [3.7, 3.8]}]
        features = {"feature1": 1.2}
        scores = {"anomaly_score": 0.9}
        aid, sid = el.log_alert(alert, telemetry_snapshot=telemetry, features=features, scores=scores)
        assert aid is not None
        assert sid is not None
        recent = el.get_recent_alerts(limit=10)
        assert any(r["id"] == aid for r in recent)
        snap = el.get_snapshot(sid)
        assert snap is not None
        assert "snapshot" in snap and isinstance(snap["snapshot"], list)
        assert snap["features"]["feature1"] == 1.2
        assert snap["scores"]["anomaly_score"] == 0.9
        el.close()
