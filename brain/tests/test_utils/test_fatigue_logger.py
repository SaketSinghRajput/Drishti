import sqlite3
import time
from pathlib import Path

from utils.fatigue_logger import FatigueLogger


def test_log_and_query(tmp_path):
    db = tmp_path / "fatigue.db"
    cfg = {"sensors": {"driver_fatigue": {"fatigue_logging": {"enable": True, "db_path": str(db), "retention_days": 30}}}}
    fl = FatigueLogger(cfg)
    sample = {"timestamp": time.time(), "drowsiness_score": 0.7, "fatigue_state": "DROWSY", "head_pose": {"pitch": 5.0}}
    fl.log_sample(sample, vehicle_hash="vh", trip_id="t1")
    rows = fl.get_recent_events(limit=5)
    assert len(rows) >= 1
    r = rows[0]
    assert r.get("state") == "DROWSY"
    fl.close()


def test_cleanup_old_events(tmp_path):
    db = tmp_path / "fatigue2.db"
    cfg = {"sensors": {"driver_fatigue": {"fatigue_logging": {"enable": True, "db_path": str(db), "retention_days": 0}}}}
    fl = FatigueLogger(cfg)
    # insert an old event by writing directly to DB
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    # timestamp is stored as INTEGER epoch in the new schema
    cur.execute("INSERT INTO fatigue_events (timestamp, ear, mar, head_pitch, score, state, event_flag, vehicle_hash, trip_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (0, None, None, None, 0.1, "NORMAL", 0, "vh", "t1"))
    conn.commit()
    conn.close()
    # cleanup should remove old event
    fl.cleanup_old_events()
    rows = fl.get_recent_events(limit=10)
    assert len(rows) == 0
    fl.close()


def test_log_high_scores_to_telemetry(tmp_path, monkeypatch):
    db = tmp_path / "fatigue3.db"
    # enable telemetry logging for high scores and set threshold low
    cfg = {"sensors": {"driver_fatigue": {"fatigue_logging": {"enable": True, "db_path": str(db), "retention_days": 30, "log_high_scores_to_telemetry": True, "high_score_threshold": 0.5}}}}
    calls = []

    class DummyTelemetry:
        def log_prediction(self, payload):
            calls.append(payload)

    dt = DummyTelemetry()
    fl = FatigueLogger(cfg, telemetry_collector=dt)
    sample = {"timestamp": time.time(), "drowsiness_score": 0.7, "fatigue_state": "DROWSY", "head_pose": {"pitch": 5.0}}
    fl.log_sample(sample, vehicle_hash="vh", trip_id="t1")
    # ensure telemetry received a prediction for high score
    assert len(calls) >= 1
    assert calls[0].get("score") >= 0.7
    fl.close()
