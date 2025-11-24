"""EventLogger: persist alerts and telemetry snapshots to SQLite.

This module provides a small, dependency-free storage layer using sqlite3.
It stores alerts and periodic telemetry snapshots and exposes simple query helpers.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


class EventLogger:
    def __init__(self, config: Dict[str, Any]):
        storage_cfg = config.get("battery_monitor", {}).get("storage", {})
        self.db_path = str(storage_cfg.get("db_path", "data/events.db"))
        self.retention_days = int(storage_cfg.get("retention_days", 30))
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            # Snapshots now include features and scores so that a single
            # snapshot row holds telemetry, extraction features, and ML scores.
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS battery_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                features_json TEXT,
                scores_json TEXT
            )
            """,
            )
            # Alerts reference snapshot_id and store high-level alert metadata
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS battery_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                severity TEXT,
                reason TEXT,
                alert_json TEXT NOT NULL,
                snapshot_id INTEGER,
                FOREIGN KEY(snapshot_id) REFERENCES battery_snapshots(id)
            )
            """,
            )
            self._conn.commit()

    def log_alert(self, alert: Dict[str, Any], telemetry_snapshot: Optional[List[Dict[str, Any]]] = None, features: Optional[Dict[str, Any]] = None, scores: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
        ts = datetime.now(timezone.utc).isoformat()
        snapshot_id = None
        with self._lock:
            cur = self._conn.cursor()
            if telemetry_snapshot is not None or features is not None or scores is not None:
                snapshot_json = json.dumps(telemetry_snapshot) if telemetry_snapshot is not None else json.dumps([])
                features_json = json.dumps(features or {})
                scores_json = json.dumps(scores or {})
                cur.execute(
                    "INSERT INTO battery_snapshots (ts, snapshot_json, features_json, scores_json) VALUES (?, ?, ?, ?)",
                    (ts, snapshot_json, features_json, scores_json),
                )
                snapshot_id = cur.lastrowid

            alert_json = json.dumps(alert)
            cur.execute(
                "INSERT INTO battery_alerts (ts, severity, reason, alert_json, snapshot_id) VALUES (?, ?, ?, ?, ?)",
                (ts, alert.get("severity"), alert.get("reason"), alert_json, snapshot_id),
            )
            alert_id = cur.lastrowid
            self._conn.commit()

        # async-ish retention cleanup could run here; keep simple synchronous for now
        self._apply_retention()
        return alert_id, snapshot_id

    def _apply_retention(self) -> None:
        if self.retention_days <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        cutoff_ts = cutoff.isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM battery_alerts WHERE ts < ?", (cutoff_ts,))
            cur.execute("DELETE FROM battery_snapshots WHERE ts < ?", (cutoff_ts,))
            self._conn.commit()

    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, ts, severity, reason, alert_json, snapshot_id FROM battery_alerts ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({"id": r[0], "ts": r[1], "severity": r[2], "reason": r[3], "alert": json.loads(r[4]), "snapshot_id": r[5]})
        return out

    def get_snapshot(self, snapshot_id: int) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT snapshot_json, features_json, scores_json FROM battery_snapshots WHERE id = ?", (snapshot_id,))
            row = cur.fetchone()
        if not row:
            return None
        return {"snapshot": json.loads(row[0]), "features": json.loads(row[1]) if row[1] else {}, "scores": json.loads(row[2]) if row[2] else {}}

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.commit()
                self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        except Exception:
            pass
