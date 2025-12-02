"""FatigueLogger: lightweight SQLite persistence for per-sample fatigue events.

Provides thread-safe inserts, simple retention cleanup, CSV export, and
optional telemetry integration for high-score samples.
"""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FatigueLogger:
    def __init__(self, config: Any, telemetry_collector: Optional[Any] = None):
        sect = config.get("sensors", {}).get("driver_fatigue", {}) if isinstance(config, dict) else {}
        fcfg = sect.get("fatigue_logging", {}) if isinstance(sect, dict) else {}
        self.enable = bool(fcfg.get("enable", True))
        self.db_path = str(fcfg.get("db_path", "data/fatigue_events.db"))
        self.retention_days = int(fcfg.get("retention_days", 30))
        self.enable_csv_export = bool(fcfg.get("enable_csv_export", False))
        self.csv_export_path = str(fcfg.get("csv_export_path", "data/fatigue_events.csv"))
        self.high_score_threshold = float(fcfg.get("high_score_threshold", 0.6))
        self.log_high_scores_to_telemetry = bool(fcfg.get("log_high_scores_to_telemetry", False))

        self._lock = threading.Lock()
        self._conn = None
        self._telemetry = telemetry_collector
        if not self.enable:
            return
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        except Exception:
            pass
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._ensure_table()
        except Exception:
            logger.exception("Failed to open fatigue SQLite DB: %s", self.db_path)
            self._conn = None

    def _ensure_table(self) -> None:
        if self._conn is None:
            return
        try:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS fatigue_events (
                    id INTEGER PRIMARY KEY,
                    timestamp INTEGER,
                    ear REAL,
                    mar REAL,
                    head_pitch REAL,
                    score REAL,
                    state TEXT,
                    event_flag INTEGER,
                    vehicle_hash TEXT,
                    trip_id TEXT
                )
                """
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to ensure fatigue_events table")

    def log_sample(self, sample: Dict[str, Any], vehicle_hash: str = "", trip_id: str = "") -> None:
        if not self.enable or self._conn is None:
            return
        try:
            # normalize timestamp into integer epoch seconds
            raw_ts = sample.get("timestamp", None)
                if raw_ts is None:
                ts_epoch = int(datetime.now(timezone.utc).timestamp())
            else:
                # accept int/float epoch or ISO string
                try:
                    if isinstance(raw_ts, (int, float)):
                        ts_epoch = int(raw_ts)
                    elif isinstance(raw_ts, str):
                        # attempt ISO parse
                        try:
                            ts_epoch = int(datetime.fromisoformat(raw_ts).timestamp())
                        except Exception:
                            # fallback: parse as float string
                            ts_epoch = int(float(raw_ts))
                        else:
                        ts_epoch = int(datetime.now(timezone.utc).timestamp())
                except Exception:
                    ts_epoch = int(datetime.now(timezone.utc).timestamp())
            score = float(sample.get("drowsiness_score", 0.0) or 0.0)
            state = str(sample.get("fatigue_state", "NORMAL"))
            hp = sample.get("head_pose") or {}
            head_pitch = float(hp.get("pitch", 0.0) or 0.0)
            ear = None
            mar = None
            # Some samples may include ear/mar; attempt to read if present
            try:
                ear = float(sample.get("ear", None)) if sample.get("ear", None) is not None else None
            except Exception:
                ear = None
            try:
                mar = float(sample.get("mar", None)) if sample.get("mar", None) is not None else None
            except Exception:
                mar = None

            event_flag = 1 if state in ("DROWSY", "MICROSLEEP") else 0
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO fatigue_events (timestamp, ear, mar, head_pitch, score, state, event_flag, vehicle_hash, trip_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ts_epoch, ear, mar, head_pitch, score, state, event_flag, vehicle_hash, trip_id),
                )
                self._conn.commit()

            # telemetry hook for high scores
            try:
                if self.log_high_scores_to_telemetry and self._telemetry and score >= self.high_score_threshold:
                    try:
                        self._telemetry.log_prediction({"score": score, "state": state})
                    except Exception:
                        logger.exception("Telemetry collector failed to log prediction")
            except Exception:
                logger.exception("Telemetry hook failed")
        except Exception:
            logger.exception("Failed to log fatigue sample to DB")

    def get_recent_events(self, limit: int = 10, state_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.enable or self._conn is None:
            return []
        try:
            with self._lock:
                cur = self._conn.cursor()
                q = "SELECT timestamp, ear, mar, head_pitch, score, state, event_flag, vehicle_hash, trip_id FROM fatigue_events"
                params = []
                if state_filter:
                    q += " WHERE state = ?"
                    params.append(state_filter)
                q += " ORDER BY id DESC LIMIT ?"
                params.append(int(limit))
                cur.execute(q, params)
                rows = cur.fetchall()
                out = []
                for r in rows:
                    out.append({
                        # return timestamp as epoch integer
                        "timestamp": int(r[0]) if r[0] is not None else None,
                        "ear": r[1],
                        "mar": r[2],
                        "head_pitch": r[3],
                        "score": r[4],
                        "state": r[5],
                        "event_flag": int(r[6] or 0),
                        "vehicle_hash": r[7],
                        "trip_id": r[8],
                    })
                return out
        except Exception:
            logger.exception("Failed to read recent events from DB")
            return []

    def export_to_csv(self) -> None:
        if not self.enable or self._conn is None or not self.enable_csv_export:
            return
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute("SELECT timestamp, ear, mar, head_pitch, score, state, event_flag, vehicle_hash, trip_id FROM fatigue_events ORDER BY id ASC")
                rows = cur.fetchall()
            os.makedirs(os.path.dirname(self.csv_export_path) or ".", exist_ok=True)
            with open(self.csv_export_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["timestamp", "ear", "mar", "head_pitch", "score", "state", "event_flag", "vehicle_hash", "trip_id"])
                for r in rows:
                    writer.writerow(list(r))
        except Exception:
            logger.exception("Failed to export fatigue events to CSV")

    def cleanup_old_events(self) -> None:
        if not self.enable or self._conn is None:
            return
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=int(self.retention_days))
            cutoff_epoch = int(cutoff.timestamp())
            with self._lock:
                cur = self._conn.cursor()
                cur.execute("DELETE FROM fatigue_events WHERE timestamp < ?", (cutoff_epoch,))
                self._conn.commit()
        except Exception:
            logger.exception("Failed to cleanup old fatigue events")

    def close(self) -> None:
        try:
            if self._conn is not None:
                try:
                    self._conn.commit()
                except Exception:
                    pass
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        except Exception:
            pass
