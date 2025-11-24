import sqlite3
import threading
import time
from typing import Optional, List, Dict


class AnnotationsDB:
    """Simple SQLite-backed annotations database with thread-safety.

    Provides methods to create and query annotations and incidents.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY,
                    incident_id TEXT,
                    label TEXT,
                    notes TEXT,
                    reviewer_id TEXT,
                    severity_override TEXT,
                    created_at INTEGER,
                    updated_at INTEGER,
                    deleted INTEGER DEFAULT 0
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    trip_id TEXT,
                    timestamp INTEGER,
                    risk_score REAL,
                    alert_level TEXT,
                    has_video INTEGER,
                    frame_count INTEGER,
                    sensor_window_path TEXT,
                    explanation_path TEXT
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_annotations_incident ON annotations(incident_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_annotations_created ON annotations(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_annotations_label ON annotations(label)")
            self._conn.commit()

    def create_annotation(self, incident_id: str, label: str, notes: str, reviewer_id: str, severity_override: Optional[str] = None) -> int:
        ts = int(time.time())
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO annotations (incident_id,label,notes,reviewer_id,severity_override,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (incident_id, label, notes, reviewer_id, severity_override, ts, ts),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_annotations(self, filters: Dict = None) -> List[Dict]:
        filters = filters or {}
        where = []
        params = []
        if "incident_id" in filters:
            where.append("incident_id = ?")
            params.append(filters["incident_id"])
        if "label" in filters:
            where.append("label = ?")
            params.append(filters["label"])
        if filters.get("exclude_deleted", True):
            where.append("deleted = 0")
        q = "SELECT * FROM annotations"
        if where:
            q += " WHERE " + " AND ".join(where)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(q, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def update_annotation(self, annotation_id: int, updates: Dict) -> bool:
        if not updates:
            # No update fields provided; nothing to do.
            return False
        sets = []
        params = []
        for k, v in updates.items():
            sets.append(f"{k} = ?")
            params.append(v)
        # Always update the updated_at timestamp
        params.append(int(time.time()))
        params.append(annotation_id)
        q = f"UPDATE annotations SET {', '.join(sets)}, updated_at = ? WHERE id = ?"
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(q, params)
            self._conn.commit()
            return cur.rowcount > 0

    def delete_annotation(self, annotation_id: int) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("UPDATE annotations SET deleted = 1, updated_at = ? WHERE id = ?", (int(time.time()), annotation_id))
            self._conn.commit()
            return cur.rowcount > 0

    def get_annotation_by_id(self, annotation_id: int) -> Optional[Dict]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def create_incident(self, incident_id: str, metadata: Dict) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO incidents (incident_id,trip_id,timestamp,risk_score,alert_level,has_video,frame_count,sensor_window_path,explanation_path) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    incident_id,
                    metadata.get("trip_id"),
                    metadata.get("timestamp"),
                    metadata.get("risk_score"),
                    metadata.get("alert_level"),
                    int(bool(metadata.get("has_video", False))),
                    int(metadata.get("frame_count", 0)),
                    metadata.get("sensor_window_path"),
                    metadata.get("explanation_path"),
                ),
            )
            self._conn.commit()
            return True

    def get_incident(self, incident_id: str) -> Optional[Dict]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_incidents(self, filters: Dict = None) -> List[Dict]:
        filters = filters or {}
        where = []
        params = []
        if "has_video" in filters:
            where.append("has_video = ?")
            params.append(int(bool(filters["has_video"])))
        if "alert_level" in filters:
            where.append("alert_level = ?")
            params.append(filters["alert_level"])
        q = "SELECT * FROM incidents"
        if where:
            q += " WHERE " + " AND ".join(where)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(q, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_annotation_stats(self) -> Dict:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM annotations WHERE deleted = 0")
            total = cur.fetchone()[0]
            cur.execute("SELECT label, COUNT(*) as cnt FROM annotations WHERE deleted = 0 GROUP BY label")
            by_label = {r[0]: r[1] for r in cur.fetchall()}
            cur.execute("SELECT COUNT(DISTINCT reviewer_id) FROM annotations WHERE deleted = 0")
            reviewers = cur.fetchone()[0]
        return {"total": total, "by_label": by_label, "unique_reviewers": reviewers}

    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
