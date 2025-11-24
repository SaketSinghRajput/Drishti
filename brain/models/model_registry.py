import json
import sqlite3
import threading
import time
from typing import Optional, Dict, Any, List

from datetime import datetime


class ModelRegistry:
    def __init__(self, db_path: str = 'data/model_registry.db'):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_tables(self):
        with self._conn() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS models(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT,
                horizon INTEGER,
                created_at TEXT,
                trained_by TEXT,
                dataset_hash TEXT,
                config_snapshot TEXT,
                model_path TEXT,
                onnx_path TEXT,
                status TEXT
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS metrics(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                metric_name TEXT,
                metric_value REAL,
                split TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS deployments(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                edge_device_id TEXT,
                deployed_at TEXT,
                status TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id)
            )''')

    def _next_version(self) -> str:
        # simplistic incremental patch versioning v1.0.N
        with self._conn() as c:
            c.execute("SELECT version FROM models ORDER BY id DESC LIMIT 1")
            row = c.fetchone()
            if not row or not row[0]:
                return 'v1.0.0'
            v = row[0].lstrip('v')
            parts = v.split('.')
            try:
                parts[-1] = str(int(parts[-1]) + 1)
                return 'v' + '.'.join(parts)
            except Exception:
                return row[0] + '.1'

    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute('SELECT id,version,horizon,created_at,trained_by,dataset_hash,config_snapshot,model_path,onnx_path,status FROM models WHERE id=?', (model_id,)).fetchone()
            if not r:
                return None
            return dict(zip(['id','version','horizon','created_at','trained_by','dataset_hash','config_snapshot','model_path','onnx_path','status'], r))

    def get_model_by_version(self, version: str, horizon: Optional[int] = None) -> Optional[Dict[str, Any]]:
        q = 'SELECT id,version,horizon,created_at,trained_by,dataset_hash,config_snapshot,model_path,onnx_path,status FROM models WHERE version=?'
        params = [version]
        if horizon is not None:
            q += ' AND horizon=?'
            params.append(horizon)
        q += ' ORDER BY id DESC LIMIT 1'
        with self._conn() as c:
            r = c.execute(q, params).fetchone()
            if not r:
                return None
            return dict(zip(['id','version','horizon','created_at','trained_by','dataset_hash','config_snapshot','model_path','onnx_path','status'], r))

    def compare_models(self, model_id_a: int, model_id_b: int) -> Dict[str, Any]:
        ma = self.get_model_metrics(model_id_a)
        mb = self.get_model_metrics(model_id_b)
        # compute simple deltas for metrics present in both
        keys = set(ma.keys()).union(set(mb.keys()))
        deltas = {}
        for k in keys:
            a = ma.get(k)
            b = mb.get(k)
            if a is None or b is None:
                deltas[k] = {'a': a, 'b': b, 'delta': None}
            else:
                deltas[k] = {'a': a, 'b': b, 'delta': float(a - b)}
        return {'deltas': deltas}

    def archive_old_models(self, horizon: Optional[int] = None, keep_last_n: int = 5):
        # mark older models as archived per-horizon, keep last N
        with self._conn() as c:
            if horizon is None:
                rows = c.execute('SELECT id,horizon FROM models ORDER BY horizon, id DESC').fetchall()
                grouped = {}
                for r in rows:
                    grouped.setdefault(r[1], []).append(r[0])
                for h, ids in grouped.items():
                    to_archive = ids[keep_last_n:]
                    for mid in to_archive:
                        c.execute('UPDATE models SET status=? WHERE id=?', ('archived', mid))
            else:
                rows = c.execute('SELECT id FROM models WHERE horizon=? ORDER BY id DESC', (horizon,)).fetchall()
                ids = [r[0] for r in rows]
                to_archive = ids[keep_last_n:]
                for mid in to_archive:
                    c.execute('UPDATE models SET status=? WHERE id=?', ('archived', mid))

    def record_deployment(self, model_id: int, edge_device_id: str, status: str = 'ok'):
        with self._conn() as c:
            c.execute('INSERT INTO deployments(model_id,edge_device_id,deployed_at,status) VALUES(?,?,?,?)', (model_id, edge_device_id, datetime.utcnow().isoformat(), status))

    def register_model(self, horizon: int, model_path: str, onnx_path: Optional[str], config_snapshot: Dict[str, Any], dataset_hash: str, metrics: Dict[str, float], trained_by: str = '') -> int:
        version = self._next_version()
        with self._lock:
            with self._conn() as c:
                c.execute('''INSERT INTO models(version,horizon,created_at,trained_by,dataset_hash,config_snapshot,model_path,onnx_path,status) VALUES(?,?,?,?,?,?,?,?,?)''',
                          (version, horizon, datetime.utcnow().isoformat(), trained_by, dataset_hash, json.dumps(config_snapshot), model_path, onnx_path or '', 'validated'))
                model_id = c.lastrowid
                for k, v in (metrics or {}).items():
                    c.execute('INSERT INTO metrics(model_id,metric_name,metric_value,split) VALUES(?,?,?,?)', (model_id, k, float(v), 'val'))
        return model_id

    def get_latest_model(self, horizon: Optional[int] = None, status: str = 'deployed') -> Optional[Dict]:
        q = 'SELECT id,version,horizon,created_at,trained_by,dataset_hash,config_snapshot,model_path,onnx_path,status FROM models'
        params = []
        if horizon is not None:
            q += ' WHERE horizon=?'
            params.append(horizon)
        q += ' AND status=?' if params else ' WHERE status=?'
        params.append(status)
        q += ' ORDER BY id DESC LIMIT 1'
        with self._conn() as c:
            r = c.execute(q, params).fetchone()
            if not r:
                return None
            return dict(zip(['id','version','horizon','created_at','trained_by','dataset_hash','config_snapshot','model_path','onnx_path','status'], r))

    def update_model_status(self, model_id: int, status: str):
        with self._conn() as c:
            c.execute('UPDATE models SET status=? WHERE id=?', (status, model_id))

    def get_model_metrics(self, model_id: int) -> Dict[str, float]:
        with self._conn() as c:
            rows = c.execute('SELECT metric_name, metric_value FROM metrics WHERE model_id=?', (model_id,)).fetchall()
            return {r[0]: r[1] for r in rows}

    def list_models(self, horizon: Optional[int] = None, limit: int = 10) -> List[Dict]:
        q = 'SELECT id,version,horizon,created_at,model_path,status FROM models'
        params = []
        if horizon is not None:
            q += ' WHERE horizon=?'
            params.append(horizon)
        q += ' ORDER BY id DESC LIMIT ?'
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(q, params).fetchall()
            out = []
            for r in rows:
                out.append(dict(zip(['id','version','horizon','created_at','model_path','status'], r)))
            return out
