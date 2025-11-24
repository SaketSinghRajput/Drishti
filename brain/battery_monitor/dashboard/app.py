"""Flask dashboard for battery monitoring.

Provides REST APIs and an in-memory DashboardState for latest telemetry and metrics.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, render_template

logger = logging.getLogger(__name__)


class DashboardState:
    """Thread-safe in-memory cache for latest telemetry and recent alerts."""

    def __init__(self, max_history: int = 100):
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._history: List[Dict[str, Any]] = []
        self._max_history = max_history

    def update_latest(self, sample: Dict[str, Any]) -> None:
        with self._lock:
            self._latest = sample
            self._history.append(sample)
            if len(self._history) > self._max_history:
                self._history.pop(0)

    def get_latest_telemetry(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest.copy() if self._latest else None

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history[-limit:])

    def get_metrics_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                return {"avg_soc": None, "min_soh": None, "alert_counts": {}, "sample_count": 0}
            socs = [s.get("soc") for s in self._history if s.get("soc") is not None]
            sohs = [s.get("soh") for s in self._history if s.get("soh") is not None]
            return {
                "avg_soc": sum(socs) / len(socs) if socs else None,
                "min_soh": min(sohs) if sohs else None,
                "alert_counts": {},
                "sample_count": len(self._history),
                "time_range": {"start": self._history[0].get("timestamp"), "end": self._history[-1].get("timestamp")},
            }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(config: Optional[Dict[str, Any]] = None, event_logger=None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    cfg = config or {}
    dashboard_cfg = cfg.get("battery_monitor", {}).get("dashboard", {}) if isinstance(cfg, dict) else {}
    max_history = int(dashboard_cfg.get("max_history_samples", 100))
    app.dashboard_state = DashboardState(max_history=max_history)
    app.event_logger = event_logger

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/telemetry/latest", methods=["GET"])
    def get_latest():
        latest = app.dashboard_state.get_latest_telemetry()
        if latest is None:
            return jsonify({"latest": None})
        # compute derived metrics
        computed = {}
        cvs = latest.get("cell_voltages") or []
        cts = latest.get("cell_temps") or []
        if cvs:
            computed["voltage_diff"] = max(cvs) - min(cvs)
            computed["min_cell_voltage"] = min(cvs)
            computed["max_cell_voltage"] = max(cvs)
        if cts:
            computed["max_cell_temp"] = max(cts)
        latest["computed_metrics"] = computed
        return jsonify(latest)

    @app.route("/api/telemetry/history", methods=["GET"])
    def get_history():
        limit = int(request.args.get("limit", 100))
        # Return a flattened list of samples. Each entry has keys: alert_id, alert_ts, sample
        history: List[Dict[str, Any]] = []
        if app.event_logger is None:
            # fallback to in-memory history; wrap each sample with alert_id=None
            hist = app.dashboard_state.get_history(limit=limit)
            for s in hist:
                history.append({"alert_id": None, "alert_ts": None, "sample": s})
            return jsonify({"history": history})
        try:
            alerts = app.event_logger.get_recent_alerts(limit=limit)
            for a in alerts:
                sid = a.get("snapshot_id")
                if sid:
                    snap = app.event_logger.get_snapshot(sid)
                    if snap and isinstance(snap.get("snapshot"), list):
                        for sample in snap.get("snapshot"):
                            history.append({"alert_id": a.get("id"), "alert_ts": a.get("ts"), "sample": sample})
            return jsonify({"history": history})
        except Exception:
            logger.exception("Failed to query history from EventLogger")
            return jsonify({"history": []}), 500

    @app.route("/api/alerts", methods=["GET"])
    def get_alerts():
        limit = int(request.args.get("limit", 100))
        severity = request.args.get("severity")
        if app.event_logger is None:
            return jsonify({"alerts": []})
        alerts = app.event_logger.get_recent_alerts(limit=limit)
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity]
        return jsonify({"alerts": alerts})

    @app.route("/api/alerts/<int:alert_id>", methods=["GET"])
    def get_alert_detail(alert_id: int):
        if app.event_logger is None:
            return jsonify({"error": "no event logger"}), 404
        alerts = app.event_logger.get_recent_alerts(limit=500)
        matching = [a for a in alerts if a.get("id") == alert_id]
        if not matching:
            return jsonify({"error": "not found"}), 404
        a = matching[0]
        snapshot = None
        sid = a.get("snapshot_id")
        if sid:
            snapshot = app.event_logger.get_snapshot(sid)
        return jsonify({"alert": a, "snapshot": snapshot})

    @app.route("/api/metrics/summary", methods=["GET"])
    def metrics_summary():
        return jsonify(app.dashboard_state.get_metrics_summary())

    @app.route("/api/telemetry/update", methods=["POST"])
    def telemetry_update():
        data = request.get_json() or {}
        # ensure timestamp
        if "timestamp" not in data:
            data["timestamp"] = _iso_now()
        app.dashboard_state.update_latest(data)
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5002)
    app = create_app()
