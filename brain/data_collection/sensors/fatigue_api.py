"""Flask-based REST API exposing fatigue status and recent events.

Provides a small factory `create_app(config)` and a thread-safe in-memory
`FatigueAPIState` used by the `DriverFatigueHandler` to update latest state.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, abort

logger = logging.getLogger(__name__)


class FatigueAPIState:
    """Singleton-like in-memory cache for latest fatigue status and events."""

    def __init__(self, max_events: int = 100):
        self._lock = threading.Lock()
        self._last_score: Optional[float] = None
        self._last_state: Optional[str] = None
        self._last_event: Optional[Dict[str, Any]] = None
        self._events: deque = deque(maxlen=int(max_events))

    def update_status(self, score: float, state: str, event: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            try:
                self._last_score = float(score)
            except Exception:
                self._last_score = None
            self._last_state = state
            if event:
                self._events.append(event)
                self._last_event = event

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "fatigue_score": self._last_score,
                "state": self._last_state,
                "last_event": self._last_event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._events)
            if limit is not None:
                return items[-int(limit) :]
            return items


_GLOBAL_STATE: Optional[FatigueAPIState] = None


def get_global_state(max_events: int = 100) -> FatigueAPIState:
    global _GLOBAL_STATE
    if _GLOBAL_STATE is None:
        _GLOBAL_STATE = FatigueAPIState(max_events=max_events)
    return _GLOBAL_STATE


def create_app(config: Any = None) -> Flask:
    app = Flask(__name__)
    cfg = config or {}
    sect = cfg.get("sensors", {}).get("driver_fatigue", {}) if isinstance(cfg, dict) else {}
    api_cfg = sect.get("fatigue_api", {}) if isinstance(sect, dict) else {}
    max_events = int(api_cfg.get("max_events_cache", 100))
    app.state = get_global_state(max_events=max_events)

    @app.route("/status", methods=["GET"])
    def status():
        try:
            return jsonify(app.state.get_status())
        except Exception:
            logger.exception("Failed to get status")
            abort(500)

    @app.route("/events", methods=["GET"])
    def events():
        limit = request.args.get("limit", None)
        try:
            if limit is not None:
                limit = int(limit)
            evs = app.state.get_events(limit=limit if limit is not None else 10)
            return jsonify({"events": evs})
        except Exception:
            logger.exception("Failed to get events")
            abort(500)

    @app.route("/query", methods=["POST"])
    def query():
        try:
            payload = request.get_json(force=True)
            if not payload or "check" not in payload:
                abort(400)
            check = payload.get("check")
            status = app.state.get_status()
            if check == "drowsy":
                return jsonify({"result": status.get("state") in ("DROWSY", "MICROSLEEP")})
            if check == "microsleep":
                return jsonify({"result": status.get("state") == "MICROSLEEP"})
            if check == "any_alert":
                return jsonify({"result": status.get("state") in ("DROWSY", "MICROSLEEP")})
            abort(400)
        except Exception:
            logger.exception("Invalid query payload")
            abort(400)

    return app


if __name__ == "__main__":
    # minimal standalone server for manual testing
    try:
        import yaml

        cfg = {}
        try:
            with open("config/default_config.yaml", "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
        except Exception:
            cfg = {}
        app = create_app(cfg)
        port = int(cfg.get("sensors", {}).get("driver_fatigue", {}).get("fatigue_api", {}).get("port", 5001))
        host = cfg.get("sensors", {}).get("driver_fatigue", {}).get("fatigue_api", {}).get("host", "0.0.0.0")
        app.run(host=host, port=port)
    except Exception:
        logger.exception("Failed to run fatigue API standalone")
