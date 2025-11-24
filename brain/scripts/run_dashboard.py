#!/usr/bin/env python
"""Run the dashboard with injected local dependencies (simple runner).

This script loads config and starts the Flask dashboard.
"""
import argparse
import yaml
import logging
import threading
import time
import requests

from dashboard import create_app
try:
    from utils.telemetry import TelemetryCollector
except Exception:
    TelemetryCollector = None

try:
    from inference.explanation_logger import ExplanationLogger
except Exception:
    ExplanationLogger = None

logger = logging.getLogger(__name__)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def poll_edge_status(state, interval_s=1.0, status_url="http://localhost:5000/status", annotations_db=None):
    while True:
        try:
            r = requests.get(status_url, timeout=1.0)
            if r.status_code == 200:
                data = r.json()
                pred = data.get('latest_prediction') or data
                state.update_latest_prediction(pred)
                # if high-risk, create incident record if not exists
                try:
                    alert_level = (pred.get('alert_level') or '').lower()
                    if alert_level in ('high', 'critical') and annotations_db is not None:
                        incident_id = pred.get('prediction_id') or f"{pred.get('trip_id','')}_{pred.get('timestamp','') }"
                        existing = annotations_db.get_incident(incident_id)
                        if not existing:
                            metadata = {
                                'trip_id': pred.get('trip_id'),
                                'timestamp': pred.get('timestamp'),
                                'risk_score': pred.get('risk_score'),
                                'alert_level': pred.get('alert_level'),
                                'has_video': pred.get('has_video', False),
                                'frame_count': pred.get('frame_count', 0),
                                'sensor_window_path': pred.get('sensor_window_path'),
                                'explanation_path': pred.get('explanation_path'),
                            }
                            annotations_db.create_incident(incident_id, metadata)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(interval_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_config.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config) or {}
    dashboard_cfg = cfg.get("dashboard", {})
    host = args.host or dashboard_cfg.get("host", "0.0.0.0")
    port = args.port or dashboard_cfg.get("port", 5004)
    # instantiate collectors if available
    tc = TelemetryCollector(dashboard_cfg.get('telemetry_dir')) if TelemetryCollector else None
    el = ExplanationLogger(dashboard_cfg.get('explanation_dir')) if ExplanationLogger else None
    app = create_app(dashboard_cfg, telemetry_collector=tc, explanation_logger=el)
    # background poller for EdgeInferenceEngine status
    t = threading.Thread(target=poll_edge_status, args=(app.dashboard_state, 1.0, dashboard_cfg.get("edge_status_url", "http://localhost:5000/status"), app.annotations_db), daemon=True)
    t.start()
    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
