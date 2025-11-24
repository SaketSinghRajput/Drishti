import os
import json
import logging
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from typing import Optional

from .state import DashboardState
from .annotations_db import AnnotationsDB
from .analytics import AnalyticsEngine

logger = logging.getLogger(__name__)


def create_app(config: dict = None, telemetry_collector=None, explanation_logger=None, video_retention_manager=None):
    """Create Flask app for dashboard.

    Dependencies are injected for testability.
    """
    cfg = config or {}
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"), static_folder=os.path.join(os.path.dirname(__file__), "static"))

    if cfg.get("enable_cors"):
        CORS(app)

    # initialize shared components
    state = DashboardState(max_alerts=cfg.get("max_cache_size", 200))
    annotations_db = AnnotationsDB(cfg.get("annotations_db_path", "data/annotations.db"))
    analytics = AnalyticsEngine(
        cfg.get("telemetry_dir", "data/telemetry"),
        cfg.get("explanation_dir", "data/explanations"),
        cache_ttl_s=cfg.get("analytics_cache_ttl_s", 300),
        telemetry_collector=telemetry_collector,
        explanation_logger=explanation_logger,
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "stats": state.get_stats()})

    @app.route("/api/realtime/latest")
    def realtime_latest():
        latest = state.get_latest_prediction()
        return jsonify(latest or {})

    @app.route("/api/realtime/sensor_health")
    def sensor_health():
        return jsonify(state.get_sensor_health())

    @app.route('/api/hardware_health', methods=['POST'])
    def hardware_health():
        """Receive hardware health status from HardwareChecker and update dashboard state."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON body"}), 400
            # Validate minimal schema
            if 'overall_status' not in data or 'components' not in data:
                return jsonify({"error": "Invalid payload; missing overall_status or components"}), 400
            sensor_health = data.get('components', {})
            if not isinstance(sensor_health, dict):
                return jsonify({"error": "Invalid components format"}), 400
            state.update_sensor_health(sensor_health)
            logger.info('Hardware health updated: %s', data.get('overall_status'))
            return jsonify({"status": "ok"}), 200
        except Exception as e:
            logger.exception('Failed to process hardware health')
            return jsonify({"error": str(e)}), 500

    @app.route("/api/alerts")
    def alerts():
        limit = int(request.args.get("limit", 100))
        return jsonify(state.get_alert_history(limit=limit))

    @app.route("/api/incidents")
    def list_incidents():
        incidents = annotations_db.get_incidents()
        return jsonify(incidents)

    @app.route("/api/incidents/<incident_id>")
    def incident_detail(incident_id):
        inc = annotations_db.get_incident(incident_id)
        if not inc:
            return jsonify({"error": "not found"}), 404
        # attach annotations
        anns = annotations_db.get_annotations({"incident_id": incident_id})
        inc["annotations"] = anns
        return jsonify(inc)

    @app.route("/api/incidents/<incident_id>/frames")
    def incident_frames(incident_id):
        inc = annotations_db.get_incident(incident_id)
        if not inc:
            return jsonify({"frames": []})
        # Query video manager if provided
        frames = []
        if video_retention_manager:
            try:
                frames = video_retention_manager.get_frames_for_incident(incident_id)
            except Exception:
                frames = []
        return jsonify({"frames": frames})

    @app.route("/api/incidents/<incident_id>/timeline")
    def incident_timeline(incident_id):
        inc = annotations_db.get_incident(incident_id)
        if not inc:
            return jsonify({}), 404
        # Best-effort: read sensor window file if path available
        sensor_path = inc.get("sensor_window_path")
        timeline = {}
        if sensor_path and os.path.exists(sensor_path):
            try:
                with open(sensor_path, "r", encoding="utf-8") as fh:
                    timeline = json.load(fh)
            except Exception:
                timeline = {}
        return jsonify(timeline)

    @app.route("/api/annotations", methods=["POST", "GET"])
    def annotations():
        if request.method == "POST":
            body = request.get_json(force=True)
            incident_id = body.get("incident_id")
            label = body.get("label")
            notes = body.get("notes", "")
            reviewer = body.get("reviewer_id", cfg.get("default_reviewer_id", "anonymous"))
            severity = body.get("severity_override")
            if not incident_id or not label:
                return jsonify({"error": "invalid"}), 400
            aid = annotations_db.create_annotation(incident_id, label, notes, reviewer, severity)
            return jsonify({"annotation_id": aid}), 201
        else:
            filters = request.args.to_dict()
            anns = annotations_db.get_annotations(filters)
            return jsonify(anns)

    @app.route("/api/annotations/<int:annotation_id>", methods=["PUT", "DELETE"])
    def annotation_item(annotation_id):
        if request.method == "PUT":
            updates = request.get_json(force=True)
            ok = annotations_db.update_annotation(annotation_id, updates)
            return jsonify({"ok": ok})
        else:
            ok = annotations_db.delete_annotation(annotation_id)
            return jsonify({"ok": ok})

    @app.route("/api/analytics/false_alarm_rate")
    def analytics_false_alarm():
        start = request.args.get('start_date')
        end = request.args.get('end_date')
        res = analytics.compute_false_alarm_rate(start, end, annotations_db)
        return jsonify(res)

    @app.route("/api/analytics/detection_latency")
    def analytics_latency():
        start = request.args.get('start_date')
        end = request.args.get('end_date')
        res = analytics.compute_detection_latency_stats(start, end)
        return jsonify(res)

    @app.route("/api/analytics/feature_importance")
    def analytics_features():
        start = request.args.get('start_date')
        end = request.args.get('end_date')
        res = analytics.compute_feature_importance_trends(start, end)
        return jsonify(res)

    @app.route("/api/analytics/alert_distribution")
    def analytics_alert_distribution():
        start = request.args.get('start_date')
        end = request.args.get('end_date')
        res = analytics.compute_alert_distribution(start, end)
        return jsonify(res)

    @app.route("/api/analytics/sensor_reliability")
    def analytics_sensor_reliability():
        start = request.args.get('start_date')
        end = request.args.get('end_date')
        res = analytics.compute_sensor_reliability(start, end)
        return jsonify(res)

    @app.route("/api/analytics/model_performance_summary")
    def analytics_model_performance():
        start = request.args.get('start_date')
        end = request.args.get('end_date')
        res = analytics.compute_model_performance_summary(annotations_db, start, end)
        return jsonify(res)

    @app.route("/api/analytics/export")
    def analytics_export():
        fmt = request.args.get('format', 'json')
        metric = request.args.get('metric', 'false_alarm_rate')
        # support a few metrics
        if metric == 'false_alarm_rate':
            data = analytics.compute_false_alarm_rate(None, None, annotations_db)
        elif metric == 'feature_importance':
            data = analytics.compute_feature_importance_trends(None, None)
        else:
            data = {}
        if fmt == 'csv':
            # very small CSV export implementation
            import io, csv

            buf = io.StringIO()
            if isinstance(data, dict):
                w = csv.writer(buf)
                for k, v in data.items():
                    w.writerow([k, v])
            else:
                buf.write(str(data))
            return buf.getvalue(), 200, {'Content-Type': 'text/csv'}
        return jsonify(data)

    # static assets served by Flask's static folder; provide convenience route
    @app.route('/static/<path:filename>')
    def static_files(filename):
        return send_from_directory(app.static_folder, filename)

    # attach objects for tests and runtime introspection
    app.dashboard_state = state
    app.annotations_db = annotations_db
    app.analytics_engine = analytics

    return app
