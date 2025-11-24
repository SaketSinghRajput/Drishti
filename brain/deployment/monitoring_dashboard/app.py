import json
import time
from datetime import datetime

from flask import Flask, jsonify, render_template
try:
    import psutil
except Exception:
    psutil = None
import requests


def create_app(config=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/system_stats")
    def system_stats():
        if psutil is None:
            return jsonify({"error": "psutil not available"}), 503
        return jsonify(
            {
                "timestamp": int(time.time()),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": psutil.virtual_memory()._asdict(),
                "disk": psutil.disk_usage("/")._asdict(),
                "uptime_s": int(time.time() - psutil.boot_time()),
            }
        )

    @app.route("/api/inference_stats")
    def inference_stats():
        # Attempt to read local status endpoint from inference engine
        try:
            r = requests.get("http://127.0.0.1:5000/status", timeout=0.5)
            if r.ok:
                j = r.json()
                # map expected fields
                return jsonify(
                    {
                        "total_inferences": j.get("total_inferences", 0),
                        "latency_ms": j.get("latency_ms", {"mean": None, "p95": None, "p99": None}),
                        "last_updated": int(time.time()),
                    }
                )
        except Exception:
            pass
        # Fallback to empty structure if unavailable
        status = {"total_inferences": 0, "latency_ms": {"mean": None, "p95": None, "p99": None}, "last_updated": int(time.time())}
        return jsonify(status)

    @app.route("/api/sensor_health")
    def sensor_health():
        # Try to fetch health from local data logger or telemetry endpoint
        try:
            r = requests.get("http://127.0.0.1:5001/health", timeout=0.5)
            if r.ok:
                return jsonify(r.json())
        except Exception:
            pass
        # Fallback summary
        return jsonify({"camera": {"status": "unknown"}, "vehicle_can": {"status": "unknown"}, "ble": {"status": "unknown"}})

    @app.route("/api/alerts")
    def alerts():
        # Attempt to read recent alerts from local alerting endpoint or logs
        try:
            r = requests.get("http://127.0.0.1:5001/alerts", timeout=0.5)
            if r.ok:
                return jsonify(r.json())
        except Exception:
            pass
        return jsonify({"alerts": []})

    @app.route('/api/hardware_health', methods=['POST', 'GET'])
    def hardware_health():
        """Receive or retrieve hardware health status.

        POST: update cached health status from HardwareChecker.
        GET: return last cached health status.
        """
        if not hasattr(app, '_hardware_health_cache'):
            # Stable schema matching HardwareChecker payload
            app._hardware_health_cache = {
                "timestamp": None,
                "overall_status": "unknown",
                "components": {},
                "disabled_modules": [],
            }

        from flask import request
        if request.method == 'POST':
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON body"}), 400
                # Basic schema validation
                if not isinstance(data, dict) or 'overall_status' not in data or 'components' not in data:
                    return jsonify({"error": "Invalid health payload; missing required keys"}), 400
                # normalize keys we expect
                app._hardware_health_cache = {
                    'timestamp': data.get('timestamp'),
                    'overall_status': data.get('overall_status', 'unknown'),
                    'components': data.get('components', {}) or {},
                    'disabled_modules': data.get('disabled_modules', []) or [],
                }
                return jsonify({"status": "ok"}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify(app._hardware_health_cache)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5003)
