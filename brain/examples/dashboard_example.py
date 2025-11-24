"""Example script demonstrating the dashboard with mocked data."""
import time
import threading
import os
from dashboard import create_app


def populate_mock_state(app):
    for i in range(10):
        app.dashboard_state.update_latest_prediction({
            "timestamp": int(time.time()),
            "risk_score": i / 10.0,
            "alert_level": "low" if i < 3 else "medium" if i < 7 else "high",
            "reason": "simulated",
        })
        time.sleep(0.2)


def main():
    cfg = {"max_cache_size": 200, "annotations_db_path": "data/annotations.db", "telemetry_dir": "data/telemetry", "explanation_dir": "data/explanations"}
    app = create_app(cfg)
    t = threading.Thread(target=populate_mock_state, args=(app,), daemon=True)
    t.start()
    app.run(port=5004)


if __name__ == "__main__":
    main()
