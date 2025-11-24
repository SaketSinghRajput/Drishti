"""Flask-based dashboard package for battery monitoring.

Provides a `create_app()` factory in `app.py` to initialize the Flask
application and REST endpoints. Use `run_dashboard()` to run the app
from the command line or pass an `EventLogger` instance to `create_app()`
to enable historical data endpoints.
"""

from .app import create_app

__all__ = ["create_app", "run_dashboard"]


def run_dashboard(config=None, event_logger=None):
    app = create_app(config=config, event_logger=event_logger)
    cfg = config or {}
    dashboard_cfg = cfg.get("battery_monitor", {}).get("dashboard", {}) if isinstance(cfg, dict) else {}
    host = dashboard_cfg.get("host", "0.0.0.0")
    port = int(dashboard_cfg.get("port", 5002))
    app.run(host=host, port=port)
