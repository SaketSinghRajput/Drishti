"""Run the fatigue REST API server.

Usage: python scripts/run_fatigue_api.py --port 5001
"""
from __future__ import annotations

import argparse
import logging
import yaml

from data_collection.sensors.fatigue_api import create_app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default_config.yaml")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args()

    cfg = {}
    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        cfg = {}

    host = args.host or cfg.get("sensors", {}).get("driver_fatigue", {}).get("fatigue_api", {}).get("host", "0.0.0.0")
    port = args.port or cfg.get("sensors", {}).get("driver_fatigue", {}).get("fatigue_api", {}).get("port", 5001)

    logging.basicConfig(level=logging.INFO)
    app = create_app(cfg)
    app.run(host=host, port=int(port))


if __name__ == "__main__":
    main()
