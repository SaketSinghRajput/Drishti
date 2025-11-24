"""Example demonstrating the fatigue integration flow.

This script runs a minimal demonstration: instantiate the handler (simulation
mode), trigger a DROWSY-like sample via monkeyed detector values (or call read
until state changes), inspect BLE outbox, query the local API server, and
inspect SQLite logs. This is intended to be run locally for manual testing.
"""
from __future__ import annotations

import threading
import time
import json
import requests

from config.config_loader import ConfigLoader
from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
from data_collection.sensors.fatigue_api import create_app


def run_api_in_thread(cfg):
    app = create_app(cfg)
    t = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001), daemon=True)
    t.start()
    return app, t


def main():
    # load config
    cfg = {}
    try:
        cfg = ConfigLoader().config
    except Exception:
        # fallback: minimal config
        cfg = {}

    # start API server in background
    app, t = run_api_in_thread(cfg)

    # create handler in simulation mode
    handler = DriverFatigueHandler(cfg, use_cv=False)

    # repeatedly read until a non-NORMAL state appears or for a few iterations
    for i in range(5):
        s = handler.read()
        print('Sample:', s)
        time.sleep(0.2)

    # query API
    try:
        r = requests.get('http://localhost:5001/status', timeout=2)
        print('API status:', r.json())
    except Exception as e:
        print('Failed to query API:', e)

    # read BLE outbox if present
    try:
        with open('data/ble_fatigue_outbox.json', 'r', encoding='utf-8') as fh:
            lines = fh.read().strip().splitlines()
            print('BLE outbox lines:', len(lines))
            if lines:
                print('Last packet:', json.loads(lines[-1]))
    except Exception:
        print('No BLE outbox present')

    # cleanup
    handler.close()


if __name__ == '__main__':
    main()
