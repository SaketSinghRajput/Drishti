#!/usr/bin/env python3
"""Main orchestrator for EV Safety Guardian - Multimodal AI Pipeline Brain.

This script initializes subsystems and runs a 1-second real-time loop that
collects sensor windows, runs inference, processes BMS telemetry, broadcasts a
BLE safety packet, updates the dashboard, triggers alerts, and logs telemetry.

The implementation is defensive: missing subsystems are stubbed so this script
can be used in dev/CI without all hardware present.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional

# Defensive imports for optional subsystems
try:
    from brain.config.config_loader import ConfigLoader
except Exception:
    # minimal stub
    class ConfigLoader:
        def __init__(self, path=None):
            self._cfg = {}

        def get(self, key, default=None):
            return default

        def set(self, key, value):
            return None

try:
    from brain.utils.telemetry import TelemetryCollector
except Exception:
    class TelemetryCollector:
        def __init__(self, cfg=None):
            pass

        def log_prediction(self, *a, **k):
            pass

        def close(self):
            pass

try:
    from brain.deployment.hardware_diagnostics.checker import HardwareChecker
except Exception:
    class HardwareChecker:
        def __init__(self, cfg=None, telemetry=None):
            self._cfg = cfg

        def run_all_tests(self):
            return {"failed_tests": [], "passed_tests": [], "skipped_tests": [], "test_results": {}}

        def get_system_health_status(self):
            return None

        def get_disabled_modules(self):
            return []

try:
    from brain.data_collection.logger import DataLogger
except Exception:
    class DataLogger:
        def __init__(self, cfg=None):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def get_current_window(self) -> Dict[str, Any]:
            return {"features": {}, "sensors": {}}

try:
    from brain.inference.edge_inference import EdgeInferenceEngine
except Exception:
    class EdgeInferenceEngine:
        def __init__(self, cfg=None):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def get_latest_prediction(self) -> Dict[str, Any]:
            return {"risk_score": 0.0, "decision_level": "LOW"}

try:
    from brain.battery_monitor.rules.bms_monitor import BMSMonitor
except Exception:
    class BMSMonitor:
        def __init__(self, cfg=None):
            pass

        def process_telemetry(self, samples: Any) -> Dict[str, Any]:
            return {"alerts": [], "max_cell_temp_c": -999}

try:
    from brain.dashboard.app import create_app
    from brain.dashboard.state import DashboardState
except Exception:
    def create_app(checker=None):
        return None

    class DashboardState:
        def __init__(self):
            pass

        def update(self, *a, **k):
            pass

try:
    from brain.utils.ble_utils import broadcast_safety_packet, hash_node_id
except Exception:
    def broadcast_safety_packet(vehicle_hash: str, packet: Dict[str, Any], config: Optional[ConfigLoader] = None):
        return None

    def hash_node_id(node_id: str, salt: str = "") -> str:
        return "" if node_id is None else str(node_id)[:8]

try:
    from brain.inference.alert_manager import AlertManager
except Exception:
    class AlertManager:
        def __init__(self, cfg=None, gsm_handler=None):
            # minimal fallback: accept gsm_handler kwarg to match primary implementation
            self.config = cfg
            self.gsm_handler = gsm_handler

        def process_decision(self, decision: Dict[str, Any]):
            pass

try:
    from data_collection.sensors.gsm_handler import GSMHandler
except Exception:
    class GSMHandler:
        def __init__(self, cfg=None):
            pass

        def send_sms(self, to: str, message: str) -> bool:
            return False

        def close(self):
            pass


_running = True


def _signal_handler(sig, frame):
    global _running
    print(f"Received signal {sig}; shutting down orchestrator...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def build_safety_packet(vehicle_hash: str, fatigue_pct: int, risk_pct: int, battery_severity: str, max_cell_temp: float, ble_neighbors: int, lat: float, lon: float, speed: float) -> Dict[str, Any]:
    return {
        "fatigue": int(fatigue_pct) if fatigue_pct is not None else -999,
        "risk": int(risk_pct) if risk_pct is not None else -999,
        "battery_severity": str(battery_severity) if battery_severity is not None else "unknown",
        "max_cell_temp": float(max_cell_temp) if max_cell_temp is not None else -999.0,
        "ble_neighbors": int(ble_neighbors) if ble_neighbors is not None else 0,
        "lat": float(lat) if lat is not None else -999.0,
        "lon": float(lon) if lon is not None else -999.0,
        "speed": float(speed) if speed is not None else -999.0,
        "ts": int(time.time() * 1000),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the EV Safety Guardian orchestrator")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--mode", choices=("dev", "prod"), default="dev")
    parser.add_argument("--sim", action="store_true", help="run in simulation mode (no hardware) and generate fake telemetry)")
    parser.add_argument("--pred-out", default=None, help="optional path to write predictions as JSON lines")
    parser.add_argument("--duration", type=int, default=0, help="Run duration in seconds (0=infinite)")
    parser.add_argument("--vehicle-id", default="vehicle-unknown")
    parser.add_argument("--trip-id", default="")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config) if 'ConfigLoader' in globals() else ConfigLoader()
    telemetry = TelemetryCollector(cfg)

    # Hardware check
    checker = HardwareChecker(cfg, telemetry=telemetry)
    print("Running boot hardware check...")
    try:
        res = checker.run_all_tests()
    except Exception:
        res = {"failed_tests": [], "passed_tests": [], "skipped_tests": [], "test_results": {}}
    health = checker.get_system_health_status() if hasattr(checker, 'get_system_health_status') else None
    disabled = checker.get_disabled_modules() if hasattr(checker, 'get_disabled_modules') else []
    if health and health.get('overall_status') != 'OK':
        print("Hardware check reported degraded status; entering degraded mode and disabling faulty modules:", disabled)

    # Initialize main subsystems
    datalogger = DataLogger(cfg)
    # BMS monitor may accept telemetry collector in some implementations
    try:
        bms = BMSMonitor(cfg, telemetry)
    except TypeError:
        try:
            bms = BMSMonitor(cfg)
        except Exception:
            bms = BMSMonitor()
    inference = EdgeInferenceEngine(cfg)

    # Initialize GSM handler if configured (simulation-first default: disabled)
    gsm_handler = None
    try:
        gsm_cfg = cfg.get('sensors.gsm', {}) if hasattr(cfg, 'get') else {}
        if bool(gsm_cfg.get('enable', False)):
            try:
                # Pass the full ConfigLoader to GSMHandler so it can read keys like
                # 'sensors.gsm.port' via self.config.get(...), matching other handlers.
                gsm_handler = GSMHandler(cfg)
            except Exception:
                gsm_handler = None
    except Exception:
        gsm_handler = None

    alert_mgr = AlertManager(cfg, gsm_handler=gsm_handler)

    # Dashboard
    # create_app expects a config dict as first arg; pass dashboard config and telemetry
    try:
        dash_cfg = cfg.get('dashboard', {}) if hasattr(cfg, 'get') else {}
    except Exception:
        dash_cfg = {}
    app = None
    dashboard_state = None
    if callable(create_app):
        try:
            # Prefer newer signature that accepts telemetry_collector
            app = create_app(dash_cfg, telemetry_collector=telemetry)
        except TypeError:
            try:
                # Fallback to older signature that accepts only config
                app = create_app(dash_cfg)
            except Exception:
                app = None
    try:
        dashboard_state = DashboardState() if 'DashboardState' in globals() else None
    except Exception:
        dashboard_state = None

    # Start components
    try:
        datalogger.start()
    except Exception:
        print("Warning: DataLogger failed to start; continuing in degraded mode")
    try:
        inference.start()
    except Exception:
        print("Warning: EdgeInferenceEngine failed to start; continuing in degraded mode")

    # Start BMS monitor if available
    try:
        if hasattr(bms, 'start'):
            try:
                bms.start()
            except Exception:
                pass
    except Exception:
        pass

    # Start dashboard thread if available and mode allows (or when enabled in config)
    enable_dashboard = False
    try:
        enable_dashboard = bool(cfg.get('orchestrator.enable_dashboard', False)) if hasattr(cfg, 'get') else False
    except Exception:
        enable_dashboard = False
    if app and (args.mode == 'dev' or enable_dashboard):
        try:
            from werkzeug.serving import make_server

            server = make_server('0.0.0.0', int(cfg.get('orchestrator.dashboard_port', 5000) if hasattr(cfg, 'get') else 5000), app)

            def run_server():
                server.serve_forever()

            t = threading.Thread(target=run_server, daemon=True)
            t.start()
            print("Dashboard started on port", cfg.get('orchestrator.dashboard_port', 5000))
        except Exception:
            print("Could not start dashboard server; continuing")

    # Canonical vehicle hash
    vhash = hash_node_id(args.vehicle_id, salt=str(cfg.get('telemetry.salt', '')))

    # Main loop
    start_ts = time.time()
    iterations = 0

    # Simple simulator used when --sim is specified. It produces EAR, blink_count and fatigue_pct.
    class SimulatedFatigue:
        def __init__(self):
            self.ear = 0.30
            self.blink_count = 0
            self.t0 = time.time()

        def step(self):
            # small random drift downward
            import random
            self.ear = max(0.08, self.ear - 0.0008 + random.uniform(-0.004, 0.002))
            # blink occasionally (prob increases with fatigue)
            fatigue_pct = int(((0.32 - self.ear) / (0.32 - 0.08)) * 100)
            if random.random() < (0.01 + fatigue_pct / 1000.0):
                self.blink_count += 1
                blinked = True
            else:
                blinked = False
            return {
                'ear': round(self.ear, 3),
                'blink_count': self.blink_count,
                'fatigue_pct': fatigue_pct,
                'blinked': blinked,
                'sim_time_s': int(time.time() - self.t0),
            }

    sim = None
    pred_out_f = None
    if args.pred_out:
        try:
            import json
            pred_out_f = open(args.pred_out, 'a', encoding='utf8')
        except Exception:
            pred_out_f = None

    if args.sim:
        sim = SimulatedFatigue()
        # override datalogger.get_current_window when available to return simulated features
        try:
            def _sim_get_window():
                s = sim.step()
                return {'features': {'fatigue_pct': s['fatigue_pct']}, 'sensors': {'sim': s}}

            if hasattr(datalogger, 'get_current_window'):
                datalogger.get_current_window = _sim_get_window
        except Exception:
            pass
        # override inference.get_latest_prediction to map fatigue -> risk
        try:
            def _sim_pred():
                # read last simulated window
                try:
                    w = datalogger.get_current_window()
                    f = w.get('features', {}) if isinstance(w, dict) else {}
                    fatigue = float(f.get('fatigue_pct', 0))
                except Exception:
                    fatigue = 0.0
                risk_score = max(0.0, min(1.0, fatigue / 100.0))
                level = 'LOW' if risk_score < 0.33 else ('MEDIUM' if risk_score < 0.66 else 'HIGH')
                return {'risk_score': risk_score, 'decision_level': level, 'fatigue_pct': int(fatigue)}

            inference.get_latest_prediction = _sim_pred
        except Exception:
            pass

    try:
        while _running:
            loop_start = time.time()
            try:
                # Prefer inference engine buffer window for building packets and sensor health
                window = None
                try:
                    if hasattr(inference, 'buffer') and hasattr(inference.buffer, 'get_current_window'):
                        window = inference.buffer.get_current_window()
                except Exception:
                    window = None

                # Fallback to datalogger snapshot only in dev mode
                if window is None and args.mode == 'dev' and hasattr(datalogger, 'get_current_window'):
                    try:
                        window = datalogger.get_current_window()
                    except Exception:
                        window = None

                features = window.get('features', {}) if isinstance(window, dict) else {}
                sensors = window.get('sensors', {}) if isinstance(window, dict) else {}

                # Example: fatigue percentage from features
                fatigue_pct = int(features.get('fatigue_pct', -999)) if isinstance(features.get('fatigue_pct', None), (int, float)) else -999

                # Inference
                pred = inference.get_latest_prediction() if hasattr(inference, 'get_latest_prediction') else None
                risk_score = pred.get('risk_score', 0.0) if pred is not None else 0.0

                # BMS: pass recent samples from window into monitor
                try:
                    bms_samples = []
                    if isinstance(window, dict):
                        bms_samples = window.get('bms', []) or []
                    if hasattr(bms, 'process_telemetry'):
                        bms_alerts = bms.process_telemetry(bms_samples)
                    else:
                        bms_alerts = []
                except Exception:
                    bms_alerts = []

                # derive battery severity from alerts
                battery_sev = 'unknown'
                max_temp = -999.0
                try:
                    severity_rank = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
                    max_rank = 0
                    for a in (bms_alerts or []):
                        sev = str(a.get('severity', '')).upper()
                        r = severity_rank.get(sev, 0)
                        if r > max_rank:
                            max_rank = r
                            battery_sev = sev if sev else battery_sev
                    if bms_samples and isinstance(bms_samples, list):
                        last = bms_samples[-1]
                        if isinstance(last, dict):
                            max_temp = float(last.get('max_cell_temp_c', last.get('max_cell_temp', -999)))
                except Exception:
                    battery_sev = 'unknown'
                    max_temp = -999.0

                # BLE neighbors and GPS
                ble_neighbors = sensors.get('ble_neighbors_count', 0) if isinstance(sensors, dict) else 0
                lat = sensors.get('gps_lat', -999.0) if isinstance(sensors, dict) else -999.0
                lon = sensors.get('gps_lon', -999.0) if isinstance(sensors, dict) else -999.0
                speed = sensors.get('gps_speed_m_s', -999.0) if isinstance(sensors, dict) else -999.0

                # Build and broadcast safety packet
                packet = build_safety_packet(vhash, fatigue_pct, int(risk_score * 100), battery_sev, max_temp, ble_neighbors, lat, lon, speed)
                try:
                    broadcast_safety_packet(vhash, packet, cfg)
                except Exception:
                    print("Warning: broadcast failed")

                # Dashboard update (best-effort) using app.dashboard_state
                try:
                    if app and hasattr(app, 'dashboard_state') and pred is not None:
                        pred_payload = dict(pred)
                        pred_payload.setdefault('timestamp', int(time.time() * 1000))
                        try:
                            app.dashboard_state.update_latest_prediction(pred_payload)
                        except Exception:
                            try:
                                app.dashboard_state.update(pred_payload)
                            except Exception:
                                pass
                except Exception:
                    pass

                # Alerts - pass full prediction payload
                try:
                    if pred is not None:
                        alert_mgr.process_decision(pred)
                except Exception:
                    pass

                # Note: EdgeInferenceEngine logs telemetry itself; avoid duplicating here

            except Exception:
                traceback.print_exc()

            iterations += 1
            # duration handling
            if args.duration and (time.time() - start_ts) >= args.duration:
                break

            # sleep to maintain ~1s loop (account for work time)
            elapsed = time.time() - loop_start
            to_sleep = max(0.0, (cfg.get('orchestrator.loop_interval_s', 1.0) if hasattr(cfg, 'get') else 1.0) - elapsed)
            time.sleep(to_sleep)
    finally:
        print("Shutting down orchestrator: stopping subsystems...")
        try:
            datalogger.stop()
        except Exception:
            pass
        try:
            inference.stop()
        except Exception:
            pass
        try:
            if hasattr(bms, 'stop'):
                try:
                    bms.stop()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            telemetry.close()
        except Exception:
            pass

    print("Orchestrator stopped. Iterations:", iterations)


if __name__ == '__main__':
    main()
