"""AlertManager: handle battery alerts with visual, GPIO, BLE and storage actions.

This module provides a lightweight implementation that degrades gracefully
when GPIO or BLE hardware is unavailable (useful for development and CI).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import hashlib

logger = logging.getLogger(__name__)


class _DummyGPIO:
    def __init__(self):
        self._pins = {}

    def setup(self, pin, mode):
        self._pins[pin] = False

    def output(self, pin, value):
        self._pins[pin] = bool(value)

    def cleanup(self):
        self._pins.clear()


class AlertManager:
    def __init__(self, config: Dict[str, Any], telemetry=None, ble_stub=None, event_logger=None):
        self.config = config
        self.telemetry = telemetry
        self.ble_stub = ble_stub
        self.event_logger = event_logger
        alerts_cfg = config.get("battery_monitor", {}).get("alerts", {})
        self.enabled = bool(alerts_cfg.get("enable", True))
        self.log_path = str(alerts_cfg.get("log_path", "data/battery_alerts.log"))
        self.shutdown_log = str(alerts_cfg.get("shutdown_log_path", "data/shutdown.log"))
        self.buzzer_pin = int(alerts_cfg.get("gpio_buzzer_pin", 18))
        self.led_pin = int(alerts_cfg.get("gpio_led_pin", 23))
        self.relay_pin = int(alerts_cfg.get("gpio_relay_pin", 24))
        self.history_size = int(alerts_cfg.get("alert_history_size", 100))
        self.ble_enabled = bool(alerts_cfg.get("ble_broadcast_enabled", True))

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        # open log file in append mode
        try:
            self._log_fh = open(self.log_path, "a", encoding="utf-8")
        except Exception:
            self._log_fh = None

        # initialize GPIO (best-effort)
        self._gpio = None
        try:
            import RPi.GPIO as GPIO  # type: ignore

            self._gpio = GPIO
            self._gpio.setmode(GPIO.BCM)
            self._gpio.setup(self.buzzer_pin, GPIO.OUT)
            self._gpio.setup(self.led_pin, GPIO.OUT)
            self._gpio.setup(self.relay_pin, GPIO.OUT)
        except Exception:
            logger.debug("RPi.GPIO not available, using DummyGPIO")
            self._gpio = _DummyGPIO()
            try:
                self._gpio.setup(self.buzzer_pin, None)
                self._gpio.setup(self.led_pin, None)
                self._gpio.setup(self.relay_pin, None)
            except Exception:
                pass

        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []

    def _hash_id(self, vid: Optional[str]) -> Optional[str]:
        if vid is None:
            return None
        salt = str(self.config.get("privacy", {}).get("node_id_salt", ""))
        return hashlib.sha256((salt + vid).encode("utf-8")).hexdigest()

    def trigger_alert(self, alert: Dict[str, Any], telemetry_snapshot: Optional[List[Dict[str, Any]]] = None, features: Optional[Dict[str, Any]] = None, scores: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        severity = str(alert.get("severity", "LOW")).upper()
        ts = datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": ts,
            "severity": severity,
            "source": alert.get("source"),
            "reason": alert.get("reason"),
            "anomaly_score": alert.get("anomaly_score"),
            "value": alert.get("value"),
        }

        # persist to file (visual/log)
        try:
            with self._lock:
                if self._log_fh:
                    self._log_fh.write(json.dumps(entry) + "\n")
                    self._log_fh.flush()
        except Exception:
            logger.exception("Failed to write battery alert to log")

        # maintain in-memory history
        with self._lock:
            self._history.append({"ts": ts, "entry": entry})
            if len(self._history) > self.history_size:
                self._history.pop(0)

        # visual handling (console)
        try:
            self._handle_visual(entry)
        except Exception:
            logger.exception("Visual handling failed")

        # GPIO actions
        try:
            self._handle_gpio(severity)
        except Exception:
            logger.exception("GPIO handling failed")

        # BLE broadcast for critical
        try:
            if severity == "CRITICAL" and self.ble_enabled and self.ble_stub is not None:
                # try to find vehicle id in snapshot metadata
                vehicle_id = None
                if telemetry_snapshot and isinstance(telemetry_snapshot, list) and telemetry_snapshot:
                    md = telemetry_snapshot[0].get("metadata", {}) or {}
                    vehicle_id = md.get("vehicle_hash") or md.get("vehicle_id")
                if vehicle_id:
                    vehicle_id = self._hash_id(vehicle_id)
                else:
                    vehicle_id = "unknown"
                self.ble_stub.broadcast(alert, vehicle_id)
        except Exception:
            logger.exception("BLE broadcast failed")

        # Event logging (DB)
        try:
            if self.event_logger is not None:
                self.event_logger.log_alert(alert, telemetry_snapshot, features or {}, scores or {})
        except Exception:
            logger.exception("EventLogger failed")

        # shutdown simulation for critical
        try:
            if severity == "CRITICAL":
                self._handle_shutdown(entry)
        except Exception:
            logger.exception("Shutdown handling failed")

    def _handle_visual(self, entry: Dict[str, Any]) -> None:
        # Simple console log for developers
        logger.info("Battery Alert: %s %s", entry.get("severity"), entry.get("reason"))

    def _handle_gpio(self, severity: str) -> None:
        # Map severity to actions
        sev = severity.upper()
        if sev == "LOW":
            return
        # MEDIUM -> LED blink
        if sev == "MEDIUM":
            self._blink_led(times=2)
            return
        # HIGH -> LED blink + buzzer
        if sev == "HIGH":
            self._blink_led(times=3)
            self._beep_buzzer(duration=1.0)
            return
        # CRITICAL -> solid LED + buzzer + relay pulse
        if sev == "CRITICAL":
            self._led_on()
            self._beep_buzzer(duration=2.0)
            self._pulse_relay(duration=0.5)

    def _blink_led(self, times: int = 3, period: float = 0.2) -> None:
        for _ in range(times):
            try:
                self._gpio.output(self.led_pin, True)
            except Exception:
                pass
            time.sleep(period)
            try:
                self._gpio.output(self.led_pin, False)
            except Exception:
                pass
            time.sleep(period)

    def _led_on(self) -> None:
        try:
            self._gpio.output(self.led_pin, True)
        except Exception:
            pass

    def _beep_buzzer(self, duration: float = 1.0) -> None:
        try:
            self._gpio.output(self.buzzer_pin, True)
            time.sleep(duration)
            self._gpio.output(self.buzzer_pin, False)
        except Exception:
            pass

    def _pulse_relay(self, duration: float = 0.5) -> None:
        try:
            self._gpio.output(self.relay_pin, True)
            time.sleep(duration)
            self._gpio.output(self.relay_pin, False)
        except Exception:
            pass

    def _handle_shutdown(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self.shutdown_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"timestamp": entry.get("timestamp"), "reason": entry.get("reason")}) + "\n")
        except Exception:
            logger.exception("Failed to write shutdown log")

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def close(self) -> None:
        try:
            if hasattr(self._gpio, "cleanup"):
                try:
                    self._gpio.cleanup()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self._log_fh:
                self._log_fh.close()
        except Exception:
            pass
