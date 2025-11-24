"""FatigueAlertManager: local actuation for driver fatigue alerts.

Provides buzzer, LED, and TTS alerts. Graceful fallbacks for non-RPi
environments (DummyGPIO) and for missing TTS engines (espeak subprocess).
"""
from __future__ import annotations

import logging
import threading
import time
import subprocess
from typing import Any, Dict

logger = logging.getLogger(__name__)


class _DummyGPIO:
    BCM = None

    def __init__(self):
        self._states = {}

    def setmode(self, *a, **k):
        return None

    def setup(self, pin, mode):
        self._states[pin] = False

    def output(self, pin, value):
        self._states[pin] = bool(value)

    def cleanup(self):
        self._states.clear()


class FatigueAlertManager:
    def __init__(self, config: Any):
        # config may be a ConfigLoader or plain dict-like
        try:
            sect = config.get("sensors.driver_fatigue") if hasattr(config, "get") else config
        except Exception:
            sect = config

        sc = sect.get("state_classification", {}) if isinstance(sect, dict) else {}
        alerts = sect.get("alerts", {}) if isinstance(sect, dict) else {}

        self.enable_alerts = bool(alerts.get("enable_alerts", True))
        self.buzzer_pin = int(alerts.get("gpio_buzzer_pin", 18))
        self.led_pin = int(alerts.get("gpio_led_pin", 23))
        self.enable_tts = bool(alerts.get("enable_tts", True))
        self.tts_engine = str(alerts.get("tts_engine", "pyttsx3"))
        self.tts_rate = float(alerts.get("tts_rate", 150)) if alerts.get("tts_rate") is not None else 150.0
        self.tts_volume = float(alerts.get("tts_volume", 1.0)) if alerts.get("tts_volume") is not None else 1.0

        # GPIO setup with graceful fallback
        try:
            import RPi.GPIO as GPIO  # type: ignore
        except Exception:
            GPIO = _DummyGPIO()
        self._GPIO = GPIO
        try:
            self._GPIO.setmode(getattr(self._GPIO, "BCM", None))
            self._GPIO.setup(self.buzzer_pin, getattr(self._GPIO, "OUT", None))
            self._GPIO.setup(self.led_pin, getattr(self._GPIO, "OUT", None))
        except Exception:
            logger.exception("Failed to initialize GPIO for FatigueAlertManager; using DummyGPIO fallback")

        # TTS engine init with fallback
        self._tts_engine = None
        if self.enable_tts:
            try:
                import pyttsx3

                self._tts_engine = pyttsx3.init()
                try:
                    self._tts_engine.setProperty("rate", int(self.tts_rate))
                    self._tts_engine.setProperty("volume", float(self.tts_volume))
                except Exception:
                    pass
            except Exception:
                # fallback to espeak via subprocess
                self._tts_engine = None

        self._lock = threading.Lock()

    def trigger_alert(self, state: str, score: float, reason: str) -> None:
        if not self.enable_alerts:
            return

        # Only act on DROWSY and MICROSLEEP
        with self._lock:
            try:
                if state == "DROWSY":
                    logger.warning("Fatigue Alert: DROWSY score=%.2f reason=%s", score, reason)
                    self._beep_buzzer(1.0)
                    self._blink_led(3)
                    self._speak("Driver drowsy, please take a break")
                elif state == "MICROSLEEP":
                    logger.warning("Fatigue Alert: MICROSLEEP score=%.2f reason=%s", score, reason)
                    self._beep_buzzer(2.0)
                    self._led_on()
                    self._speak("Wake up! Pull over immediately")
                else:
                    # no local alert for NORMAL/TIRED
                    pass
            except Exception:
                logger.exception("Failed to trigger fatigue alert")

    def _beep_buzzer(self, duration: float) -> None:
        try:
            self._GPIO.output(self.buzzer_pin, True)
            time.sleep(duration)
            self._GPIO.output(self.buzzer_pin, False)
        except Exception:
            logger.exception("Buzzer operation failed")

    def _blink_led(self, times: int, period: float = 0.2) -> None:
        try:
            for _ in range(times):
                self._GPIO.output(self.led_pin, True)
                time.sleep(period)
                self._GPIO.output(self.led_pin, False)
                time.sleep(period)
        except Exception:
            logger.exception("LED blink failed")

    def _led_on(self) -> None:
        try:
            self._GPIO.output(self.led_pin, True)
        except Exception:
            logger.exception("LED on failed")

    def _led_off(self) -> None:
        try:
            self._GPIO.output(self.led_pin, False)
        except Exception:
            logger.exception("LED off failed")

    def _speak(self, message: str) -> None:
        try:
            if self._tts_engine is not None:
                self._tts_engine.say(message)
                self._tts_engine.runAndWait()
            else:
                # fallback to espeak
                subprocess.run(["espeak", message], check=False)
        except Exception:
            logger.exception("TTS failed for message: %s", message)

    def close(self) -> None:
        try:
            if self._GPIO is not None:
                try:
                    self._GPIO.cleanup()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self._tts_engine is not None:
                try:
                    # pyttsx3 has no explicit stop/close; best-effort
                    pass
                except Exception:
                    pass
        except Exception:
            pass
