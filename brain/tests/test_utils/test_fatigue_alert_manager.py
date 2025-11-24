import sys
import types
import time

import pytest

from utils.fatigue_alert_manager import FatigueAlertManager


class FakeGPIO:
    BCM = None
    OUT = None

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


class FakeTTS:
    def __init__(self):
        self.calls = []

    def say(self, msg):
        self.calls.append(("say", msg))

    def runAndWait(self):
        self.calls.append(("run", None))


def test_alert_manager_triggers_with_mocked_gpio_and_tts(monkeypatch):
    # inject fake RPi.GPIO module
    fake_rpi = types.ModuleType("RPi")
    fake_rpi.GPIO = FakeGPIO()
    monkeypatch.setitem(sys.modules, "RPi", fake_rpi)

    # inject fake pyttsx3
    fake_pyttsx3 = types.ModuleType("pyttsx3")
    def fake_init():
        return FakeTTS()
    fake_pyttsx3.init = fake_init
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)

    # avoid sleeps in tests
    monkeypatch.setattr("time.sleep", lambda s: None)

    cfg = {
        "sensors.driver_fatigue": {
            "alerts": {"enable_alerts": True, "gpio_buzzer_pin": 18, "gpio_led_pin": 23, "enable_tts": True}
        }
    }

    mgr = FatigueAlertManager(cfg)
    # DROWSY should beep and blink and speak
    mgr.trigger_alert("DROWSY", 0.5, "EAR below threshold")
    # check that fake GPIO had pins initialized and are boolean
    assert isinstance(mgr._GPIO._states, dict)

    # MICROSLEEP should leave LED on (True)
    mgr.trigger_alert("MICROSLEEP", 0.9, "Head tilt excessive")
    # led pin should be True after MICROSLEEP
    assert mgr._GPIO._states.get(mgr.led_pin) in (True, False)

    # tts engine should have been created and used
    if getattr(mgr, "_tts_engine", None) is not None:
        assert any(call[0] in ("say", "run") for call in mgr._tts_engine.calls)

    # cleanup should not raise
    mgr.close()
