import pytest

from config.config_loader import ConfigLoader
from battery_monitor.rules.rule_engine import RuleEngine


class DummyConfig:
    def __init__(self, mapping=None):
        self._m = mapping or {}

    def get(self, key, default=None):
        parts = key.split(".")
        cur = self._m
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur


def test_max_cell_temp_check_triggered():
    cfg = DummyConfig({"battery_monitor": {"rules": {"thresholds": {"max_cell_temp_c": 70}}}})
    re = RuleEngine(cfg)
    telem = {"cell_temps": [65, 72, 68]}
    alerts = re.evaluate(telem)
    assert any(a["rule_name"] == "max_cell_temp" or "temp" in a["reason"].lower() for a in alerts)


def test_voltage_diff_check_triggered():
    cfg = DummyConfig({"battery_monitor": {"rules": {"thresholds": {"voltage_diff_threshold_v": 0.2}}}})
    re = RuleEngine(cfg)
    telem = {"cell_voltages": [3.4, 3.65, 3.5]}
    alerts = re.evaluate(telem)
    assert any(a["rule_name"] == "voltage_diff" for a in alerts)


def test_current_spike_check():
    cfg = DummyConfig({"battery_monitor": {"rules": {"thresholds": {"current_spike_threshold_a": 50}}}})
    re = RuleEngine(cfg)
    telem = {"pack_current": 120}
    alerts = re.evaluate(telem)
    assert any(a["rule_name"] == "current_spike" for a in alerts)


def test_missing_fields_no_crash():
    cfg = DummyConfig({})
    re = RuleEngine(cfg)
    telem = {"cell_temps": None}
    alerts = re.evaluate(telem)
    assert isinstance(alerts, list)
