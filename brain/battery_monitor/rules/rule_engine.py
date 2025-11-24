"""RuleEngine: evaluate BMS telemetry against configurable thresholds.

This module implements simple, deterministic rules for battery health
monitoring. Rules are configurable via `battery_monitor.rules` in the
configuration (see `config/bms_rules.yaml`).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RuleAlert:
    rule_name: str
    triggered: bool
    severity: str
    reason: str
    value: Any
    source: str = "rule"


class RuleEngine:
    def __init__(self, config):
        # config may be ConfigLoader or mapping
        try:
            self.config = config
            self.thresholds = self.config.get("battery_monitor.rules.thresholds", {})
            self.severities = self.config.get("battery_monitor.rules.severities", {})
        except Exception:
            # tolerate plain dict
            self.config = None
            self.thresholds = {}
            self.severities = {}

        # defaults
        self.thresholds.setdefault("max_cell_temp_c", 70.0)
        self.thresholds.setdefault("voltage_diff_threshold_v", 0.25)
        self.thresholds.setdefault("current_spike_threshold_a", 100.0)
        self.thresholds.setdefault("soh_drop_rate_percent_per_day", 0.5)
        self.thresholds.setdefault("soh_history_window_samples", 60)
        # Pack-level defaults (power bus)
        # Pack-level thresholds are expected to be provided via config/bms_rules.yaml
        # Use get(..., default) in checks to fall back to sensible defaults when missing.

        # severity mapping defaults
        self.severities.setdefault("low", 0.2)
        self.severities.setdefault("medium", 0.5)
        self.severities.setdefault("high", 0.75)
        self.severities.setdefault("critical", 0.9)

        # internal state for SoH history (recent samples)
        self._soh_history: Deque[float] = deque(maxlen=int(self.thresholds.get("soh_history_window_samples", 60)))
        self._lock = Lock()
        
        # determine whether SoH tracking is enabled in config
        try:
            # support ConfigLoader-like objects with dotted keys
            enabled = True
            if hasattr(self.config, "get"):
                val = self.config.get("battery_monitor.rules.enable_soh_tracking", None)
                if val is None:
                    # try nested dict access if config is a plain mapping
                    top = self.config.get("battery_monitor", {})
                    val = top.get("rules", {}).get("enable_soh_tracking", None) if isinstance(top, dict) else None
                if val is not None:
                    enabled = bool(val)
            else:
                # plain mapping
                enabled = bool(self.config.get("battery_monitor", {}).get("rules", {}).get("enable_soh_tracking", True))
        except Exception:
            enabled = True
        self.enable_soh_tracking = enabled

    def severity_label(self, ratio: float) -> str:
        # ratio 0..inf where 0 = at threshold, >0 exceeds. Map to severity labels
        # normalize to 0..1 scale by simple logistic-ish mapping using severities config
        if ratio <= 0:
            return "low"
        # map by thresholds
        if ratio >= self.severities.get("critical", 0.9):
            return "CRITICAL"
        if ratio >= self.severities.get("high", 0.75):
            return "HIGH"
        if ratio >= self.severities.get("medium", 0.5):
            return "MEDIUM"
        return "LOW"

    def _max_cell_temp_check(self, telemetry: Dict[str, Any]) -> Optional[RuleAlert]:
        temps = telemetry.get("cell_temps")
        if not temps:
            return None
        try:
            max_t = max(float(t) for t in temps)
        except Exception:
            return None
        thresh = float(self.thresholds.get("max_cell_temp_c", 70.0))
        if max_t > thresh:
            ratio = (max_t - thresh) / max(1.0, thresh)
            severity = self.severity_label(ratio)
            reason = f"Max cell temp {max_t:.1f}C > {thresh:.1f}C"
            return RuleAlert("max_cell_temp", True, severity, reason, max_t)
        return RuleAlert("max_cell_temp", False, "LOW", "", max_t)

    def _voltage_diff_check(self, telemetry: Dict[str, Any]) -> Optional[RuleAlert]:
        volts = telemetry.get("cell_voltages")
        if not volts:
            return None
        try:
            vmin = min(float(v) for v in volts)
            vmax = max(float(v) for v in volts)
        except Exception:
            return None
        diff = vmax - vmin
        thresh = float(self.thresholds.get("voltage_diff_threshold_v", 0.25))
        if diff > thresh:
            ratio = (diff - thresh) / max(1e-3, thresh)
            severity = self.severity_label(ratio)
            reason = f"Voltage diff {diff:.3f}V > {thresh:.3f}V"
            return RuleAlert("voltage_diff", True, severity, reason, diff)
        return RuleAlert("voltage_diff", False, "LOW", "", diff)

    def _current_spike_check(self, telemetry: Dict[str, Any]) -> Optional[RuleAlert]:
        # support both 'pack_current' (CAN) and 'pack_current_a' (ADC)
        cur = telemetry.get("pack_current") if telemetry.get("pack_current") is not None else telemetry.get('pack_current_a')
        if cur is None:
            return None
        try:
            val = abs(float(cur))
        except Exception:
            return None
        # prefer pack-specific threshold if present
        thresh = float(self.thresholds.get("pack_current_spike_a", self.thresholds.get("current_spike_threshold_a", 100.0)))
        if val > thresh:
            ratio = (val - thresh) / max(1.0, thresh)
            severity = self.severity_label(ratio)
            reason = f"Pack current {val:.1f}A > {thresh:.1f}A"
            return RuleAlert("current_spike", True, severity, reason, val)
        return RuleAlert("current_spike", False, "LOW", "", val)

    def _pack_voltage_check(self, telemetry: Dict[str, Any]) -> Optional[RuleAlert]:
        # pack_voltage may be present as 'pack_voltage' or 'pack_voltage_v'
        v = telemetry.get('pack_voltage') if telemetry.get('pack_voltage') is not None else telemetry.get('pack_voltage_v')
        if v is None:
            return None
        try:
            val = float(v)
        except Exception:
            return None
        vmin = float(self.thresholds.get('pack_voltage_min_v', 3.0))
        vmax = float(self.thresholds.get('pack_voltage_max_v', 5.0))
        if val < vmin or val > vmax:
            # compute ratio relative to range
            if val < vmin:
                ratio = (vmin - val) / max(1e-3, vmin)
            else:
                ratio = (val - vmax) / max(1e-3, vmax)
            severity = self.severity_label(ratio)
            reason = f"Pack voltage {val:.3f}V out of range [{vmin:.3f},{vmax:.3f}]"
            return RuleAlert('pack_voltage_out_of_range', True, severity, reason, val)
        return RuleAlert('pack_voltage_out_of_range', False, 'LOW', '', val)

    def _pack_temp_check(self, telemetry: Dict[str, Any]) -> Optional[RuleAlert]:
        t = telemetry.get('pack_temp_c') if telemetry.get('pack_temp_c') is not None else telemetry.get('max_temp_c')
        if t is None:
            return None
        try:
            val = float(t)
        except Exception:
            return None
        thresh = float(self.thresholds.get('pack_temp_max_c', 70.0))
        if val > thresh:
            ratio = (val - thresh) / max(1.0, thresh)
            severity = self.severity_label(ratio)
            reason = f"Pack temp {val:.1f}C > {thresh:.1f}C"
            return RuleAlert('pack_temp_high', True, severity, reason, val)
        return RuleAlert('pack_temp_high', False, 'LOW', '', val)

    def _voltage_drop_check(self, telemetry: Dict[str, Any]) -> Optional[RuleAlert]:
        # track voltage history across evaluate calls using a small deque
        try:
            if not hasattr(self, '_pack_voltage_history'):
                from collections import deque

                self._pack_voltage_history = deque(maxlen=5)
        except Exception:
            return None
        v = telemetry.get('pack_voltage') if telemetry.get('pack_voltage') is not None else telemetry.get('pack_voltage_v')
        if v is None:
            return None
        try:
            val = float(v)
        except Exception:
            return None
        self._pack_voltage_history.append(val)
        if len(self._pack_voltage_history) < 2:
            return RuleAlert('voltage_drop', False, 'LOW', '', 0.0)
        drop = float(self._pack_voltage_history[0]) - float(self._pack_voltage_history[-1])
        thresh = float(self.thresholds.get('voltage_drop_threshold_v', 0.5))
        if drop > thresh:
            ratio = (drop - thresh) / max(1e-3, thresh)
            severity = self.severity_label(ratio)
            reason = f"Voltage drop {drop:.3f}V > {thresh:.3f}V"
            return RuleAlert('voltage_drop_detected', True, severity, reason, drop)
        return RuleAlert('voltage_drop', False, 'LOW', '', drop)

    def _soh_drop_rate_check(self, telemetry: Dict[str, Any]) -> Optional[RuleAlert]:
        soh = telemetry.get("soh")
        if soh is None:
            return None
        try:
            val = float(soh)
        except Exception:
            return None
        # If SoH tracking is disabled via config, do not maintain history or emit this check
        if not getattr(self, "enable_soh_tracking", True):
            return None

        with self._lock:
            self._soh_history.append(val)
            if len(self._soh_history) < 2:
                return RuleAlert("soh_drop_rate", False, "LOW", "", 0.0)
            # compute drop per sample, scale to percent per day assuming 1 sample per second
            window = list(self._soh_history)
            drop = window[0] - window[-1]
            # samples count to days (samples per day = len * (86400 / window_seconds))
            # assume samples are 1s apart for simplicity
            seconds = len(window)
            drop_per_day = (drop / max(1.0, seconds)) * 86400.0
            thresh = float(self.thresholds.get("soh_drop_rate_percent_per_day", 0.5))
            if drop_per_day > thresh:
                ratio = (drop_per_day - thresh) / max(1e-6, thresh)
                severity = self.severity_label(ratio)
                reason = f"SoH drop {drop_per_day:.3f}%/day > {thresh:.3f}%/day"
                return RuleAlert("soh_drop_rate", True, severity, reason, drop_per_day)
            return RuleAlert("soh_drop_rate", False, "LOW", "", drop_per_day)

    def evaluate(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate configured rules against a telemetry sample and return alerts."""
        alerts: List[Dict[str, Any]] = []
        try:
            checks = [
                self._max_cell_temp_check,
                self._voltage_diff_check,
                self._current_spike_check,
                # pack-level checks
                self._pack_voltage_check,
                self._pack_temp_check,
                self._voltage_drop_check,
                self._soh_drop_rate_check,
            ]
            for chk in checks:
                try:
                    res = chk(telemetry)
                except Exception:
                    logger.exception("Rule check failed: %s", chk)
                    res = None
                if res is None:
                    continue
                if isinstance(res, RuleAlert):
                    if res.triggered:
                        alerts.append({
                            "rule_name": res.rule_name,
                            "triggered": True,
                            "severity": res.severity,
                            "reason": res.reason,
                            "value": res.value,
                        })
                else:
                    # legacy dict
                    alerts.append(res)
        except Exception:
            logger.exception("Evaluation failed")
        return alerts
