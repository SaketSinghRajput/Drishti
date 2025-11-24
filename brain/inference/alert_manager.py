"""AlertManager: two-stage alert escalation and action generation."""
from __future__ import annotations

import logging
import time
import threading
from typing import Dict, Any, Optional

from config.config_loader import ConfigLoader
from utils.time_utils import get_utc_timestamp

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, config: ConfigLoader, gsm_handler: Optional[Any] = None):
        self.config = config
        self.stage1_dur = float(self.config.get("inference.alert_stage1_duration_s", 0))
        self.stage2_dur = float(self.config.get("inference.alert_stage2_duration_s", 5.0))
        # Normalize configured minimum level for stage2
        raw_min = str(self.config.get("inference.alert_stage2_min_level", "medium"))
        self.stage2_min = self._normalize_level(raw_min)
        self._lock = threading.Lock()
        self._current_stage = "none"
        self._stage_entry_ts = 0.0
        self._last_level = "none"
        # GSM integration
        self.gsm_handler = gsm_handler
        try:
            self.enable_gsm = bool(self.config.get('inference.alerts.enable_gsm', False))
        except Exception:
            self.enable_gsm = False
        try:
            self.gsm_phone = str(self.config.get('inference.alerts.gsm_phone_number', ''))
        except Exception:
            self.gsm_phone = ''
        if self.enable_gsm and not self.gsm_phone:
            logger.warning('GSM alerts enabled but no phone number configured; disabling GSM alerts')
            self.enable_gsm = False

    def _normalize_level(self, raw_level: Any) -> str:
        """Normalize various alert level representations into canonical levels.

        Canonical set: 'none', 'low', 'medium', 'high', 'critical'
        """
        try:
            s = (str(raw_level) if raw_level is not None else "").strip().lower()
        except Exception:
            s = ""

        mapping = {
            'none': 'none',
            'ok': 'none',
            'clear': 'none',
            'low': 'low',
            'medium': 'medium',
            'med': 'medium',
            'high': 'high',
            'severe': 'high',
            'high_risk': 'high',
            'crit': 'critical',
            'critical': 'critical',
        }

        if s in mapping:
            return mapping[s]
        if 'crit' in s:
            return 'critical'
        if 'high' in s:
            return 'high'
        if 'med' in s:
            return 'medium'
        if 'low' in s:
            return 'low'
        return 'none'

    def _actions_for_stage(self, stage: str, message: str) -> Dict[str, Any]:
        if stage == "none":
            return {"visual": False, "audio": False, "haptic": False, "escalation": False, "message": message}
        if stage == "stage1":
            return {"visual": True, "audio": True, "haptic": False, "escalation": False, "message": message}
        if stage == "stage2":
            return {"visual": True, "audio": True, "haptic": True, "escalation": True, "message": message}
        return {"visual": False, "audio": False, "haptic": False, "escalation": False, "message": message}

    def process_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Process decision and return alert action dict.

        decision should include 'alert_level' and 'alert_active' and optional 'reason'/'explanation'.
        """
        ts = time.time()
        # Normalize incoming level to canonical values so downstream logic is predictable
        raw_level = decision.get("alert_level", "none")
        level = self._normalize_level(raw_level)
        active = bool(decision.get("alert_active", False))
        message = decision.get("reason") or (decision.get("explanation", {}) or {}).get("message", "")

        with self._lock:
            if not active or level == "none":
                # clear
                self._current_stage = "none"
                self._stage_entry_ts = 0.0
                self._last_level = "none"
                return {"stage": "none", "actions": self._actions_for_stage("none", message), "timestamp": get_utc_timestamp()}

            # if level changed, reset timer
            if level != self._last_level:
                self._stage_entry_ts = ts
                self._last_level = level

            elapsed = ts - self._stage_entry_ts

            # Stage 1 immediate
            if elapsed >= self.stage1_dur:
                target_stage = "stage1"
            else:
                target_stage = "none"

            # Possible escalate to stage2 if level >= min and elapsed >= stage2_dur
            # Use a canonical ordered severity list (none < low < medium < high < critical)
            order = ["none", "low", "medium", "high", "critical"]
            try:
                min_idx = order.index(self.stage2_min)
            except ValueError:
                min_idx = 2
            cur_idx = order.index(level) if level in order else 0
            if cur_idx >= min_idx and elapsed >= self.stage2_dur:
                target_stage = "stage2"

            prev_stage = self._current_stage
            self._current_stage = target_stage
            actions = self._actions_for_stage(target_stage, message)

            # Trigger GSM SMS on transition into stage2 for critical/high alerts
            try:
                if self.enable_gsm and self.gsm_handler and target_stage == 'stage2' and prev_stage != 'stage2':
                    # only for high/critical levels
                    # Trigger only for high/critical canonical levels
                    if level in ('high', 'critical'):
                        # format message
                        try:
                            lat = None
                            lon = None
                            # Try multiple locations where GPS might be present in the decision payload
                            loc = decision.get('gps') or decision.get('location') or {}
                            if isinstance(loc, dict) and ('lat' in loc or 'lon' in loc):
                                lat = loc.get('lat')
                                lon = loc.get('lon')
                            else:
                                # Fallback to sensors namespace fields used by the window/decision schema
                                sensors = decision.get('sensors') or {}
                                if isinstance(sensors, dict):
                                    lat = sensors.get('gps_lat') or sensors.get('lat')
                                    lon = sensors.get('gps_lon') or sensors.get('lon')
                            timestamp = get_utc_timestamp()
                            sms = f"EV Safety CRITICAL: {message}. Location: [{lat},{lon}]. Time: {timestamp}"
                        except Exception:
                            sms = f"EV Safety CRITICAL: {message}. Time: {get_utc_timestamp()}"
                        try:
                            ok = self.gsm_handler.send_sms(self.gsm_phone, sms)
                            if ok:
                                logger.info('GSM SMS sent to %s', self.gsm_phone)
                            else:
                                logger.warning('GSM SMS failed to send to %s', self.gsm_phone)
                        except Exception as e:
                            logger.exception('Exception while sending GSM SMS: %s', e)
            except Exception:
                logger.exception('Error in GSM alert handling')

            return {"stage": target_stage, "actions": actions, "timestamp": get_utc_timestamp(), "elapsed_s": elapsed}

    def reset(self) -> None:
        with self._lock:
            self._current_stage = "none"
            self._stage_entry_ts = 0.0
            self._last_level = "none"

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {"current_stage": self._current_stage, "stage_entry_ts": self._stage_entry_ts, "last_level": self._last_level}
