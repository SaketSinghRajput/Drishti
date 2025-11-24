"""Decision unit that maps risk scores to alert levels with hysteresis and context checks."""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Any, Optional

from config.config_loader import ConfigLoader
from utils.time_utils import get_utc_timestamp

logger = logging.getLogger(__name__)


class DecisionUnit:
    def __init__(self, config: ConfigLoader):
        self.config = config
        thresholds = dict(self.config.get("model.risk_thresholds", {}))
        # Defaults if not present
        self.low = float(thresholds.get("low", 0.1))
        self.medium = float(thresholds.get("medium", 0.3))
        self.high = float(thresholds.get("high", 0.6))

        self.margin = float(self.config.get("inference.alert_hysteresis_margin", 0.05))
        self.dwell_s = float(self.config.get("inference.alert_hysteresis_dwell_s", 2.0))
        self.critical_sensors = list(self.config.get("inference.critical_sensors", []))
        self.min_data_quality = float(self.config.get("inference.min_data_quality_threshold", 0.5))

        self._lock = threading.Lock()
        self._current_level = "none"
        self._last_change_ts = 0.0
        # LLM config (optional blending/override)
        self.llm_enabled = bool(self.config.get("inference.llm_integration.enabled", False))
        self.llm_override_conf = float(self.config.get("inference.llm_integration.override_confidence", 0.9))

    def _score_to_level(self, score: float) -> str:
        if score < self.low:
            return "none"
        if score < self.medium:
            return "low"
        if score < self.high:
            return "medium"
        return "high"

    def update(self, risk_score: float, sensor_health: Dict[str, Any], data_quality: Optional[float] = 1.0, explanation: Optional[Dict[str, Any]] = None, llm_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main decision logic. Returns decision dict.

        Applies hysteresis and context checks. Thread-safe.
        """
        ts = time.time()
        with self._lock:
            uncertainty = False
            reasons = []
            # Context checks
            for s in self.critical_sensors:
                info = sensor_health.get(s)
                if info is None:
                    uncertainty = True
                    reasons.append(f"Uncertain: {s} not available")
                elif info.get("stale", True):
                    uncertainty = True
                    stale_duration = info.get("stale_duration_s", "unknown")
                    reasons.append(f"Uncertain: {s} stale ({stale_duration}s)")
            if data_quality is None:
                data_quality = 1.0
            if data_quality < self.min_data_quality:
                uncertainty = True
                reasons.append("Uncertain: low data quality")

            # Clamp score
            score = max(0.0, min(1.0, float(risk_score)))

            # If LLM provided an augmentation, blend or override based on confidence
            if self.llm_enabled and llm_output:
                try:
                    llm_score = float(llm_output.get("llm_score", 0.0))
                    llm_conf = float(llm_output.get("confidence", 0.0))
                    # If confidence exceeds override threshold, take LLM score
                    if llm_conf >= self.llm_override_conf:
                        reasons.append(f"LLM override (conf={llm_conf:.2f})")
                        score = max(0.0, min(1.0, llm_score))
                    else:
                        # blend: weight by LLM confidence
                        blended = (1.0 - llm_conf) * score + llm_conf * llm_score
                        reasons.append(f"LLM adjusted (conf={llm_conf:.2f})")
                        score = max(0.0, min(1.0, blended))
                except Exception:
                    # ignore LLM problems and proceed
                    pass
            target_level = self._score_to_level(score)

            # Hysteresis: determine if we should change level
            allow_change = (ts - self._last_change_ts) >= self.dwell_s
            prev_level = self._current_level
            changed = False

            if target_level == prev_level:
                # no change
                pass
            else:
                # Escalation requires exceeding target threshold + margin
                # De-escalation requires dropping below target threshold - margin
                if self._should_transition(prev_level, target_level, score, allow_change):
                    self._current_level = target_level
                    self._last_change_ts = ts
                    changed = True

            # alert_active reflects underlying level; uncertainty is orthogonal flag
            alert_active = self._current_level != "none"
            # Use explanation message only when explainability is available
            exp_available = bool(explanation and explanation.get("explainability_available"))
            if exp_available and explanation.get("message"):
                reason_text = explanation.get("message")
            elif reasons:
                reason_text = ", ".join(reasons)
            else:
                if changed:
                    reason_text = f"Escalated to {self._current_level}" if self._level_index(target_level) > self._level_index(prev_level) else f"De-escalated to {self._current_level}"
                else:
                    reason_text = "Stable"

            # If uncertainty but explanation is available, add which sensors are available
            if uncertainty and exp_available:
                available_sensors = [s for s in self.critical_sensors if sensor_health.get(s) and not sensor_health[s].get("stale")]
                if available_sensors:
                    reason_text = reason_text + f"; Relying on: {', '.join(available_sensors)}"

            # If LLM reasoning available, surface it prominently
            llm_reasoning = None
            try:
                if llm_output and isinstance(llm_output, dict):
                    llm_reasoning = llm_output.get("reasoning") or llm_output.get("explanation")
                    if llm_reasoning:
                        # prepend LLM reasoning to the reason text for visibility
                        reason_text = f"LLM: {llm_reasoning}; {reason_text}"
            except Exception:
                pass

            decision = {
                "risk_score": score,
                "alert_level": self._current_level,
                "alert_active": bool(alert_active),
                "uncertainty_flag": bool(uncertainty),
                "reason": reason_text,
                "llm_reasoning": llm_reasoning,
                "timestamp": get_utc_timestamp(),
                "metadata": {"prev_level": prev_level, "last_change_ts": self._last_change_ts, "dwell_s": self.dwell_s, "margin": self.margin, "llm": llm_output},
                "explanation": explanation if explanation else {},
            }
            return decision

    def _should_transition(self, prev: str, target: str, score: float, allow_time: bool) -> bool:
        # For escalation: require score >= target_threshold + margin
        # For de-escalation: require score <= target_threshold - margin
        if not allow_time:
            return False
        if prev == "none":
            # Escalate from none to low/medium/high
            thresh = self._level_threshold(target)
            return score >= (thresh + self.margin)
        else:
            # General rule: if target is higher than prev, require exceed; if lower, require drop below
            prev_idx = self._level_index(prev)
            target_idx = self._level_index(target)
            thresh = self._level_threshold(target)
            if target_idx > prev_idx:
                # escalate
                return score >= (thresh + self.margin)
            else:
                # de-escalate
                return score <= (thresh - self.margin)

    def _level_threshold(self, level: str) -> float:
        if level == "none":
            return 0.0
        if level == "low":
            return self.low
        if level == "medium":
            return self.medium
        if level == "high":
            return self.high
        return 1.0

    def _level_index(self, level: str) -> int:
        order = ["none", "low", "medium", "high"]
        try:
            return order.index(level)
        except ValueError:
            return 0

    def reset(self) -> None:
        with self._lock:
            self._current_level = "none"
            self._last_change_ts = 0.0

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {"current_level": self._current_level, "last_change_ts": self._last_change_ts}
