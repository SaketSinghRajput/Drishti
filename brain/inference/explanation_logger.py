"""ExplanationLogger: structured logging for predictions and explanations."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from config.config_loader import ConfigLoader
from utils.time_utils import get_utc_timestamp

logger = logging.getLogger(__name__)


class ExplanationLogger:
    def __init__(self, config: ConfigLoader, output_dir: Optional[str] = None):
        self.config = config
        self.enabled = bool(self.config.get("inference.explanation_log_enabled", True))
        self.retention_h = int(self.config.get("inference.explanation_log_retention_hours", 24))
        self.buffer_size = int(self.config.get("inference.explanation_log_buffer_size", 10))
        self.include_full_shap = bool(self.config.get("inference.explanation_log_include_full_shap", False))
        self.output_dir = output_dir or str(self.config.get("inference.explanation_log_output_dir", "data/explanations"))
        os.makedirs(self.output_dir, exist_ok=True)
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._current_file = None
        # Only open a new file when logging is enabled to avoid side effects
        if self.enabled:
            self._open_new_file()

    def _open_new_file(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_path = os.path.join(self.output_dir, f"explanations_{ts}.jsonl")
        self._current_file = open(self._current_path, "a", encoding="utf-8")

    def _format_entry(self, decision: Dict[str, Any], explanation: Dict[str, Any], features: Dict[str, Any], sensor_health: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "timestamp": get_utc_timestamp(),
            "trip_id": decision.get("metadata", {}).get("trip_id") if decision.get("metadata") else None,
            "vehicle_hash": decision.get("metadata", {}).get("vehicle_hash") if decision.get("metadata") else None,
            "risk_score": decision.get("risk_score"),
            "alert_level": decision.get("alert_level"),
            "alert_active": decision.get("alert_active"),
            "uncertainty_flag": decision.get("uncertainty_flag"),
            "explanation_message": explanation.get("message") if explanation else None,
            "top_features": explanation.get("top_features") if explanation else None,
            "feature_snapshot": features,
            "sensor_health": sensor_health,
            "decision_metadata": decision.get("metadata"),
        }
        if self.include_full_shap and explanation:
            entry["full_shap_values"] = explanation.get("shap_values")
        # privacy handling: anonymize vehicle_hash if configured
        if bool(self.config.get("privacy.hash_vehicle_ids", True)) and entry.get("vehicle_hash"):
            entry["vehicle_hash"] = str(hash(entry["vehicle_hash"]))
        return entry

    def log_prediction(self, decision: Dict[str, Any], explanation: Dict[str, Any], features: Dict[str, Any], sensor_health: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        entry = self._format_entry(decision, explanation or {}, features or {}, sensor_health or {})
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        if not self._current_file:
            # nothing to flush to; drop buffer safely
            self._buffer = []
            return
        try:
            for e in self._buffer:
                line = json.dumps(e, default=str)
                self._current_file.write(line + "\n")
            self._current_file.flush()
            os.fsync(self._current_file.fileno())
            self._buffer = []
            # rotate logs based on retention
            self._rotate()
        except Exception as ex:
            logger.exception("Failed to flush explanation logs: %s", ex)

    def _rotate(self) -> None:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self.retention_h)
            for fn in os.listdir(self.output_dir):
                if not fn.startswith("explanations_"):
                    continue
                path = os.path.join(self.output_dir, fn)
                # parse timestamp from filename
                try:
                    ts = fn.split("_")[1].split(".")[0]
                    file_dt = datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                    if file_dt < cutoff:
                        os.remove(path)
                except Exception:
                    continue
        except Exception as ex:
            logger.exception("Log rotation failed: %s", ex)

    def close(self) -> None:
        with self._lock:
            try:
                # only flush if we have an open file
                if self._current_file:
                    self._flush()
            finally:
                try:
                    if self._current_file:
                        self._current_file.close()
                        self._current_file = None
                except Exception:
                    pass
