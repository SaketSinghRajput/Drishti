"""TelemetryCollector: buffered telemetry logging with anonymization and rotation."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import base64

from utils import encryption as _encryption

from config.config_loader import ConfigLoader

logger_name = "utils.telemetry"


class TelemetryCollector:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.enabled = bool(self.config.get("telemetry.enabled", True))
        self.output_dir = str(self.config.get("telemetry.output_dir", "data/telemetry"))
        self.anonymize_ids = bool(self.config.get("telemetry.anonymize_ids", True))
        self.anonymize_location = bool(self.config.get("telemetry.anonymize_location", True))
        self.grid_m = int(self.config.get("telemetry.location_grid_size_m", 100))
        self.retention_days = int(self.config.get("telemetry.retention_days", 30))
        self.buffer_size = int(self.config.get("telemetry.buffer_size", 20))
        os.makedirs(self.output_dir, exist_ok=True)

        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._current_file = None
        # encryption settings
        self.encrypt = bool(self.config.get("telemetry.encrypt", False))
        self._encryption_key = self._get_encryption_key()
        if self.encrypt and not self._encryption_key:
            # cannot enable encryption without a key; fall back to disabled
            self.encrypt = False

        if self.enabled:
            self._open_file()

    def _open_file(self) -> None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.output_dir, f"telemetry_{ts}")
        if self.encrypt:
            # write to a temporary plaintext file, will be encrypted on flush
            path = base + ".jsonl.tmp"
        else:
            path = base + ".jsonl"
        self._current_base = base
        self._current_file = open(path, "a", encoding="utf-8")

    def _hash_id(self, vid: Optional[str]) -> Optional[str]:
        if vid is None:
            return None
        if not self.anonymize_ids:
            return vid
        # simple sha256 with optional salt from config
        salt = str(self.config.get("privacy.node_id_salt", ""))
        return hashlib.sha256((salt + vid).encode("utf-8")).hexdigest()

    def _grid_location(self, lat: float, lon: float) -> Dict[str, float]:
        # coarse grid: truncate lat/lon by grid size (approximate using meters ~ degrees conversion)
        # approximate degrees per meter at equator ~ 1/111320
        deg_per_m = 1.0 / 111320.0
        step = self.grid_m * deg_per_m
        return {"lat_grid": round(lat / step) * step, "lon_grid": round(lon / step) * step}

    def _format_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        # Apply anonymization and redaction
        out = dict(entry)
        meta = out.get("metadata", {}) or {}
        if "vehicle_hash" in meta:
            meta["vehicle_hash"] = self._hash_id(meta.get("vehicle_hash"))
        if "trip_id" in meta:
            meta["trip_id"] = self._hash_id(meta.get("trip_id"))
        out["metadata"] = meta

        # Location gridding if available
        features = out.get("features", {}) or {}
        if self.anonymize_location and "latitude" in features and "longitude" in features:
            try:
                lat = float(features.get("latitude"))
                lon = float(features.get("longitude"))
                grid = self._grid_location(lat, lon)
                out.setdefault("redacted_location", grid)
                # remove raw coords
                features.pop("latitude", None)
                features.pop("longitude", None)
            except Exception:
                pass

        return out

    def _flush(self) -> None:
        if not self._buffer:
            return
        if not self._current_file:
            # if logging disabled at runtime, drop buffer
            self._buffer = []
            return
        for e in self._buffer:
            try:
                line = json.dumps(self._format_entry(e), default=str)
                self._current_file.write(line + "\n")
            except Exception:
                # best-effort: skip problematic entries
                continue
        self._current_file.flush()
        os.fsync(self._current_file.fileno())
        # If encryption enabled, close plaintext temp, encrypt to .jsonl.enc, remove temp, and open new file
        if self.encrypt:
            try:
                try:
                    self._current_file.close()
                except Exception:
                    pass
                tmp_path = self._current_base + ".jsonl.tmp"
                enc_path = self._current_base + ".jsonl.enc"
                with open(tmp_path, "rb") as f:
                    data = f.read()
                enc = _encryption.encrypt_bytes(data, self._encryption_key)
                with open(enc_path, "wb") as f:
                    f.write(enc)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            except Exception:
                # If encryption failed, keep going but do not crash
                pass
            # open a new file for subsequent writes
            try:
                self._open_file()
            except Exception:
                self._current_file = None
        else:
            self._buffer = []
        # Always call rotate to clean up old files
        self._rotate()

    def _rotate(self) -> None:
        try:
            cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
            for fn in os.listdir(self.output_dir):
                if not fn.startswith("telemetry_"):
                    continue
                path = os.path.join(self.output_dir, fn)
                try:
                    ts = fn.split("_")[1].split(".")[0]
                    dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                    if dt < cutoff:
                        os.remove(path)
                except Exception:
                    continue
        except Exception:
            pass

    def _get_encryption_key(self) -> Optional[bytes]:
        # Try environment variable first (hex or base64), then config file
        env_name = str(self.config.get("telemetry.encryption_key_env", "TELEMETRY_ENCRYPTION_KEY"))
        val = os.environ.get(env_name)
        if val:
            # try hex
            try:
                return bytes.fromhex(val)
            except Exception:
                pass
            # try base64
            try:
                return base64.b64decode(val)
            except Exception:
                pass

        key_file = self.config.get("telemetry.encryption_key_file", None)
        if key_file:
            try:
                with open(str(key_file), "rb") as f:
                    data = f.read().strip()
                    # try hex
                    try:
                        return bytes.fromhex(data.decode("utf-8"))
                    except Exception:
                        pass
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        pass
            except Exception:
                return None
        return None

    def log_prediction(self, risk_score: float, alert_level: str, features: Dict[str, Any], sensor_health: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "prediction",
            "risk_score": float(risk_score),
            "alert_level": alert_level,
            "features": features,
            "sensor_health": sensor_health,
            "metadata": metadata or {},
        }
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def log_outcome(self, prediction_id: str, actual_outcome: str, time_to_event_s: Optional[float] = None, severity: Optional[str] = None) -> None:
        if not self.enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "outcome",
            "prediction_id": prediction_id,
            "actual_outcome": actual_outcome,
            "time_to_event_s": time_to_event_s,
            "severity": severity,
        }
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def log_false_alarm(self, prediction_id: str, reason: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "false_alarm",
            "prediction_id": prediction_id,
            "reason": reason,
            "context": context or {},
        }
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def log_battery_alert(self, alert: Dict[str, Any], telemetry_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Log a battery monitoring alert (rule or ML source).

        The alert dict is expected to contain: source, severity, reason, anomaly_score (optional).
        telemetry_snapshot may be a list/dict of recent BMS samples.
        """
        if not self.enabled:
            return
        # sanitize telemetry_snapshot metadata by hashing any vehicle/trip ids
        if telemetry_snapshot is not None:
            try:
                if isinstance(telemetry_snapshot, list):
                    for s in telemetry_snapshot:
                        if not isinstance(s, dict):
                            continue
                        meta = s.get("metadata") or {}
                        if "vehicle_hash" in meta:
                            meta["vehicle_hash"] = self._hash_id(meta.get("vehicle_hash"))
                        if "trip_id" in meta:
                            meta["trip_id"] = self._hash_id(meta.get("trip_id"))
                        s["metadata"] = meta
                elif isinstance(telemetry_snapshot, dict):
                    meta = telemetry_snapshot.get("metadata") or {}
                    if "vehicle_hash" in meta:
                        meta["vehicle_hash"] = self._hash_id(meta.get("vehicle_hash"))
                    if "trip_id" in meta:
                        meta["trip_id"] = self._hash_id(meta.get("trip_id"))
                    telemetry_snapshot["metadata"] = meta
            except Exception:
                # best-effort: do not fail logging due to snapshot sanitization
                pass

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "battery_alert",
            "source": alert.get("source", "rule"),
            "severity": alert.get("severity", "LOW"),
            "reason": alert.get("reason", ""),
            "anomaly_score": alert.get("anomaly_score", None),
            "telemetry_snapshot": telemetry_snapshot,
        }
        # anonymize vehicle identifiers if present in metadata or snapshot
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def log_hardware_check(self, results: Dict[str, Any]) -> None:
        """Log hardware diagnostics check results.

        Args:
            results: Dict from HardwareChecker.run_all_tests() with keys:
                     timestamp, expected_sensors, passed_tests, failed_tests,
                     skipped_tests, test_results.
        """
        if not self.enabled:
            return

        entry = {
            "timestamp": results.get("timestamp") or datetime.utcnow().isoformat(),
            "type": "hardware_check",
            "expected_sensors": results.get("expected_sensors", []),
            "passed_tests": results.get("passed_tests", []),
            "failed_tests": results.get("failed_tests", []),
            "skipped_tests": results.get("skipped_tests", []),
            "overall_status": ("OK" if not results.get("failed_tests") else "FAILED"),
        }

        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def get_stats(self) -> Dict[str, int]:
        # Lightweight stats from buffer; persisted files not counted here
        with self._lock:
            preds = sum(1 for e in self._buffer if e.get("type") == "prediction")
            outs = sum(1 for e in self._buffer if e.get("type") == "outcome")
            fas = sum(1 for e in self._buffer if e.get("type") == "false_alarm")
            bats = sum(1 for e in self._buffer if e.get("type") == "battery_alert")
            hw = sum(1 for e in self._buffer if e.get("type") == "hardware_check")
        return {"buffered_predictions": preds, "buffered_outcomes": outs, "buffered_false_alarms": fas, "buffered_battery_alerts": bats, "buffered_hardware_checks": hw}

    def close(self) -> None:
        with self._lock:
            try:
                if self._current_file:
                    self._flush()
            finally:
                try:
                    if self._current_file:
                        self._current_file.close()
                        self._current_file = None
                except Exception:
                    pass
