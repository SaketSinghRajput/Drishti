import threading
import copy
import time
from collections import deque
from typing import Dict, Optional, List


class DashboardState:
    """Thread-safe in-memory cache for real-time dashboard data."""

    def __init__(self, max_alerts: int = 200):
        self._lock = threading.Lock()
        self._latest_prediction = None
        self._alert_history = deque(maxlen=max_alerts)
        self._incident_cache = {}
        self._sensor_health = {}
        self._latest_fatigue = None
        self._fatigue_history = deque(maxlen=100)
        self._latest_ble = None
        self._ble_history = deque(maxlen=100)

    def update_latest_prediction(self, prediction: Dict):
        with self._lock:
            self._latest_prediction = copy.deepcopy(prediction)
            alert_level = prediction.get("alert_level")
            if alert_level in ("medium", "high", "critical"):
                entry = {
                    "timestamp": prediction.get("timestamp"),
                    "alert_level": alert_level,
                    "risk_score": prediction.get("risk_score"),
                    "reason": prediction.get("reason"),
                }
                self._alert_history.append(entry)

    def get_latest_prediction(self) -> Optional[Dict]:
        with self._lock:
            return copy.deepcopy(self._latest_prediction)

    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        with self._lock:
            items = list(self._alert_history)
        return items[-limit:]

    def add_incident(self, incident_id: str, metadata: Dict):
        with self._lock:
            self._incident_cache[incident_id] = metadata.copy()

    def get_incident(self, incident_id: str) -> Optional[Dict]:
        with self._lock:
            v = self._incident_cache.get(incident_id)
            return copy.deepcopy(v) if v is not None else None

    def update_sensor_health(self, sensor_health: Dict):
        with self._lock:
            self._sensor_health = sensor_health.copy()

    def get_sensor_health(self) -> Dict:
        with self._lock:
            return copy.deepcopy(self._sensor_health)

    def update_fatigue_metrics(self, fatigue_data: Dict):
        """Store latest fatigue metrics; expects a dict, adds timestamp if missing."""
        if not isinstance(fatigue_data, dict):
            fatigue_data = {}
        with self._lock:
            entry = copy.deepcopy(fatigue_data)
            ts = entry.get("timestamp") if isinstance(entry, dict) else None
            if ts is None:
                entry["timestamp"] = time.time()
            self._latest_fatigue = copy.deepcopy(entry)
            self._fatigue_history.append(entry)

    def update_ble_status(self, ble_data: Dict):
        """Store latest BLE status; expects a dict, adds timestamp if missing."""
        if not isinstance(ble_data, dict):
            ble_data = {}

        def _to_int_safe(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        with self._lock:
            entry = copy.deepcopy(ble_data)
            if not isinstance(entry.get("neighbors"), list):
                entry["neighbors"] = []
            if not isinstance(entry.get("neighbors_count"), int):
                entry["neighbors_count"] = len(entry["neighbors"])

            packet_stats = entry.get("packet_stats")
            if not isinstance(packet_stats, dict):
                packet_stats = {}
            entry["packet_stats"] = {
                "total_sent": _to_int_safe(packet_stats.get("total_sent", 0)) if packet_stats else 0,
                "total_received": _to_int_safe(packet_stats.get("total_received", 0)) if packet_stats else 0,
                "failed_sends": _to_int_safe(packet_stats.get("failed_sends", 0)) if packet_stats else 0,
            }
            ts = entry.get("timestamp") if isinstance(entry, dict) else None
            if not isinstance(ts, (int, float)):
                entry["timestamp"] = time.time()
            self._latest_ble = copy.deepcopy(entry)
            self._ble_history.append(entry)

    def get_latest_fatigue(self) -> Optional[Dict]:
        with self._lock:
            return copy.deepcopy(self._latest_fatigue)

    def get_fatigue_history(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            items = list(self._fatigue_history)
        return items[-limit:]

    def get_latest_ble(self) -> Optional[Dict]:
        with self._lock:
            return copy.deepcopy(self._latest_ble)

    def get_ble_history(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            items = list(self._ble_history)
        return items[-limit:]

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "latest_prediction_present": self._latest_prediction is not None,
                "alert_history_size": len(self._alert_history),
                "incident_cache_size": len(self._incident_cache),
                "latest_fatigue_present": self._latest_fatigue is not None,
                "latest_ble_present": self._latest_ble is not None,
            }

    def clear(self):
        with self._lock:
            self._latest_prediction = None
            self._alert_history.clear()
            self._incident_cache.clear()
            self._sensor_health.clear()
            self._latest_fatigue = None
            self._fatigue_history.clear()
            self._latest_ble = None
            self._ble_history.clear()
