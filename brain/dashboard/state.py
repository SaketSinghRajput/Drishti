import threading
import copy
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

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "latest_prediction_present": self._latest_prediction is not None,
                "alert_history_size": len(self._alert_history),
                "incident_cache_size": len(self._incident_cache),
            }

    def clear(self):
        with self._lock:
            self._latest_prediction = None
            self._alert_history.clear()
            self._incident_cache.clear()
            self._sensor_health.clear()
