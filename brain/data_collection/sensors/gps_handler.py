"""Simulated GPS sensor handler."""
from __future__ import annotations

import random
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .base import SensorHandler
from utils.time_utils import get_utc_timestamp


class GPSHandler(SensorHandler):
    def __init__(self, config):
        super().__init__(config)
        self.sample_rate = int(self.config.get("sensors.gps.sample_rate", 1))
        self.accuracy_threshold = float(self.config.get("sensors.gps.accuracy_threshold", 10.0))
        # start at a base location
        self._lat = 37.0
        self._lon = -122.0
        self._speed = 0.0
        self._heading = 0.0

    def get_name(self) -> str:
        return "gps"

    def read(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            try:
                # simple random-walk simulation
                self._speed = max(0.0, self._speed + random.uniform(-1.0, 1.0))
                self._heading = (self._heading + random.uniform(-5, 5)) % 360
                self._lat += random.uniform(-0.0001, 0.0001)
                self._lon += random.uniform(-0.0001, 0.0001)
                self._health.status = "ok"
                self._health.last_successful_read = datetime.now(timezone.utc)
                self._record_read(True)
                return {
                    "lat": self._lat,
                    "lon": self._lon,
                    "speed_kph": round(self._speed, 2),
                    "heading": round(self._heading, 2),
                    "accuracy_m": round(random.uniform(1.0, self.accuracy_threshold), 1),
                    "timestamp": get_utc_timestamp(),
                }
            except Exception:
                self._health.error_count += 1
                self._health.status = "error"
                self._record_read(False)
                return None

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": self._health.status,
            "last_successful_read": self._health.last_successful_read.isoformat() if self._health.last_successful_read else None,
            "error_count": self._health.error_count,
            "availability": self._health.availability_percent,
        }

    def is_available(self) -> bool:
        return self._health.status != "error"
