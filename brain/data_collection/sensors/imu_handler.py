"""Simulated IMU sensor handler."""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .base import SensorHandler
from utils.time_utils import get_utc_timestamp


class IMUHandler(SensorHandler):
    def __init__(self, config):
        super().__init__(config)
        self.sample_rate = int(self.config.get("sensors.imu.sample_rate", 50))
        self._yaw_rate = 0.0
        self._pitch = 0.0
        self._roll = 0.0
        self._ax = 0.0
        self._ay = 0.0
        self._az = 9.8

    def get_name(self) -> str:
        return "imu"

    def read(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            try:
                # small variations with occasional larger events
                self._yaw_rate += random.uniform(-2.0, 2.0)
                self._pitch += random.uniform(-0.5, 0.5)
                self._roll += random.uniform(-0.5, 0.5)
                self._ax = random.uniform(-3, 3)
                self._ay = random.uniform(-3, 3)
                self._az = 9.8 + random.uniform(-0.3, 0.3)
                self._health.status = "ok"
                self._health.last_successful_read = datetime.now(timezone.utc)
                self._record_read(True)
                return {
                    "yaw_rate": round(self._yaw_rate, 3),
                    "pitch": round(self._pitch, 3),
                    "roll": round(self._roll, 3),
                    "ax": round(self._ax, 3),
                    "ay": round(self._ay, 3),
                    "az": round(self._az, 3),
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
