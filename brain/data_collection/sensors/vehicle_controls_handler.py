"""Simulated vehicle controls handler (steering, brake, throttle)."""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .base import SensorHandler
from utils.time_utils import get_utc_timestamp


class VehicleControlsHandler(SensorHandler):
    def __init__(self, config):
        super().__init__(config)
        self.sample_rate = int(self.config.get("sensors.steering.sample_rate", 20))
        self._steering = 0.0
        self._brake_pressure = 0.0
        self._brake_flag = False
        self._throttle = 0.0

    def get_name(self) -> str:
        return "vehicle_controls"

    def read(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            try:
                # simulate driver inputs
                self._steering += random.uniform(-5, 5)
                # keep steering in plausible bounds
                self._steering = max(-720, min(720, self._steering))
                # throttle and brakes are inversely correlated in this simple sim
                if random.random() < 0.05:
                    # braking event
                    self._brake_flag = True
                    self._brake_pressure = random.uniform(0.5, 1.0)
                    self._throttle = random.uniform(0.0, 0.2)
                else:
                    self._brake_flag = False
                    self._brake_pressure = 0.0
                    self._throttle = random.uniform(0.0, 1.0)

                self._health.status = "ok"
                self._health.last_successful_read = datetime.now(timezone.utc)
                self._record_read(True)
                return {
                    "steering_angle": round(self._steering, 2),
                    "brake_pressure": round(self._brake_pressure, 3),
                    "brake_flag": bool(self._brake_flag),
                    "throttle_position": round(self._throttle, 3),
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
