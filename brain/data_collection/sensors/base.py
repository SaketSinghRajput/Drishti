"""Base sensor handler abstractions."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from config.config_loader import ConfigLoader


@dataclass
class SensorHealthStatus:
    status: str = "unknown"
    last_successful_read: Optional[datetime] = None
    error_count: int = 0
    availability_percent: float = 0.0
    # counters for availability calculation
    total_reads: int = 0
    successful_reads: int = 0


class SensorHandler(ABC):
    """Abstract base class for sensor handlers.

    Implementations must be thread-safe and provide read(), get_health(),
    is_available(), and get_name().
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._health = SensorHealthStatus()

        # counters
        self._total_reads = 0
        self._successful_reads = 0

    def _record_read(self, success: bool) -> None:
        """Update counters and availability percentage."""
        self._total_reads += 1
        if success:
            self._successful_reads += 1
        # avoid division by zero
        if self._total_reads > 0:
            # store as percentage 0..100
            self._health.availability_percent = (self._successful_reads / self._total_reads) * 100.0
        else:
            self._health.availability_percent = 0.0

    @abstractmethod
    def read(self) -> Optional[Dict[str, Any]]:
        """Read sensor data and return a dict with timestamp and sensor fields."""

    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """Return health summary for this sensor."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the sensor is available for reading."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the sensor name identifier."""
