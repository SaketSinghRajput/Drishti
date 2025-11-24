"""Sliding window manager for aggregating sensor readings into windows."""
from __future__ import annotations

import threading
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from config.config_loader import ConfigLoader
from utils.time_utils import get_utc_timestamp, parse_timestamp
from utils.validation import compute_data_quality_score

# Default expected sensors; can be overridden via config key
DEFAULT_EXPECTED_SENSORS = ["gps", "imu", "vehicle_controls", "driver_fatigue"]


class WindowManager:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.window_size_s = float(self.config.get("data_collection.window_size_s", 5))
        self.window_interval_s = float(self.config.get("data_collection.window_interval_s", 1))
        self._lock = threading.Lock()
        self._buffer: Dict[str, deque] = {}
        # use timezone-aware UTC datetimes to avoid naive/aware comparisons
        self._last_flush = datetime.now(timezone.utc)
        # expected sensors from config or default
        self.expected_sensors = list(self.config.get("data_collection.expected_sensors", DEFAULT_EXPECTED_SENSORS))

    def add_reading(self, sensor_name: str, data: Dict[str, Any]) -> None:
        ts = data.get("timestamp")
        with self._lock:
            if sensor_name not in self._buffer:
                self._buffer[sensor_name] = deque()
            # append new reading and trim any entries older than window_size_s
            self._buffer[sensor_name].append(data)
            try:
                cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_size_s)
                # remove old entries from the left while they are older than cutoff
                dq = self._buffer[sensor_name]
                while dq and parse_timestamp(dq[0].get("timestamp")) < cutoff:
                    dq.popleft()
            except Exception:
                # conservative: if parsing fails, keep the data; trimming can occur later
                pass

    def should_flush(self) -> bool:
        with self._lock:
            elapsed = (datetime.now(timezone.utc) - self._last_flush).total_seconds()
            return elapsed >= self.window_interval_s

    def get_current_window(self, sensor_health_snapshot: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Return a window containing lists of samples per sensor covering the last window_size_s seconds.

        If `sensor_health_snapshot` is provided it will be attached under `metadata.sensor_health`.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_size_s)
        window: Dict[str, Any] = {}
        with self._lock:
            for sensor, dq in list(self._buffer.items()):
                # trim old samples
                try:
                    while dq and parse_timestamp(dq[0].get("timestamp")) < cutoff:
                        dq.popleft()
                except Exception:
                    # if a timestamp cannot be parsed just keep items; validation will catch later
                    pass

                # include all samples within the window (ordered by timestamp)
                if dq:
                    # copy to avoid external mutation
                    window[sensor] = [s.copy() for s in list(dq)]

            # metadata
            window["window_id"] = str(uuid.uuid4())
            window["timestamp"] = get_utc_timestamp()
            # attach sensor health snapshot if provided
            metadata: Dict[str, Any] = {}
            metadata["sensor_health"] = sensor_health_snapshot or {}
            # determine missing sensors
            missing = [s for s in self.expected_sensors if s not in window]
            metadata["missing_sensors"] = missing
            # data quality score (0..1)
            metadata["data_quality_score"] = compute_data_quality_score(window, self.expected_sensors)

            window["metadata"] = metadata

            # update last flush to timezone-aware UTC
            self._last_flush = datetime.now(timezone.utc)

        return window

    def clear_window(self) -> None:
        with self._lock:
            for dq in self._buffer.values():
                dq.clear()
