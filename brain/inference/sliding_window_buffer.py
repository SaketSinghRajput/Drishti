"""Sliding window buffer optimized for low-latency inference.

This buffer stores per-sensor deques of recent readings (timestamped) and
provides snapshot windows matching the project's sensor window schema.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class SlidingWindowBuffer:
    """Lightweight sliding window buffer for inference.

    Differences from data_collection.window_manager.WindowManager:
      - In-memory only (no file I/O)
      - Optimized for single-window snapshot extraction
      - Thread-safe via internal Lock
      - Minimal copying for low latency
    """

    def __init__(self, config: ConfigLoader, vehicle_hash: Optional[str] = None, trip_id: Optional[str] = None):
        self.config = config
        self.window_size_s = float(self.config.get("inference.window_size_s", 5))
        self.buffer_max_age_s = float(self.config.get("inference.buffer_max_age_s", 10))
        self.expected_sensors: List[str] = list(self.config.get("data_collection.expected_sensors", []))
        self._vehicle_hash = vehicle_hash
        self._trip_id = trip_id

        # Estimate maxlen from sample rates; default to 200 entries if missing
        self._maxlen_map = {}
        for s in self.expected_sensors:
            rate = self.config.get(f"sensors.{s}.sample_rate", None)
            try:
                rate = float(rate) if rate is not None else None
            except Exception:
                rate = None
            if rate and rate > 0:
                maxlen = int(self.window_size_s * rate) + 2
            else:
                maxlen = 200
            self._maxlen_map[s] = maxlen

        self._lock = threading.Lock()
        self._deques: Dict[str, collections.deque] = {s: collections.deque(maxlen=self._maxlen_map[s]) for s in self.expected_sensors}
        self._last_update: Dict[str, float] = {s: 0.0 for s in self.expected_sensors}

    def add_reading(self, sensor_name: str, data: Dict[str, Any]) -> None:
        """Append a timestamped reading for sensor_name.

        If timestamp missing in data, current time is used.
        Unknown sensors are tolerated and stored in a dynamic deque.
        """
        ts = data.get("timestamp", time.time())
        entry = dict(data)
        entry.setdefault("timestamp", ts)

        with self._lock:
            if sensor_name not in self._deques:
                # Allow unknown sensors but limit size
                self._deques[sensor_name] = collections.deque(maxlen=200)
                self._last_update[sensor_name] = 0.0
            self._deques[sensor_name].append(entry)
            self._last_update[sensor_name] = ts

            # Evict older-than-window entries for this sensor
            cutoff = ts - self.window_size_s
            dq = self._deques[sensor_name]
            while dq and dq[0]["timestamp"] < cutoff:
                dq.popleft()

    def get_current_window(self) -> Dict[str, Any]:
        """Return a snapshot dict with per-sensor lists and metadata fields.

        Returns shallow copies of lists to avoid deep copy overhead.
        """
        with self._lock:
            now = time.time()
            window_id = str(uuid.uuid4())
            window = {s: list(self._deques.get(s, [])) for s in self._deques.keys()}
            # Top-level timestamp/window_id/trip_id and nested metadata per schema
            window["timestamp"] = now
            window["window_id"] = window_id
            window["trip_id"] = self._trip_id
            window["metadata"] = {"vehicle_hash": self._vehicle_hash}
            sensor_health = self._compute_sensor_health(now)
            window["sensor_health_snapshot"] = sensor_health
        return window

    def _compute_sensor_health(self, now: float) -> Dict[str, Any]:
        health = {}
        for s, last in self._last_update.items():
            expected_interval = None
            rate = self.config.get(f"sensors.{s}.sample_rate", None)
            try:
                if rate is not None:
                    expected_interval = 1.0 / float(rate)
            except Exception:
                expected_interval = None
            stale = False
            if last <= 0:
                stale = True
            elif expected_interval is not None:
                if now - last > max(2 * expected_interval, 1.0):
                    stale = True
            health[s] = {"last_update": last, "stale": stale}
        return health

    def is_ready(self) -> bool:
        """Return True when minimum required sensors have at least one sample."""
        with self._lock:
            for s in self.expected_sensors:
                dq = self._deques.get(s)
                if not dq or len(dq) == 0:
                    return False
            return True

    def clear(self) -> None:
        """Reset buffer state (useful for trip boundaries)."""
        with self._lock:
            for s in list(self._deques.keys()):
                self._deques[s].clear()
                self._last_update[s] = 0.0

    def get_sensor_health_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._compute_sensor_health(time.time()))

    def get_buffer_size(self) -> Dict[str, int]:
        with self._lock:
            return {s: len(self._deques.get(s, [])) for s in self._deques.keys()}