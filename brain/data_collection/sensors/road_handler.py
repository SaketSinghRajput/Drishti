"""Road sensor handler (simulation-first).

Provides simulated road geometry and signage information: curvature_radius_m,
speed_limit_kph, upcoming_turn, intersection_flag, slope_percent, timestamp.
Designed to be thread-safe and config-driven. Future integrations may use map
APIs or CAN bus data when enable_simulation is False.

The `upcoming_turn` field may be one of: 'none', 'left', 'right', 'sharp'.
When a new simulated segment is created the handler will occasionally mark an
upcoming sharp turn when the curvature radius falls below
`sensors.road.sharp_turn_curvature_threshold_m` (configured in meters).
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Dict, Optional

from config.config_loader import ConfigLoader
from data_collection.sensors.base import SensorHandler
from utils.time_utils import get_utc_timestamp

logger = logging.getLogger(__name__)


class RoadHandler(SensorHandler):
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.cfg = config.get("sensors.road", {})
        self.sample_rate = float(self.cfg.get("sample_rate", 1))
        self.enable_simulation = bool(self.cfg.get("enable_simulation", True))
        self._lock = threading.Lock()
        # simulation state
        self._curvature = float(self.cfg.get("default_curvature_radius_m", 1000.0))
        self._speed_limit = float(self.cfg.get("default_speed_limit_kph", 50.0))
        self._upcoming_turn = "none"
        self._intersection = False
        self._slope = 0.0
        self._segment_start = time.time()
        self._segment_duration = float(self.cfg.get("simulation_segment_duration_s", 30))
        # sharp turn tuning
        self._sharp_turn_curvature_threshold = float(self.cfg.get("sharp_turn_curvature_threshold_m", 50.0))
        self._sharp_turn_probability = float(self.cfg.get("sharp_turn_probability", 0.5))

    def get_name(self) -> str:
        return "road"

    def is_available(self) -> bool:
        return True

    def get_health(self) -> Dict[str, Any]:
        return {"status": "ok", "availability": 1.0}

    def _maybe_update_segment(self) -> None:
        now = time.time()
        if now - self._segment_start > self._segment_duration:
            # transition to a new segment
            with self._lock:
                # curvature between 50 (sharp) and 2000 (straight)
                self._curvature = float(max(20.0, random.gauss(self._curvature, 100.0)))
                self._speed_limit = float(random.choice([30, 40, 50, 60, 80, 100]))
                p = random.random()
                if p < float(self.cfg.get("intersection_probability", 0.1)):
                    self._upcoming_turn = random.choice(["left", "right"])
                    self._intersection = True
                else:
                    # default to no upcoming turn
                    self._upcoming_turn = "none"
                    self._intersection = False
                # if the new curvature is below the configured sharp threshold,
                # there is a chance this segment represents an upcoming sharp turn
                try:
                    if self._curvature < float(self._sharp_turn_curvature_threshold) and random.random() < float(self._sharp_turn_probability):
                        self._upcoming_turn = "sharp"
                except Exception:
                    # be tolerant of bad config values
                    pass
                # slope between -10 and 10 percent
                self._slope = float(random.uniform(-3.0, 6.0))
                self._segment_start = now

    def read(self) -> Optional[Dict[str, Any]]:
        ts = get_utc_timestamp()
        if not self.enable_simulation:
            logger.debug("RoadHandler production mode requested but not implemented; using simulation")
        self._maybe_update_segment()
        with self._lock:
            curvature = float(self._curvature)
            speed_limit = float(self._speed_limit)
            upcoming_turn = self._upcoming_turn
            intersection = bool(self._intersection)
            slope = float(self._slope)
        return {
            "timestamp": ts,
            "curvature_radius_m": curvature,
            "speed_limit_kph": speed_limit,
            "upcoming_turn": upcoming_turn,
            "intersection_flag": intersection,
            "slope_percent": slope,
        }
