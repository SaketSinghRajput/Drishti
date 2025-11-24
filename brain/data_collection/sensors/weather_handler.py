"""Weather sensor handler (simulation-first).

Provides simulated weather samples: precip_type, precip_intensity_mm_hr, visibility_m,
ambient_light_level, road_surface, timestamp. Designed to be thread-safe and
config-driven. Future integrations can replace simulation with real API calls
by setting `enable_simulation: false` and providing API endpoints/keys.
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


class WeatherHandler(SensorHandler):
    """Simulated weather sensor.

    Emits dicts with keys:
      - timestamp
      - precip_type: one of 'none','rain','snow'
      - precip_intensity_mm_hr: float
      - visibility_m: float
      - ambient_light_level: string or numeric
      - road_surface: one of 'dry','wet','oily','icy'
    """

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.cfg = config.get("sensors.weather", {})
        self.sample_rate = float(self.cfg.get("sample_rate", 0.1))
        self.enable_simulation = bool(self.cfg.get("enable_simulation", True))
        self._lock = threading.Lock()
        # simulation state
        self._state = {
            "precip_type": "none",
            "precip_intensity_mm_hr": 0.0,
            "visibility_m": float(self.cfg.get("default_visibility_m", 5000)),
            "ambient_light_level": "day",
            "road_surface": "dry",
        }
        self._last_transition = time.time()
        self._transition_prob = float(self.cfg.get("simulation_transition_probability", 0.05))

    def get_name(self) -> str:
        return "weather"

    def is_available(self) -> bool:
        return True

    def get_health(self) -> Dict[str, Any]:
        return {"status": "ok", "availability": 1.0}

    def _maybe_transition(self) -> None:
        """Occasionally change weather state based on transition probability."""
        with self._lock:
            if random.random() < self._transition_prob:
                # choose transition
                p = random.random()
                if p < 0.6:
                    self._state["precip_type"] = "none"
                    self._state["precip_intensity_mm_hr"] = 0.0
                    self._state["visibility_m"] = float(self.cfg.get("default_visibility_m", 5000))
                    self._state["road_surface"] = "dry"
                elif p < 0.85:
                    self._state["precip_type"] = "rain"
                    self._state["precip_intensity_mm_hr"] = round(random.uniform(0.1, 10.0), 2)
                    self._state["visibility_m"] = max(50.0, self._state["visibility_m"] * float(self.cfg.get("rain_visibility_reduction", 0.5)))
                    self._state["road_surface"] = "wet"
                else:
                    self._state["precip_type"] = "snow"
                    self._state["precip_intensity_mm_hr"] = round(random.uniform(0.1, 5.0), 2)
                    self._state["visibility_m"] = max(20.0, self._state["visibility_m"] * float(self.cfg.get("snow_visibility_reduction", 0.3)))
                    self._state["road_surface"] = "icy"
                # ambient light may vary by time
                hour = int(time.gmtime().tm_hour)
                if 6 <= hour < 18:
                    self._state["ambient_light_level"] = "day"
                else:
                    self._state["ambient_light_level"] = "night"
                self._last_transition = time.time()

    def read(self) -> Optional[Dict[str, Any]]:
        ts = get_utc_timestamp()
        if not self.enable_simulation:
            # production integration point: call external API or hardware sensor
            # Not implemented in this phase; fall back to simulation
            logger.debug("WeatherHandler production mode requested but not implemented; using simulation")
        # simulate
        self._maybe_transition()
        with self._lock:
            s = dict(self._state)
        return {
            "timestamp": ts,
            "precip_type": s.get("precip_type", "none"),
            "precip_intensity_mm_hr": float(s.get("precip_intensity_mm_hr", 0.0)),
            "visibility_m": float(s.get("visibility_m", float(self.cfg.get("default_visibility_m", 5000)))),
            "ambient_light_level": s.get("ambient_light_level", "day"),
            "road_surface": s.get("road_surface", "dry"),
        }
