"""BLEMeshHandler: Simulated BLE neighbor discovery handler.

This handler simulates BLE neighbor discovery for development and testing.
A production implementation should replace this with platform-specific BLE
scanning (e.g. BlueZ/pybluez/bleak) and properly authenticated neighbor IDs.

The handler returns dicts with keys:
 - timestamp: ISO8601 UTC
 - neighbors: list of neighbor snapshots, each with id_hash, rssi, dist_m,
   rel_heading, rel_speed_mps, last_seen_ms

The handler maintains simple internal state to simulate nearby nodes, uses an
exponential moving average for RSSI smoothing, and enforces max_neighbors and
RSSI thresholds from config.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Dict, List, Optional

from config.config_loader import ConfigLoader
from data_collection.sensors.base import SensorHandler
from utils.time_utils import get_utc_timestamp
from utils.ble_utils import rssi_to_distance, smooth_rssi, hash_node_id

logger = logging.getLogger(__name__)


class BLEMeshHandler(SensorHandler):
    """Simulated BLE mesh neighbor discovery handler.

    Config keys used from `sensors.ble_mesh`:
      - sample_rate (Hz)
      - scan_interval_ms
      - rssi_threshold
      - max_neighbors
      - distance_calibration_params
      - smoothing_alpha
      - node_id_salt
    """

    def __init__(self, config: ConfigLoader) -> None:
        super().__init__(config)
        self._cfg = config
        sect = "sensors.ble_mesh"
        self.sample_rate = float(self._cfg.get(f"{sect}.sample_rate", 2))
        self.scan_interval_ms = int(self._cfg.get(f"{sect}.scan_interval_ms", 500))
        self.rssi_threshold = float(self._cfg.get(f"{sect}.rssi_threshold", -80))
        self.max_neighbors = int(self._cfg.get(f"{sect}.max_neighbors", 20))
        self.calib = self._cfg.get(f"{sect}.distance_calibration_params", {"a": -59, "b": 2.5})
        self.smoothing_alpha = float(self._cfg.get(f"{sect}.smoothing_alpha", 0.3))
        self.rssi_history_len = int(self._cfg.get(f"{sect}.rssi_history_len", 50))
        self.node_id_salt = str(self._cfg.get(f"{sect}.node_id_salt", ""))

        # internal neighbor state: id -> dict(rssi_history:list, last_t: float, rel_heading, rel_speed_mps)
        self._neighbors: Dict[str, Dict] = {}
        self._lock = threading.Lock()

        # health counters
        self._total_reads = 0
        self._successful_reads = 0

        # initialize with a few neighbors (respect max_neighbors)
        init_count = min(random.randint(1, 6), self.max_neighbors)
        for i in range(init_count):
            nid = f"node_{i}"
            hid = hash_node_id(nid, self.node_id_salt)
            self._neighbors[hid] = {
                "rssi_history": [random.uniform(-70, -50)],
                "last_seen_t": time.time(),
                "rel_heading": random.uniform(0, 360),
                "rel_speed_mps": random.uniform(-5, 5),
            }

    def get_name(self) -> str:
        return "ble_mesh"

    def is_available(self) -> bool:
        # For simulation, always available unless disabled
        return True

    def get_health(self) -> Dict:
        # report basic health and neighbor count
        with self._lock:
            neighbor_count = len(self._neighbors)
            availability = (self._successful_reads / self._total_reads) if self._total_reads else 0.0
        return {"status": "ok", "availability": availability, "neighbor_count": neighbor_count}

    def read(self) -> Optional[Dict]:
        # simulate a scan
        self._total_reads += 1
        now = time.time()
        try:
            with self._lock:
                # random churn: occasionally add or remove a neighbor
                if random.random() < 0.2 and len(self._neighbors) < self.max_neighbors:
                    nid = f"node_{int(now) % 10000}_{random.randint(0,999)}"
                    hid = hash_node_id(nid, self.node_id_salt)
                    self._neighbors[hid] = {
                        "rssi_history": [random.uniform(-80, -50)],
                        "last_seen_t": now,
                        "rel_heading": random.uniform(0, 360),
                        "rel_speed_mps": random.uniform(-5, 5),
                    }
                if random.random() < 0.1 and self._neighbors:
                    # remove a random neighbor
                    k = random.choice(list(self._neighbors.keys()))
                    del self._neighbors[k]

                neighbors_out: List[Dict] = []
                for hid, st in list(self._neighbors.items()):
                    # simulate new RSSI reading
                    raw_rssi = st.get("rssi_history", [random.uniform(-80, -50)])[-1] + random.uniform(-3, 3)
                    st.setdefault("rssi_history", []).append(raw_rssi)
                    # apply smoothing using history (previous values)
                    smoothed = smooth_rssi(raw_rssi, st.get("rssi_history", [])[:-1], alpha=self.smoothing_alpha)
                    # append smoothed into history as the recent representation
                    st["rssi_history"][-1] = smoothed
                    # cap history length
                    if len(st["rssi_history"]) > self.rssi_history_len:
                        st["rssi_history"] = st["rssi_history"][-self.rssi_history_len :]

                    # compute distance from RSSI
                    dist_m = rssi_to_distance(smoothed, self.calib)

                    # determine if neighbor is seen on this scan (meets RSSI threshold)
                    seen = smoothed >= self.rssi_threshold

                    # compute last_seen_ms based on previously recorded last_seen_t
                    last_seen_t = st.get("last_seen_t", 0.0)
                    last_seen_ms = int((now - last_seen_t) * 1000) if last_seen_t else int(1e9)

                    # if seen now, update last_seen_t and set last_seen_ms to 0
                    if seen:
                        st["last_seen_t"] = now
                        last_seen_ms = 0

                    neighbors_out.append(
                        {
                            "id_hash": hid,
                            "rssi": round(smoothed, 2),
                            "dist_m": round(dist_m, 2),
                            "rel_heading": round(st.get("rel_heading", 0.0), 1),
                            "rel_speed_mps": round(st.get("rel_speed_mps", 0.3), 2),
                            "last_seen_ms": int(last_seen_ms),
                        }
                    )

                # enforce max_neighbors
                neighbors_out = neighbors_out[: self.max_neighbors]

            self._successful_reads += 1
            return {"timestamp": get_utc_timestamp(), "neighbors": neighbors_out}
        except Exception as e:
            logger.exception("BLEMeshHandler read failed: %s", e)
            return None
