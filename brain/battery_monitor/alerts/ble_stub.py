"""BLEBroadcastStub: simulate BLE broadcasts by writing compact JSON packets to a file.

Designed for development/testing where real BLE hardware isn't available.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

import hashlib


class BLEBroadcastStub:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        alerts_cfg = config.get("battery_monitor", {}).get("alerts", {})
        # use human-readable JSON file by default
        self.outbox_path = str(alerts_cfg.get("ble_outbox_path", "data/ble_outbox.json"))
        os.makedirs(os.path.dirname(self.outbox_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        # open file lazily on broadcast to avoid locking during init

    def _hash_id(self, vid: Optional[str]) -> Optional[str]:
        if vid is None:
            return None
        salt = str(self.config.get("privacy", {}).get("node_id_salt", ""))
        return hashlib.sha256((salt + vid).encode("utf-8")).hexdigest()

    def broadcast(self, alert: Dict[str, Any], vehicle_id: Optional[str] = None) -> None:
        # vehicle_id should already be hashed by caller, but allow raw ID too
        if vehicle_id is not None and len(vehicle_id) < 32:
            vehicle_id = self._hash_id(vehicle_id)

        # canonical human-readable BLE outbox packet
        packet = {
            "timestamp": int(time.time()),
            "vehicle_id": vehicle_id or "unknown",
            "severity": alert.get("severity"),
            "reason": alert.get("reason"),
            "anomaly_score": alert.get("anomaly_score"),
        }

        # write as a single JSON object to a .json file (append-friendly)
        with self._lock:
            with open(self.outbox_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(packet, default=str) + "\n")

    def close(self) -> None:
        # nothing to close, file is opened per-write
        return
