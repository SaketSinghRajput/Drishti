"""Vehicle CAN handler integrated with SensorHandler and CAN ingestion.

Parses basic OBD-II PIDs (0x0D speed, 0x0C RPM, 0x11 throttle) when available,
and falls back to a simulation generator when `enable_simulation` is true.
"""
import time
import threading
import logging
from typing import Any, Dict, Optional

from config.config_loader import ConfigLoader
from data_collection.sensors.base import SensorHandler
from battery_monitor.ingestion.can_ingestion import CANIngestion

logger = logging.getLogger(__name__)


class VehicleCANHandler(SensorHandler):
    def __init__(self, config: ConfigLoader):
        # Accept either a ConfigLoader-like object (with dotted-key `get`) or a plain dict.
        self._raw_cfg = config or {}
        if hasattr(config, "get") and not isinstance(config, dict):
            cfg = config
        else:
            # plain dict fallback
            class _DictCfg:
                def __init__(self, d):
                    self._d = d or {}

                def get(self, k, default=None):
                    # support dotted keys like 'sensors.vehicle_can'
                    if isinstance(k, str) and "." in k:
                        parts = k.split(".")
                        v = self._d
                        for p in parts:
                            if isinstance(v, dict):
                                v = v.get(p)
                            else:
                                v = None
                                break
                        return v if v is not None else default
                    return self._d.get(k, default)

            cfg = _DictCfg(config or {})

        super().__init__(cfg)
        self._cfg = cfg
        # prefer dotted get first (ConfigLoader), fall back to dict nested lookup
        vcfg = cfg.get("sensors.vehicle_can", None)
        if vcfg is None and isinstance(self._raw_cfg, dict):
            vcfg = self._raw_cfg.get("sensors", {}).get("vehicle_can", {})
        vcfg = vcfg or {}
        self.interface = vcfg.get("can_interface", "can0")
        self.bitrate = int(vcfg.get("can_bitrate", 500000))
        self.enable_simulation = bool(vcfg.get("enable_simulation", True))
        self.obd_pids = list(vcfg.get("obd_pids", [0x0D, 0x0C, 0x11]))
        self.sample_rate = int(vcfg.get("sample_rate", 10))

        self._ingest = None
        self._last_sample: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

        if not self.enable_simulation:
            try:
                self._ingest = CANIngestion({"battery_monitor.ingestion.can_interface": self.interface, "battery_monitor.ingestion.can_bitrate": self.bitrate})
                self._ingest.start()
            except Exception:
                logger.exception("Failed to start CAN ingestion; falling back to simulation")
                self.enable_simulation = True

    def get_name(self) -> str:
        return "vehicle_can"

    def read(self) -> Optional[Dict[str, Any]]:
        ts = time.time()
        if self.enable_simulation:
            import random

            speed = max(0, 50 + random.gauss(0, 5))
            steering = random.uniform(-15.0, 15.0)
            brake = random.choice([0.0, 0.0, random.uniform(0, 30)])
            sample = {"timestamp": ts, "speed_kph": speed, "steering_angle": steering, "brake_pressure": brake}
            with self._lock:
                self._last_sample = sample
            self._record_read(True)
            return sample

        # read from CAN ingestion buffer
        if self._ingest is None:
            self._record_read(False)
            return None

        msg = self._ingest.read_message(timeout=0.1)
        if msg is None:
            self._record_read(False)
            return None

        can_id, data = msg
        try:
            # OBD-II response format: [0x41, PID, payload...]
            if len(data) >= 3 and data[0] == 0x41:
                pid = data[1]
                if pid == 0x0D:
                    speed = data[2]
                    sample = {"timestamp": ts, "speed_kph": float(speed)}
                elif pid == 0x0C and len(data) >= 4:
                    A = data[2]
                    B = data[3]
                    rpm = ((A * 256) + B) / 4.0
                    sample = {"timestamp": ts, "engine_rpm": float(rpm)}
                elif pid == 0x11:
                    th = data[2]
                    throttle = float(th) * 100.0 / 255.0
                    sample = {"timestamp": ts, "throttle_percent": throttle}
                else:
                    # Unhandled PID, record raw
                    sample = {"timestamp": ts, "raw_can_id": can_id, "raw_data": list(data)}
            else:
                sample = {"timestamp": ts, "raw_can_id": can_id, "raw_data": list(data)}
            with self._lock:
                self._last_sample = sample
            self._record_read(True)
            return sample
        except Exception:
            logger.exception("Failed to parse CAN message")
            self._record_read(False)
            return None

    def get_health(self) -> Dict[str, Any]:
        with self._lock:
            last = self._last_sample
        return {"status": "ok" if last is not None else "unknown", "last_seen": (last or {}).get("timestamp"), "available": last is not None}

    def is_available(self) -> bool:
        return self.enable_simulation or (self._ingest is not None and self._ingest.get_status().get("available", False))

