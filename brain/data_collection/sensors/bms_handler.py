"""BMSHandler sensor integrating CSV simulation and CAN ingestion.

Extends SensorHandler and provides read(), health tracking, and graceful
fallback between simulation and CAN modes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .base import SensorHandler
from config.config_loader import ConfigLoader
from utils.validation import validate_bms_telemetry
from utils.time_utils import get_utc_timestamp

logger = logging.getLogger(__name__)


class BMSHandler(SensorHandler):
    def __init__(self, config: ConfigLoader):
        super().__init__(config)
        self.config = config
        self.sample_rate = int(self.config.get('battery_monitor.ingestion.sample_rate', 1))
        self.enable_simulation = bool(self.config.get('battery_monitor.ingestion.enable_simulation', True))
        self._can = None
        self._parser = None
        self._sim = None
        self._can_connected = False
        self._parse_error_count = 0
        self._last_valid_sample = None

        # Initialize either simulator or CAN ingestion
        try:
            if self.enable_simulation:
                from battery_monitor.ingestion.csv_simulator import BMSCSVSimulator

                self._sim = BMSCSVSimulator(self.config)
                # generate a short sample if file missing
                try:
                    self._sim.generate_sample(duration_s=10)
                except Exception:
                    pass
            else:
                from battery_monitor.ingestion.can_ingestion import CANIngestion
                from battery_monitor.ingestion.bms_parser import BMSCANParser

                self._can = CANIngestion(self.config)
                self._parser = BMSCANParser(self.config)
                try:
                    self._can.start()
                    self._can_connected = True
                except Exception:
                    self._can_connected = False
        except Exception as e:
            logger.exception('Failed to initialize BMS handler: %s', e)

    def get_name(self) -> str:
        return 'bms'

    def read(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            try:
                if self.enable_simulation and self._sim:
                    sample = self._sim.read_next_sample()
                    if not sample:
                        # try to rewind and read again for long-running simulation
                        try:
                            self._sim.reset()
                            sample = self._sim.read_next_sample()
                        except Exception:
                            sample = None
                    if not sample:
                        # don't mark immediate error; allow transient EOF
                        self._record_read(False)
                        # only escalate to error after multiple failures
                        if self._health.error_count > 5:
                            self._health.status = 'error'
                        return None

                    ok, errs = validate_bms_telemetry(sample)
                    if not ok:
                        self._parse_error_count += 1
                        self._record_read(False)
                        self._health.status = 'error'
                        return None

                    self._health.status = 'ok'
                    self._health.last_successful_read = datetime.now(timezone.utc)
                    self._record_read(True)
                    self._last_valid_sample = sample
                    return sample
                else:
                    if not self._can:
                        self._record_read(False)
                        self._health.status = 'error'
                        return None

                    # refresh CAN availability
                    try:
                        status = self._can.get_status()
                        self._can_connected = bool(status.get('available'))
                    except Exception:
                        self._can_connected = False

                    msg = self._can.read_message(timeout=0.5)
                    if not msg:
                        # if repeated failures, update can_connected
                        self._record_read(False)
                        self._health.error_count += 1
                        if self._health.error_count > 5:
                            self._can_connected = False
                        return None

                    can_id, data = msg
                    raw = self._parser.parse_message(can_id, data)
                    # Attempt to obtain a recent power-bus sample from an attached VoltageCurrentHandler
                    power_bus_data = None
                    try:
                        vch = getattr(self, '_vch', None)
                        if vch is None:
                            # maybe set by DataLogger after initialization (best-effort lookup)
                            vch = None
                        if vch is not None and hasattr(vch, 'read'):
                            try:
                                power_bus_data = vch.read()
                            except Exception:
                                power_bus_data = None
                    except Exception:
                        power_bus_data = None

                    norm = self._parser.normalize_telemetry(raw, power_bus_data=power_bus_data)
                    ok, errs = validate_bms_telemetry(norm)
                    if not ok:
                        self._parse_error_count += 1
                        self._record_read(False)
                        self._health.status = 'error'
                        return None

                    self._health.status = 'ok'
                    self._health.last_successful_read = datetime.now(timezone.utc)
                    self._record_read(True)
                    self._last_valid_sample = norm
                    return norm
            except Exception as e:
                logger.exception('BMSHandler read failed: %s', e)
                self._health.error_count += 1
                self._record_read(False)
                self._health.status = 'error'
                return None

    def get_health(self) -> Dict[str, Any]:
        return {
            'status': self._health.status,
            'last_successful_read': self._health.last_successful_read.isoformat() if self._health.last_successful_read else None,
            'error_count': self._health.error_count,
            'availability': self._health.availability_percent,
            'can_connected': self._can_connected,
            'parse_error_count': self._parse_error_count,
        }

    def is_available(self) -> bool:
        return self._health.status != 'error' and (self.enable_simulation or self._can_connected)

    def stop(self) -> None:
        try:
            if self._can:
                self._can.stop()
        except Exception:
            pass
