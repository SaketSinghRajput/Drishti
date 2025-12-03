"""VoltageCurrentHandler: pack-level voltage/current/temperature via I2C ADC or CSV simulation.

Simulation-first handler. In production it attempts to import CircuitPython
drivers for INA219 (voltage/current) and ADS1115 (thermistor ADC). If those
libraries or hardware are not available the handler falls back to CSV simulation.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config.config_loader import ConfigLoader
from data_collection.sensors.base import SensorHandler
from utils.time_utils import get_utc_timestamp

logger = logging.getLogger(__name__)

try:
    # Prefer the lightweight utils wrapper which centralizes the optional
    # CircuitPython imports. This keeps the handler code simpler and allows
    # other modules to reuse the same availability flag.
    from utils.voltage_current_handler import (
        _HAS_CIRCUIT as _has_circuit,
        ADCResolution,
        INA219,
        ADS1115,
        AnalogIn,
    )
except Exception:
    # Fallback: mark circuit libraries as unavailable
    _has_circuit = False
    ADCResolution = None
    INA219 = None
    ADS1115 = None
    AnalogIn = None


class VoltageCurrentHandler(SensorHandler):
    def __init__(self, config: ConfigLoader):
        super().__init__(config)
        self.enable = bool(self.config.get('sensors.voltage_current.enable', False))
        self.enable_sim = bool(self.config.get('sensors.voltage_current.enable_simulation', True))
        self.sample_rate = int(self.config.get('sensors.voltage_current.sample_rate', 1))
        self.i2c_bus = int(self.config.get('sensors.voltage_current.i2c_bus', 1))
        self.ina219_address = int(self.config.get('sensors.voltage_current.ina219_address', 0x40))
        self.ads1115_address = int(self.config.get('sensors.voltage_current.ads1115_address', 0x48))
        self.thermistor_channel = int(self.config.get('sensors.voltage_current.thermistor_channel', 0))
        self.csv_path = str(self.config.get('sensors.voltage_current.csv_simulator_path', 'data/voltage_current_simulation.csv'))
        self.voltage_range = tuple(self.config.get('sensors.voltage_current.voltage_range', [3.0, 5.0]))
        self.current_range = tuple(self.config.get('sensors.voltage_current.current_range', [-10.0, 10.0]))
        self.temp_range = tuple(self.config.get('sensors.voltage_current.temp_range', [-20.0, 70.0]))

        self._csv_lock = threading.Lock()
        self._csv_file = None
        self._csv_reader = None
        self._csv_iter = None

        self._i2c_lock = threading.Lock()
        self._ina219 = None
        self._ads = None

        # initial health
        self._health.status = 'unknown'
        self._health.error_count = 0
        self._health.availability_percent = 0.0

        if not self.enable:
            self._health.status = 'disabled'

        # try to init production devices lazily
        if self.enable and not self.enable_sim and _has_circuit:
            try:
                self._init_i2c_devices()
            except Exception:
                logger.exception('I2C device init failed; will use simulation if enabled')

    def get_name(self) -> str:
        return 'voltage_current'

    def _init_i2c_devices(self) -> bool:
        if not _has_circuit:
            return False
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            # INA219
            try:
                self._ina219 = INA219(i2c, address=self.ina219_address)
            except Exception:
                self._ina219 = None
            # ADS1115
            try:
                self._ads = ADS1115(i2c, address=self.ads1115_address)
            except Exception:
                self._ads = None
            return True
        except Exception as e:
            logger.exception('Failed to initialize I2C bus: %s', e)
            return False

    def _open_csv(self):
        try:
            self._csv_file = open(self.csv_path, 'r', encoding='utf-8')
            self._csv_reader = csv.DictReader(self._csv_file)
            self._csv_iter = iter(self._csv_reader)
            return True
        except Exception as e:
            logger.warning('CSV simulator open failed: %s', e)
            self._csv_file = None
            self._csv_reader = None
            self._csv_iter = None
            return False

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.enable:
            return None

        with self._lock:
            # Simulation path
            if self.enable_sim or not _has_circuit:
                try:
                    if self._csv_iter is None:
                        if not self._open_csv():
                            self._health.error_count += 1
                            self._health.status = 'error'
                            return None
                    try:
                        row = next(self._csv_iter)
                    except StopIteration:
                        # wrap around
                        try:
                            self._csv_file.seek(0)
                            self._csv_reader = csv.DictReader(self._csv_file)
                            self._csv_iter = iter(self._csv_reader)
                            row = next(self._csv_iter)
                        except Exception:
                            self._health.error_count += 1
                            self._health.status = 'error'
                            return None

                    voltage = float(row.get('voltage_v') or row.get('pack_voltage_v') or 0.0)
                    current = float(row.get('current_a') or row.get('pack_current_a') or 0.0)
                    temp = float(row.get('temp_c') or row.get('max_temp_c') or 0.0)
                    power = voltage * current

                    # range checks
                    out_of_range_fields = []
                    try:
                        if not (self.voltage_range[0] <= voltage <= self.voltage_range[1]):
                            out_of_range_fields.append('voltage')
                    except Exception:
                        pass
                    try:
                        if not (self.current_range[0] <= current <= self.current_range[1]):
                            out_of_range_fields.append('current')
                    except Exception:
                        pass
                    try:
                        if not (self.temp_range[0] <= temp <= self.temp_range[1]):
                            out_of_range_fields.append('temp')
                    except Exception:
                        pass

                    sample = {
                        'pack_voltage_v': float(voltage),
                        'pack_current_a': float(current),
                        'max_temp_c': float(temp),
                        'power_w': float(power),
                        'timestamp': row.get('timestamp') or datetime.now(timezone.utc).isoformat(),
                    }

                    if out_of_range_fields:
                        # Surface flags in sample but still return the reading
                        sample['sensor_out_of_range'] = True
                        sample['out_of_range_fields'] = out_of_range_fields
                        self._health.error_count += 1
                        self._health.status = 'warning'
                        # record when last out-of-range occurred
                        try:
                            self._health.last_out_of_range = sample.get('timestamp')
                        except Exception:
                            pass
                    else:
                        self._health.status = 'ok'

                    # health
                    self._record_read(True)
                    return sample
                except Exception as e:
                    logger.exception('Simulation read failed: %s', e)
                    self._health.error_count += 1
                    self._record_read(False)
                    self._health.status = 'error'
                    return None

            # Production path (I2C)
            try:
                if self._ina219 is None or self._ads is None:
                    # attempt init
                    self._init_i2c_devices()
                voltage = None
                current = None
                temp = None
                power = None
                with self._i2c_lock:
                    if self._ina219:
                        try:
                            voltage = float(self._ina219.bus_voltage)
                            current = float(self._ina219.current)
                        except Exception:
                            logger.exception('INA219 read failed')
                    if self._ads:
                        try:
                            chan = AnalogIn(self._ads, self.thermistor_channel)
                            # simple conversion: assume channel returns volts proportional to temp
                            temp = float(chan.voltage) * 100.0  # crude scaling, replace with thermistor math if available
                        except Exception:
                            logger.exception('ADS1115 read failed')
                if voltage is None and current is None and temp is None:
                    self._health.error_count += 1
                    self._record_read(False)
                    self._health.status = 'error'
                    return None
                if voltage is not None and current is not None:
                    power = voltage * current
                # range checks (apply same logic as simulation path)
                out_of_range_fields = []
                try:
                    if voltage is not None and not (self.voltage_range[0] <= voltage <= self.voltage_range[1]):
                        out_of_range_fields.append('voltage')
                except Exception:
                    pass
                try:
                    if current is not None and not (self.current_range[0] <= current <= self.current_range[1]):
                        out_of_range_fields.append('current')
                except Exception:
                    pass
                try:
                    if temp is not None and not (self.temp_range[0] <= temp <= self.temp_range[1]):
                        out_of_range_fields.append('temp')
                except Exception:
                    pass

                sample = {
                    'pack_voltage_v': float(voltage) if voltage is not None else None,
                    'pack_current_a': float(current) if current is not None else None,
                    'max_temp_c': float(temp) if temp is not None else None,
                    'power_w': float(power) if power is not None else None,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }

                if out_of_range_fields:
                    sample['sensor_out_of_range'] = True
                    sample['out_of_range_fields'] = out_of_range_fields
                    self._health.error_count += 1
                    self._health.status = 'warning'
                    try:
                        self._health.last_out_of_range = sample.get('timestamp')
                    except Exception:
                        pass
                else:
                    self._health.status = 'ok'

                self._record_read(True)
                return sample
            except Exception as e:
                logger.exception('Production read failed: %s', e)
                self._health.error_count += 1
                self._record_read(False)
                self._health.status = 'error'
                return None

    def get_health(self) -> Dict[str, Any]:
        return {
            'status': self._health.status,
            'error_count': self._health.error_count,
            'availability': self._health.availability_percent,
            'mode': 'simulation' if self.enable_sim or not _has_circuit else 'i2c',
            'last_out_of_range': getattr(self._health, 'last_out_of_range', None),
        }

    def is_available(self) -> bool:
        if not self.enable:
            return False
        if self.enable_sim:
            return os.path.exists(self.csv_path)
        # check device objects
        return (_has_circuit and (self._ina219 is not None or self._ads is not None))

    def close(self) -> None:
        try:
            if self._csv_file:
                try:
                    self._csv_file.close()
                except Exception:
                    pass
                self._csv_file = None
                self._csv_reader = None
                self._csv_iter = None
        except Exception:
            pass
