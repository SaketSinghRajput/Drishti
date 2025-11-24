"""GSMHandler: send SMS via AT commands or simulation outbox.

This handler is simulation-first. In production it uses pyserial to talk
to a USB-attached GSM modem (SIM800L-like) and implements a simple
send_sms(phone,message) API and health checks via AT command.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.config_loader import ConfigLoader
from data_collection.sensors.base import SensorHandler
from utils.time_utils import get_utc_timestamp

logger = logging.getLogger(__name__)

try:
    import serial
    from serial import SerialException, SerialTimeoutException
except Exception:
    serial = None
    SerialException = Exception
    SerialTimeoutException = Exception


class GSMHandler(SensorHandler):
    def __init__(self, config: ConfigLoader):
        super().__init__(config)
        self.enable = bool(self.config.get('sensors.gsm.enable', False))
        self.enable_sim = bool(self.config.get('sensors.gsm.enable_simulation', True))
        self.port = str(self.config.get('sensors.gsm.port', '/dev/ttyUSB0'))
        self.baudrate = int(self.config.get('sensors.gsm.baudrate', 115200))
        self.timeout_s = float(self.config.get('sensors.gsm.timeout_s', 5.0))
        self.outbox_path = str(self.config.get('sensors.gsm.outbox_path', 'data/gsm_outbox.txt'))

        self._serial = None
        self._serial_lock = threading.Lock()
        # set initial health
        self._health.status = 'unknown'
        self._health.error_count = 0
        self._health.availability_percent = 0.0
        # If disabled in config, mark explicitly as disabled for health reporting
        if not self.enable:
            self._health.status = 'disabled'

    def get_name(self) -> str:
        return 'gsm'

    def read(self) -> Optional[Dict[str, Any]]:
        # passive handler; no polling data to return
        return None

    def _init_serial(self) -> bool:
        if self.enable_sim:
            return False
        if serial is None:
            logger.warning('pyserial not available; cannot open GSM serial port')
            self._health.status = 'error'
            return False
        with self._serial_lock:
            try:
                if self._serial and getattr(self._serial, 'is_open', True):
                    return True
                self._serial = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout_s)
                # basic health probe
                ok = self._send_at_expect_ok('AT')
                if ok:
                    self._health.status = 'ok'
                    return True
                else:
                    self._health.status = 'error'
                    return False
            except Exception as e:
                logger.exception('Failed to open GSM serial port: %s', e)
                self._health.error_count += 1
                self._health.status = 'error'
                try:
                    if self._serial:
                        try:
                            self._serial.close()
                        except Exception:
                            pass
                        self._serial = None
                except Exception:
                    pass
                return False

    def _send_at_expect_ok(self, cmd: str) -> bool:
        try:
            if self._serial is None:
                return False
            # write cmd with CRLF
            self._serial.write((cmd + '\r\n').encode('utf-8'))
            deadline = time.time() + (self.timeout_s or 2.0)
            buf = b''
            while time.time() < deadline:
                try:
                    chunk = self._serial.read(64)
                except Exception:
                    chunk = b''
                if chunk:
                    buf += chunk
                    if b'OK' in buf:
                        return True
                    if b'ERROR' in buf:
                        return False
                else:
                    time.sleep(0.05)
            return False
        except Exception:
            return False

    def send_sms(self, phone: str, message: str) -> bool:
        """Send SMS to `phone`. Returns True if sent (or simulated) otherwise False."""
        if not self.enable:
            logger.debug('GSMHandler disabled in config; not sending SMS')
            return False
        with self._lock:
            # Simulation mode: append to outbox file
            if self.enable_sim or serial is None:
                try:
                    os.makedirs(os.path.dirname(self.outbox_path) or '.', exist_ok=True)
                    entry = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'phone': phone,
                        'message': message,
                        'status': 'simulated',
                    }
                    with open(self.outbox_path, 'a', encoding='utf-8') as fh:
                        fh.write(json.dumps(entry) + '\n')
                    # record as successful virtual send
                    self._record_read(True)
                    self._health.status = 'ok'
                    return True
                except Exception as e:
                    logger.exception('Failed to write GSM outbox: %s', e)
                    self._record_read(False)
                    self._health.error_count += 1
                    self._health.status = 'error'
                    return False

            # Production mode: use serial
            try:
                if not self._init_serial():
                    return False
                with self._serial_lock:
                    # set text mode
                    if not self._send_at_expect_ok('AT+CMGF=1'):
                        raise SerialException('CMGF failed')
                    # send CMGS
                    cmd = f'AT+CMGS="{phone}"'
                    # write command
                    self._serial.write((cmd + '\r\n').encode('utf-8'))
                    # wait for '>' prompt
                    prompt_deadline = time.time() + (self.timeout_s or 5.0)
                    prompt_buf = b''
                    got_prompt = False
                    while time.time() < prompt_deadline:
                        chunk = self._serial.read(64)
                        if chunk:
                            prompt_buf += chunk
                            if b'>' in prompt_buf:
                                got_prompt = True
                                break
                        else:
                            time.sleep(0.05)
                    if not got_prompt:
                        raise SerialTimeoutException('No > prompt from modem')
                    # send message + ctrl-z
                    self._serial.write((message + '\x1A').encode('utf-8'))
                    # wait for +CMGS or OK
                    resp_deadline = time.time() + (self.timeout_s or 10.0)
                    resp_buf = b''
                    sent_ok = False
                    while time.time() < resp_deadline:
                        chunk = self._serial.read(128)
                        if chunk:
                            resp_buf += chunk
                            if b'+CMGS' in resp_buf or b'OK' in resp_buf:
                                sent_ok = True
                                break
                        else:
                            time.sleep(0.05)
                    if not sent_ok:
                        raise SerialTimeoutException('No CMGS acknowledgement')
                    # success
                    self._record_read(True)
                    self._health.status = 'ok'
                    return True
            except Exception as e:
                logger.exception('GSM send_sms failed: %s', e)
                self._health.error_count += 1
                self._record_read(False)
                self._health.status = 'error'
                # close serial to force reinit next time
                try:
                    if self._serial:
                        try:
                            self._serial.close()
                        except Exception:
                            pass
                        self._serial = None
                except Exception:
                    pass
                return False

    def get_health(self) -> Dict[str, Any]:
        return {
            'status': self._health.status,
            'error_count': self._health.error_count,
            'availability': self._health.availability_percent,
            'mode': 'simulation' if self.enable_sim or serial is None else 'serial',
        }

    def is_available(self) -> bool:
        # Consider disabled handlers as not available
        if not self.enable:
            return False
        return self._health.status != 'error'

    def close(self) -> None:
        try:
            with self._serial_lock:
                if self._serial:
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                    self._serial = None
        except Exception:
            pass
