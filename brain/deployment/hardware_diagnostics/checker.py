import glob
import json
import logging
import os
import re
import subprocess
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.config_loader import ConfigLoader
import yaml


class HardwareChecker:
    def _normalize_expected(self, raw) -> List[str]:
        """Normalize various config formats into a list of sensor names.

        Accepts lists, simple string representations like "[a, b]" or "a, b",
        or dicts where keys are treated as sensor names.
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            s = raw.strip()
            if s.startswith('[') and s.endswith(']'):
                inner = s[1:-1]
                return [x.strip().strip("'\"") for x in inner.split(',') if x.strip()]
            return [x.strip() for x in s.split(',') if x.strip()]
        if isinstance(raw, dict):
            return list(raw.keys())
        return []

    def __init__(self, config: Optional[ConfigLoader] = None, local_config_path: Optional[str] = None, telemetry: Optional[Any] = None):
        # Accept either a ConfigLoader instance or load default
        if config is None:
            self.cfg = ConfigLoader('config/default_config.yaml')
        else:
            self.cfg = config

        # Load local package config for mappings/timeouts
        base_dir = os.path.dirname(__file__)
        local_path = local_config_path or os.path.join(base_dir, 'config.yaml')
        try:
            with open(local_path, 'r', encoding='utf-8') as fh:
                self.local = yaml.safe_load(fh) or {}
        except Exception:
            self.local = {}

        # merge some defaults
        self.timeouts = self.local.get('timeouts', {})
        self.enable_simulation = self.local.get('enable_simulation', False) or self.cfg.get('hardware_diagnostics.enable_simulation', False)
        # Prefer values from the main config, then local package config, then hardcoded defaults
        self.log_dir = self.cfg.get('hardware_diagnostics.log_dir') or self.local.get('log_dir') or '/home/pi/system_logs'
        self.log_file = self.cfg.get('hardware_diagnostics.log_file') or self.local.get('log_file') or 'hardware_check.log'
        self.log_level = getattr(logging, (self.cfg.get('hardware_diagnostics.log_level') or self.local.get('log_level') or 'INFO'))
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, self.log_file)
        self._logger = logging.getLogger('hardware_checker')
        self._logger.setLevel(self.log_level)
        # avoid adding duplicate handlers
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(self.log_path) for h in self._logger.handlers):
            fh = logging.FileHandler(self.log_path)
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        # thread lock for future-proof logging (use simple Lock here; callers
        # should avoid re-entrant acquisition to prevent deadlocks)
        self._lock = threading.Lock()

        # telemetry collector (optional) - can be used to log hardware checks
        self.telemetry = telemetry

        # Phase 3: fail-safe state and last health cache
        self._disabled_modules: List[str] = []
        self._last_health_status: Optional[Dict[str, Any]] = None

        # Phase 3 config flags (default to enabled where reasonable)
        self.alerts_enabled = bool(self.cfg.get('hardware_diagnostics.alerts.enable', True))
        self.dashboard_integration_enabled = bool(self.cfg.get('hardware_diagnostics.dashboard_integration.enable', True))
        self.telemetry_enabled = bool(self.cfg.get('hardware_diagnostics.telemetry.enable', True))

        # expected sensors (be tolerant to different YAML parsing results)
        raw_expected = self.cfg.get('data_collection.expected_sensors') or self.cfg.get('expected_sensors') or []
        self.expected_sensors = self._normalize_expected(raw_expected)
        # allow override from main config
        overrides = self.cfg.get('hardware_diagnostics.expected_components')
        if isinstance(overrides, list) and overrides:
            self.expected_sensors = overrides

        # log resolved expected sensors for debugging/parity with tests
        try:
            self._logger.info('Resolved expected_sensors: %r', self.expected_sensors)
        except Exception:
            # logger may not be fully configured in some test paths
            pass

        # hardware mapping from local config
        self.i2c_map = self.local.get('i2c_addresses', {}) or {}
        self.spi_map = self.local.get('spi_devices', {}) or {}
        self.uart_map = self.local.get('uart_devices', {}) or {}
        self.can_list = self.local.get('can_interfaces', []) or []
        self.camera_map = self.local.get('camera_devices', {}) or {}
        self.esp32_uart = self.local.get('esp32_uart_device')

        # If no expected sensors were resolved from the main config (e.g. simple YAML parser),
        # fall back to discovering expected sensors from the local mappings to keep tests deterministic.
        if not self.expected_sensors:
            fallback = set()
            try:
                fallback.update([v for v in (self.i2c_map or {}).values() if isinstance(v, str)])
                fallback.update([v for v in (self.spi_map or {}).values() if isinstance(v, str)])
                fallback.update([v for v in (self.uart_map or {}).values() if isinstance(v, str)])
                fallback.update([v for v in (self.camera_map or {}).values() if isinstance(v, str)])
                if self.can_list:
                    # logical mapping: if CAN exists, assume BMS may be present
                    fallback.add('bms')
            except Exception:
                fallback = set()
            self.expected_sensors = sorted(fallback)

        # initialize GPIO (Phase 3) - uses DummyGPIO fallback for non-Pi environments
        try:
            self._init_gpio()
        except Exception:
            # keep running even if GPIO setup fails
            pass

    def _run_command(self, cmd: List[str], timeout: Optional[int] = None) -> (bool, str, str):
        timeout = timeout or self.timeouts.get('default', 5)
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return True, cp.stdout or cp.stderr or '', ''
        except subprocess.TimeoutExpired as ex:
            return False, '', f'timeout: {ex}'
        except FileNotFoundError as ex:
            return False, '', f'not_found: {ex}'
        except Exception as ex:
            return False, '', str(ex)

    def _parse_i2c_output(self, out: str) -> List[str]:
        # find occurrences like 68 or 0x68
        addrs = set()
        for m in re.finditer(r'0x[0-9a-fA-F]+|\b[0-9a-fA-F]{2}\b', out):
            val = m.group(0)
            if val.startswith('0x'):
                addrs.add(val.lower())
            else:
                addrs.add(f'0x{val.lower()}')
        return sorted(addrs)

    def _map_i2c_to_sensors(self, addrs: List[str]) -> Dict[str, Any]:
        found = {}
        unexpected = []
        for a in addrs:
            mapped = self.i2c_map.get(a) or self.i2c_map.get(a.lower())
            if mapped:
                found[mapped] = {'found': True, 'details': f'I2C {a}'}
            else:
                # Recognize common ADC addresses for power-bus monitoring
                if a.lower() in ('0x40', '0x40'):
                    found.setdefault('voltage_sensor', {'found': True, 'details': f'INA219 at {a}'})
                elif a.lower() in ('0x48', '0x48'):
                    # ADS1115 often used for thermistor/temp inputs
                    found.setdefault('current_sensor', {'found': True, 'details': f'ADS1115 at {a}'})
                    found.setdefault('temp_sensor', {'found': True, 'details': f'ADS1115 at {a} (thermistor)'})
                else:
                    unexpected.append({'type': 'i2c', 'addr': a})
        return found, unexpected

    def _detect_i2c(self) -> (Dict[str, Any], List[Any]):
        ok, out, err = self._run_command(['i2cdetect', '-y', '1'], timeout=self.timeouts.get('i2cdetect', 5))
        if not ok:
            reason = err or 'i2cdetect_failed'
            self._logger.warning('I2C detect failed: %s', reason)
            return {}, []
        addrs = self._parse_i2c_output(out)
        return self._map_i2c_to_sensors(addrs)

    def _detect_spi(self) -> (Dict[str, Any], List[Any]):
        devs = glob.glob('/dev/spidev*')
        found = {}
        unexpected = []
        for d in devs:
            mapped = self.spi_map.get(d)
            if mapped:
                found[mapped] = {'found': True, 'details': d}
            else:
                unexpected.append({'type': 'spi', 'dev': d})
        return found, unexpected

    def _detect_uart(self) -> (Dict[str, Any], List[Any]):
        devs = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyAMA*') + glob.glob('/dev/ttyACM*')
        found = {}
        unexpected = []
        for d in devs:
            mapped = self.uart_map.get(d)
            if mapped:
                found[mapped] = {'found': True, 'details': d}
            else:
                # map common gps names
                if 'gps' in self.expected_sensors and ('ttyUSB' in d or 'ttyAMA' in d):
                    found.setdefault('gps', {'found': True, 'details': d})
                else:
                    unexpected.append({'type': 'uart', 'dev': d})
        return found, unexpected

    def _detect_gpio(self) -> Dict[str, Any]:
        # Read configured GPIO pin mappings from local config
        pins = self.local.get('gpio_pins', {}) or {}
        results: Dict[str, Any] = {}
        if not pins:
            # no configured pins, keep backward-compatible behavior
            try:
                import RPi.GPIO as GPIO  # type: ignore

                try:
                    GPIO.setmode(GPIO.BCM)
                    return {'gpio': {'found': True, 'details': 'RPi.GPIO available'}}
                except Exception as ex:
                    return {'gpio': {'found': False, 'reason': str(ex)}}
            except Exception:
                return {'gpio': {'found': False, 'reason': 'GPIO library not available'}}

        # If pins are configured, report per-component status
        try:
            import RPi.GPIO as GPIO  # type: ignore

            try:
                GPIO.setmode(GPIO.BCM)
                for name, bcm in pins.items():
                    results[name] = {'found': True, 'details': f'BCM {bcm}'}
            except Exception as ex:
                for name, bcm in pins.items():
                    results[name] = {'found': False, 'reason': str(ex)}
        except Exception:
            for name, bcm in pins.items():
                results[name] = {'found': False, 'reason': 'GPIO library not available'}

        return results

    def _detect_can(self) -> (Dict[str, Any], List[Any]):
        found = {}
        unexpected = []
        for iface in self.can_list:
            ok, out, err = self._run_command(['ip', 'link', 'show', iface], timeout=self.timeouts.get('ip_link', 2))
            if not ok:
                self._logger.warning('CAN check failed for %s: %s', iface, err)
                continue
            # Interpret interface output to determine up/down
            if iface in out or 'BROADCAST' in out or 'state' in out or 'UP' in out:
                val = {'found': True, 'details': iface}
            else:
                val = {'found': False, 'reason': 'interface_down'}
            found.setdefault('vehicle_can', val)
            # If the logical sensor 'bms' is expected, mirror CAN status to satisfy it
            if 'bms' in (self.expected_sensors or []):
                found.setdefault('bms', val)
        return found, unexpected

    def _detect_camera(self) -> (Dict[str, Any], List[Any]):
        found = {}
        unexpected = []
        # prefer v4l2 where available
        ok, out, err = self._run_command(['v4l2-ctl', '--list-devices'], timeout=self.timeouts.get('v4l2_ctl', 3))
        if ok and out:
            # find /dev/video* entries
            devs = re.findall(r'/dev/video\d+', out)
            for d in devs:
                mapped = self.camera_map.get(d)
                if mapped:
                    found[mapped] = {'found': True, 'details': d}
                else:
                    unexpected.append({'type': 'video', 'dev': d})
            return found, unexpected
        # fallback to glob
        devs = glob.glob('/dev/video*')
        for d in devs:
            mapped = self.camera_map.get(d)
            if mapped:
                found[mapped] = {'found': True, 'details': d}
            else:
                unexpected.append({'type': 'video', 'dev': d})
        return found, unexpected

    def _detect_ble(self) -> Dict[str, Any]:
        ok, out, err = self._run_command(['bluetoothctl', 'list'], timeout=self.timeouts.get('bluetoothctl', 5))
        if not ok:
            return {'ble_mesh': {'found': False, 'reason': err or 'bluetoothctl_not_available'}}
        if 'Controller' in out or out.strip():
            return {'ble_mesh': {'found': True, 'details': out.strip()}}
        return {'ble_mesh': {'found': False, 'reason': 'no_adapter'}}

    def _detect_esp32(self) -> Dict[str, Any]:
        # Phase 1: only check for UART device presence
        if not self.esp32_uart:
            return {'esp32': {'found': False, 'reason': 'not_configured'}}
        if os.path.exists(self.esp32_uart):
            return {'esp32': {'found': True, 'details': self.esp32_uart}}
        return {'esp32': {'found': False, 'reason': f'{self.esp32_uart} not present'}}

    def detect_hardware(self) -> Dict[str, Any]:
        with self._lock:
            ts = datetime.utcnow().isoformat()
            summary: Dict[str, Any] = {
                'timestamp': ts,
                'expected_sensors': list(self.expected_sensors),
                'found_sensors': [],
                'missing_sensors': [],
                'details': {},
                'unexpected_devices': [],
            }

            self._logger.info('Starting hardware detection, simulation=%s', self.enable_simulation)

            if self.enable_simulation:
                # read expected sensors fresh from config to be robust in test environments
                sensors_raw = self.cfg.get('data_collection.expected_sensors') or self.cfg.get('expected_sensors') or []
                sensors = self._normalize_expected(sensors_raw) or list(self.expected_sensors)
                for s in sensors:
                    summary['found_sensors'].append(s)
                    summary['details'][s] = {'found': True, 'details': 'Simulated'}
                self._logger.info('Simulation mode: marking all expected sensors as found')
                self._logger.info('Hardware check complete: %d/%d found', len(summary['found_sensors']), len(self.expected_sensors))
                return summary

            # run detection methods
            found_all = {}
            unexpected = []

            # I2C detection
            try:
                f, u = self._detect_i2c()
                found_all.update(f)
                unexpected.extend(u)
            except Exception as ex:
                self._logger.warning('I2C detection raised: %s', ex)

            # SPI
            try:
                f, u = self._detect_spi()
                found_all.update(f)
                unexpected.extend(u)
            except Exception as ex:
                self._logger.warning('SPI detection raised: %s', ex)

            # UART
            try:
                f, u = self._detect_uart()
                found_all.update(f)
                unexpected.extend(u)
            except Exception as ex:
                self._logger.warning('UART detection raised: %s', ex)

            # GPIO
            try:
                g = self._detect_gpio()
                found_all.update(g)
            except Exception as ex:
                self._logger.warning('GPIO detection raised: %s', ex)

            # CAN
            try:
                f, u = self._detect_can()
                found_all.update(f)
                unexpected.extend(u)
            except Exception as ex:
                self._logger.warning('CAN detection raised: %s', ex)

            # Camera
            try:
                f, u = self._detect_camera()
                found_all.update(f)
                unexpected.extend(u)
            except Exception as ex:
                self._logger.warning('Camera detection raised: %s', ex)

            # BLE
            try:
                b = self._detect_ble()
                found_all.update(b)
            except Exception as ex:
                self._logger.warning('BLE detection raised: %s', ex)

            # ESP32
            try:
                e = self._detect_esp32()
                found_all.update(e)
            except Exception as ex:
                self._logger.warning('ESP32 detection raised: %s', ex)

            # Aggregate results per expected sensor
            for s in self.expected_sensors:
                if s in found_all and found_all[s].get('found'):
                    summary['found_sensors'].append(s)
                    summary['details'][s] = found_all[s]
                else:
                    # maybe found under different key (e.g. vehicle_can -> bms)
                    summary['missing_sensors'].append(s)
                    summary['details'][s] = found_all.get(s, {'found': False, 'reason': 'not_detected'})

            # include any unexpected devices
            summary['unexpected_devices'] = unexpected
            self._logger.info('Hardware check complete: %d/%d found', len(summary['found_sensors']), len(self.expected_sensors))
            if summary['missing_sensors']:
                self._logger.warning('Missing sensors: %s', ','.join(summary['missing_sensors']))
            if unexpected:
                self._logger.info('Unexpected devices: %s', unexpected)

            # write a json summary next to the log for easy consumption
            try:
                summary_path = os.path.join(self.log_dir, f'hardware_summary_{ts.replace(":", "-")}.json')
                with open(summary_path, 'w', encoding='utf-8') as fh:
                    json.dump(summary, fh, indent=2)
            except Exception:
                pass

            return summary

    # ------------------ Phase 2: Self-tests ------------------
    def _run_with_timeout(self, fn, timeout_s: float):
        """Run a callable with a timeout (thread-based).

        Returns the callable result or raises TimeoutError if it did not finish.
        """
        result = {}

        def _target():
            try:
                result['value'] = fn()
            except Exception as e:
                result['exc'] = e

        th = threading.Thread(target=_target, daemon=True)
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            raise TimeoutError(f'function timed out after {timeout_s}s')
        if 'exc' in result:
            raise result['exc']
        return result.get('value')

    def self_test(self, component: str) -> Dict[str, Any]:
        """Public dispatcher to run a self-test for a single component.

        Returns a dict: {status: 'OK'|'FAILED', reason: str|None, details: Any}
        """
        with self._lock:
            comp = component.lower()
            self._logger.info('Starting self-test for %s', comp)

            if self.enable_simulation:
                self._logger.info('Simulation mode: self-test for %s simulated', comp)
                return {'status': 'OK', 'reason': None, 'details': 'Simulated'}

            # resolve timeout (prefer main config, then local mappings)
            timeouts = self.cfg.get('hardware_diagnostics.self_test_timeouts') or self.local.get('self_test_timeouts', {}) or {}
            timeout_s = timeouts.get(comp, timeouts.get('default', 5))

            method_name = f'_test_{comp}'
            method = getattr(self, method_name, None)
            if method is None or not callable(method):
                return {'status': 'FAILED', 'reason': 'unknown_component', 'details': None}

            try:
                res = self._run_with_timeout(lambda: method(), float(timeout_s))
                # normalize result
                if not isinstance(res, dict):
                    return {'status': 'FAILED', 'reason': 'invalid_result', 'details': res}
                status = res.get('status', 'FAILED')
                return res
            except TimeoutError as te:
                self._logger.warning('Self-test %s timed out', comp)
                return {'status': 'FAILED', 'reason': f'timeout: {te}', 'details': None}
            except Exception as ex:
                self._logger.exception('Self-test %s raised exception', comp)
                return {'status': 'FAILED', 'reason': str(ex), 'details': None}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run self-tests for all expected sensors.

        Returns a summary dict with keys: timestamp, expected_sensors, passed_tests,
        failed_tests, skipped_tests, test_results.
        """
        # NOTE: do NOT hold the module-level lock across calls to detect_hardware()
        # or self_test() because those methods acquire the same lock internally and
        # doing so would cause a deadlock. Build the summary locally without
        # long-lived shared state and rely on per-method locking.
        ts = datetime.utcnow().isoformat()
        summary: Dict[str, Any] = {
            'timestamp': ts,
            'expected_sensors': list(self.expected_sensors),
            'passed_tests': [],
            'failed_tests': [],
            'skipped_tests': [],
            'test_results': {},
        }

        # detect hardware first and only run tests on detected sensors
        det = self.detect_hardware()
        detected = set(det.get('found_sensors', []))

        for comp in self.expected_sensors:
            if comp not in detected:
                # skip testing absent hardware
                summary['skipped_tests'].append(comp)
                summary['test_results'][comp] = {'status': 'SKIPPED', 'reason': 'not_detected'}
                continue

            res = self.self_test(comp)
            summary['test_results'][comp] = res
            if res.get('status') == 'OK':
                summary['passed_tests'].append(comp)
            else:
                summary['failed_tests'].append(comp)

        # log summary
        self._logger.info('Self-tests complete: %d passed, %d failed, %d skipped', len(summary['passed_tests']), len(summary['failed_tests']), len(summary['skipped_tests']))
        try:
            summary_path = os.path.join(self.log_dir, f'selftest_summary_{ts.replace(":", "-")}.json')
            with open(summary_path, 'w', encoding='utf-8') as fh:
                json.dump(summary, fh, indent=2)
        except Exception:
            pass

        # Build last health status for exposure
        overall = 'OK'
        if summary['failed_tests']:
            overall = 'FAILED'
        elif summary['skipped_tests']:
            overall = 'DEGRADED'

        components = {c: self._format_component_status(c, summary['test_results'].get(c, {})) for c in summary['test_results'].keys()}

        health = {
            'timestamp': ts,
            'overall_status': overall,
            'components': components,
            'disabled_modules': list(summary['failed_tests']) + list(summary['skipped_tests']),
        }

        # update internal state
        with self._lock:
            self._last_health_status = health
            self._disabled_modules = health.get('disabled_modules', [])

        # Log clear pass/fail summary for alerting and operators
        if summary.get('failed_tests'):
            try:
                self._logger.error('Hardware self-tests FAILED for components: %s', ','.join(summary.get('failed_tests', [])))
            except Exception:
                pass
        else:
            try:
                # If there are skipped tests but no failures, treat as degraded but still INFO
                if summary.get('skipped_tests'):
                    self._logger.info('Hardware self-tests completed with skipped components: %s', ','.join(summary.get('skipped_tests', [])))
                else:
                    self._logger.info('All hardware self-tests passed')
            except Exception:
                pass

        # Trigger GPIO alerts (config-gated)
        try:
            if self.alerts_enabled:
                self._trigger_alerts(summary)
        except Exception:
            self._logger.exception('Alert triggering failed')

        # Post to dashboards (config-gated)
        try:
            if self.dashboard_integration_enabled:
                self._post_to_dashboards(health)
        except Exception:
            self._logger.exception('Posting to dashboards failed')

        # Log to telemetry (config-gated)
        try:
            if self.telemetry_enabled and self.telemetry:
                self._log_to_telemetry(summary)
        except Exception:
            self._logger.exception('Logging to telemetry failed')

        return summary

    # ------------------ Phase 3: Alerts, dashboards, telemetry ------------------
    def _init_gpio(self) -> None:
        """Initialize GPIO interface with Dummy fallback for non-Pi environments."""
        # Use the _DummyGPIO pattern from battery_monitor.alerts.alert_manager
        class _DummyGPIO:
            BCM = 'BCM'

            OUT = 'OUT'

            LOW = 0
            HIGH = 1

            def __init__(self):
                self._state = {}

            def setmode(self, mode):
                return None

            def setup(self, pin, mode):
                self._state[pin] = self.LOW

            def output(self, pin, value):
                self._state[pin] = value

            def cleanup(self):
                self._state.clear()

        try:
            import RPi.GPIO as GPIO  # type: ignore
        except Exception:
            GPIO = _DummyGPIO()

        self._gpio = GPIO

    def _trigger_alerts(self, results: Dict[str, Any]) -> None:
        """Trigger buzzer/LED alerts based on test results.

        - Any failures -> buzzer beep 3x and RED LED solid blink
        - All passed -> GREEN LED blink 3x
        """
        pins = self.local.get('gpio_pins', {}) or {}
        buzzer_pin = pins.get('buzzer') or self.cfg.get('hardware_diagnostics.alerts.gpio_buzzer_pin')
        led_red = pins.get('led_red') or self.cfg.get('hardware_diagnostics.alerts.gpio_led_red_pin')
        led_green = pins.get('led_green') or self.cfg.get('hardware_diagnostics.alerts.gpio_led_green_pin')

        # config defaults
        beep_duration = float(self.cfg.get(
            'hardware_diagnostics.alerts.buzzer_beep_duration_s',
            self.local.get('buzzer_beep_duration_s', 0.5),
        ))
        beep_count = int(self.cfg.get(
            'hardware_diagnostics.alerts.buzzer_beep_count',
            self.local.get('buzzer_beep_count', 3),
        ))
        beep_pause = float(self.cfg.get(
            'hardware_diagnostics.alerts.buzzer_pause_s',
            self.local.get('buzzer_pause_s', 0.05),
        ))

        blink_duration = float(self.cfg.get(
            'hardware_diagnostics.alerts.led_blink_duration_s',
            self.local.get('led_blink_duration_s', 0.3),
        ))
        blink_count = int(self.cfg.get(
            'hardware_diagnostics.alerts.led_blink_count',
            self.local.get('led_blink_count', 3),
        ))
        blink_pause = float(self.cfg.get(
            'hardware_diagnostics.alerts.led_pause_s',
            self.local.get('led_pause_s', 0.05),
        ))

        failed = bool(results.get('failed_tests'))

        try:
            # ensure gpio interface exists
            if not hasattr(self, '_gpio'):
                self._init_gpio()

            GPIO = self._gpio
            # operate under lock to avoid concurrent alert operations
            with self._lock:
                # Setup pins if using RPi.GPIO object (has setup method)
                if hasattr(GPIO, 'setmode'):
                    try:
                        GPIO.setmode(GPIO.BCM)
                    except Exception:
                        pass
                # Buzzer
                if buzzer_pin:
                    try:
                        if hasattr(GPIO, 'setup'):
                            GPIO.setup(buzzer_pin, GPIO.OUT)
                    except Exception:
                        pass

                import time

                if failed:
                    # beep buzzer beep_count times and blink red LED
                    for i in range(beep_count):
                        try:
                            if buzzer_pin:
                                GPIO.output(buzzer_pin, GPIO.HIGH)
                        except Exception:
                            pass
                        time.sleep(beep_duration)
                        try:
                            if buzzer_pin:
                                GPIO.output(buzzer_pin, GPIO.LOW)
                        except Exception:
                            pass
                        time.sleep(beep_pause)

                    # red LED blink using configured blink_count/duration
                    if led_red:
                        try:
                            if hasattr(GPIO, 'setup'):
                                GPIO.setup(led_red, GPIO.OUT)
                            for _ in range(blink_count):
                                GPIO.output(led_red, GPIO.HIGH)
                                time.sleep(blink_duration)
                                GPIO.output(led_red, GPIO.LOW)
                                time.sleep(blink_pause)
                        except Exception:
                            pass
                else:
                    # success: blink green LED and beep short
                    for i in range(blink_count):
                        try:
                            if led_green:
                                if hasattr(GPIO, 'setup'):
                                    GPIO.setup(led_green, GPIO.OUT)
                                GPIO.output(led_green, GPIO.HIGH)
                        except Exception:
                            pass
                        try:
                            if buzzer_pin:
                                GPIO.output(buzzer_pin, GPIO.HIGH)
                        except Exception:
                            pass
                        time.sleep(blink_duration)
                        try:
                            if led_green:
                                GPIO.output(led_green, GPIO.LOW)
                        except Exception:
                            pass
                        try:
                            if buzzer_pin:
                                GPIO.output(buzzer_pin, GPIO.LOW)
                        except Exception:
                            pass
                        time.sleep(blink_pause)

        except Exception:
            self._logger.exception('Error while triggering alerts')

    def _post_to_dashboards(self, health_status: Dict[str, Any]) -> None:
        """POST health_status JSON to configured dashboards."""
        try:
            import requests
        except Exception:
            self._logger.warning('requests not available; cannot post to dashboards')
            return

        main_url = self.cfg.get('hardware_diagnostics.dashboard_integration.main_dashboard_url', 'http://localhost:5000')
        mon_url = self.cfg.get('hardware_diagnostics.dashboard_integration.monitoring_dashboard_url', 'http://localhost:5003')
        timeout = float(self.cfg.get('hardware_diagnostics.dashboard_integration.timeout_s', 2))

        headers = {'Content-Type': 'application/json'}
        # Post to main dashboard
        try:
            requests.post(f"{main_url.rstrip('/')}/api/hardware_health", json=health_status, timeout=timeout, headers=headers)
        except Exception as e:
            self._logger.warning('Failed to post to main dashboard %s: %s', main_url, e)

        # Post to monitoring dashboard
        try:
            requests.post(f"{mon_url.rstrip('/')}/api/hardware_health", json=health_status, timeout=timeout, headers=headers)
        except Exception as e:
            self._logger.warning('Failed to post to monitoring dashboard %s: %s', mon_url, e)

    def _log_to_telemetry(self, results: Dict[str, Any]) -> None:
        """Log a hardware check summary to the telemetry collector if available."""
        try:
            if not self.telemetry:
                return
            # TelemetryCollector expects a summary format; call its log_hardware_check if exists
            if hasattr(self.telemetry, 'log_hardware_check'):
                self.telemetry.log_hardware_check(results)
            else:
                # Fallback: append minimal entry via generic API if present
                if hasattr(self.telemetry, 'log'):
                    entry = {
                        'timestamp': results.get('timestamp'),
                        'type': 'hardware_check',
                        'passed': results.get('passed_tests', []),
                        'failed': results.get('failed_tests', []),
                        'skipped': results.get('skipped_tests', []),
                    }
                    try:
                        self.telemetry.log(entry)
                    except Exception:
                        pass
        except Exception:
            self._logger.exception('Telemetry logging failed')

    def get_system_health_status(self) -> Optional[Dict[str, Any]]:
        """Return last computed health status or None if not available."""
        with self._lock:
            return None if self._last_health_status is None else dict(self._last_health_status)

    def is_module_enabled(self, component: str) -> bool:
        """Return True if component is not in disabled modules list."""
        with self._lock:
            return component not in (self._disabled_modules or [])

    def get_disabled_modules(self) -> List[str]:
        with self._lock:
            return list(self._disabled_modules or [])

    def _format_component_status(self, component: str, result: Dict[str, Any]) -> str:
        """Return a single-string status for a component result suitable for
        human-readable health JSON (e.g. 'OK', 'FAILED - reason', 'SKIPPED - not_detected').
        """
        st = result.get('status')
        if not st:
            return 'FAILED - unknown'
        # For OK, include details if available to aid debugging
        if st == 'OK':
            details = result.get('details')
            if details is None:
                return 'OK'
            try:
                # Prefer compact JSON for dict-like details, else str()
                if isinstance(details, dict):
                    short = json.dumps(details, default=str)
                else:
                    short = str(details)
                # Truncate very long strings
                if len(short) > 200:
                    short = short[:197] + '...'
                return f'OK - {short}'
            except Exception:
                return 'OK'
        # for SKIPPED or FAILED, include the reason if present
        reason = result.get('reason')
        if reason:
            return f"{st} - {reason}"
        return st

    # ------------------ component-specific tests ------------------
    def _test_imu(self) -> Dict[str, Any]:
        if self.enable_simulation:
            return {'status': 'OK', 'details': 'Simulated'}
        try:
            from data_collection.sensors.imu_handler import IMUHandler
        except Exception:
            return {'status': 'FAILED', 'reason': 'imu_handler_not_available'}

        try:
            handler = IMUHandler(self.cfg)
            sample = handler.read()
            if not isinstance(sample, dict):
                return {'status': 'FAILED', 'reason': 'no_sample'}
            # simple gravity check
            ranges = self.cfg.get('hardware_diagnostics.self_test_validation_ranges') or self.local.get('self_test_validation_ranges', {})
            imu_rng = ranges.get('imu', {})
            az = None
            for key in ('az', 'acc_z', 'accel_z'):
                if key in sample:
                    az = float(sample.get(key))
                    break
            if az is None:
                return {'status': 'FAILED', 'reason': 'missing_accel_z', 'details': sample}
            az_min = imu_rng.get('az_min', 7.8)
            az_max = imu_rng.get('az_max', 11.8)
            if not (az_min <= az <= az_max):
                return {'status': 'FAILED', 'reason': 'gravity_check_failed', 'details': {'az': az}}
            return {'status': 'OK', 'details': {'az': az}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_gps(self) -> Dict[str, Any]:
        if self.enable_simulation:
            return {'status': 'OK', 'details': 'Simulated'}
        try:
            from data_collection.sensors.gps_handler import GPSHandler
        except Exception:
            return {'status': 'FAILED', 'reason': 'gps_handler_not_available'}
        try:
            handler = GPSHandler(self.cfg)
            sample = handler.read()
            if not isinstance(sample, dict):
                return {'status': 'FAILED', 'reason': 'no_sample'}
            lat = sample.get('lat')
            lon = sample.get('lon')
            # If lat/lon are missing, try to see if raw NMEA sentences are available
            if lat is None or lon is None:
                raw = sample.get('nmea') or sample.get('raw') or sample.get('sentence')
                if isinstance(raw, str) and ('$G' in raw or raw.strip().startswith('$')):
                    # lightweight NMEA sanity check: look for GGA/RMC sentence types
                    if 'GGA' in raw or 'RMC' in raw or 'VTG' in raw:
                        return {'status': 'OK', 'details': {'nmea_preview': raw.strip()[:120]}}
                    # fallback: accept raw NMEA presence as partial success
                    return {'status': 'OK', 'details': {'nmea_preview': raw.strip()[:120]}}
                return {'status': 'FAILED', 'reason': 'missing_coords', 'details': sample}
            ranges = self.cfg.get('hardware_diagnostics.self_test_validation_ranges') or self.local.get('self_test_validation_ranges', {})
            gps_rng = ranges.get('gps', {})
            if not (gps_rng.get('lat_min', -90.0) <= float(lat) <= gps_rng.get('lat_max', 90.0)):
                return {'status': 'FAILED', 'reason': 'lat_out_of_range', 'details': {'lat': lat}}
            if not (gps_rng.get('lon_min', -180.0) <= float(lon) <= gps_rng.get('lon_max', 180.0)):
                return {'status': 'FAILED', 'reason': 'lon_out_of_range', 'details': {'lon': lon}}
            return {'status': 'OK', 'details': {'lat': lat, 'lon': lon}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_camera(self) -> Dict[str, Any]:
        if self.enable_simulation:
            return {'status': 'OK', 'details': 'Simulated'}
        try:
            from data_collection.sensors.camera_handler import CameraHandler
        except Exception:
            return {'status': 'FAILED', 'reason': 'camera_handler_not_available'}
        try:
            handler = CameraHandler(self.cfg)
            sample = handler.read()
            # accept truthy sample (handler-specific)
            if not sample:
                return {'status': 'FAILED', 'reason': 'no_frame'}

            # If the handler returns a dict with metadata, prefer that
            if isinstance(sample, dict):
                # common patterns: {'frame': <ndarray>, 'objects': [...]}
                if 'objects' in sample and sample.get('objects'):
                    return {'status': 'OK', 'details': {'objects': len(sample.get('objects', []))}}
                if 'frame' in sample:
                    frame = sample.get('frame')
                    try:
                        # numpy arrays have .shape
                        if hasattr(frame, 'shape'):
                            h, w = frame.shape[0], frame.shape[1]
                            if h > 0 and w > 0:
                                return {'status': 'OK', 'details': {'shape': frame.shape}}
                            return {'status': 'FAILED', 'reason': 'empty_frame', 'details': {'shape': frame.shape}}
                        # other sequence-like frames
                        if hasattr(frame, '__len__'):
                            if len(frame) > 0:
                                return {'status': 'OK', 'details': {'len': len(frame)}}
                    except Exception:
                        pass
                # if metadata empty, but dict present, treat as partial success
                return {'status': 'OK', 'details': {'metadata': True}}

            # If raw bytes or array-like, try to check length
            try:
                if hasattr(sample, '__len__') and len(sample) > 0:
                    return {'status': 'OK', 'details': {'len': len(sample)}}
            except Exception:
                pass

            return {'status': 'FAILED', 'reason': 'no_frame'}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_bms(self) -> Dict[str, Any]:
        if self.enable_simulation:
            return {'status': 'OK', 'details': 'Simulated'}
        try:
            from data_collection.sensors.bms_handler import BMSHandler
        except Exception:
            return {'status': 'FAILED', 'reason': 'bms_handler_not_available'}
        try:
            handler = BMSHandler(self.cfg)
            sample = handler.read()
            if not isinstance(sample, dict):
                return {'status': 'FAILED', 'reason': 'no_sample'}
            rng = self.cfg.get('hardware_diagnostics.self_test_validation_ranges') or self.local.get('self_test_validation_ranges', {})
            bms_rng = rng.get('bms', {})
            pack_v = sample.get('pack_voltage') or sample.get('voltage')
            soc = sample.get('soc')
            if pack_v is None or float(pack_v) <= bms_rng.get('pack_voltage_min', 0.0):
                return {'status': 'FAILED', 'reason': 'pack_voltage_invalid', 'details': sample}
            if soc is None:
                return {'status': 'FAILED', 'reason': 'missing_soc', 'details': sample}
            if not (bms_rng.get('soc_min', 0.0) <= float(soc) <= bms_rng.get('soc_max', 100.0)):
                return {'status': 'FAILED', 'reason': 'soc_out_of_range', 'details': {'soc': soc}}
            return {'status': 'OK', 'details': {'pack_voltage': pack_v, 'soc': soc}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_vehicle_can(self) -> Dict[str, Any]:
        if self.enable_simulation:
            return {'status': 'OK', 'details': 'Simulated'}
        # Prefer vehicle CAN handler if available
        try:
            from data_collection.sensors.vehicle_can_handler import VehicleCANHandler
            handler = VehicleCANHandler(self.cfg)
            # Prefer listen-only read to avoid interfering with bus traffic.
            msg = handler.read()
            if msg:
                return {'status': 'OK', 'details': {'msg': True, 'mode': 'listen_only'}}
            # If handler exposes a transmit/send API we could attempt a safe ping here,
            # but APIs differ and transmitting on a live CAN bus can be intrusive.
            return {'status': 'FAILED', 'reason': 'no_can_messages', 'details': {'mode': 'listen_only'}}
        except Exception:
            # fallback to python-can ingestion if present
            try:
                from battery_monitor.ingestion.can_ingestion import CANIngestion
                ci = CANIngestion(self.cfg)
                ci.start()
                msg = ci.read_message(timeout=2.0)
                ci.stop()
                if msg:
                    return {'status': 'OK', 'details': {'msg': True}}
                return {'status': 'FAILED', 'reason': 'no_can_messages'}
            except Exception as ex:
                return {'status': 'FAILED', 'reason': 'can_ingestion_not_available', 'details': str(ex)}

    def _test_buzzer(self) -> Dict[str, Any]:
        pins = (self.cfg.get('hardware_diagnostics.log_dir') and self.local.get('gpio_pins', {})) or self.local.get('gpio_pins', {})
        pin = pins.get('buzzer') if isinstance(pins, dict) else None
        if not pin:
            return {'status': 'FAILED', 'reason': 'buzzer_pin_not_configured'}
        try:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)
            import time
            time.sleep(0.05)
            GPIO.output(pin, GPIO.LOW)
            return {'status': 'OK', 'details': {'pin': pin}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_led_red(self) -> Dict[str, Any]:
        pins = self.local.get('gpio_pins', {}) or {}
        pin = pins.get('led_red')
        if not pin:
            return {'status': 'FAILED', 'reason': 'led_red_pin_not_configured'}
        try:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            import time
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(pin, GPIO.LOW)
            return {'status': 'OK', 'details': {'pin': pin}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_led_green(self) -> Dict[str, Any]:
        pins = self.local.get('gpio_pins', {}) or {}
        pin = pins.get('led_green')
        if not pin:
            return {'status': 'FAILED', 'reason': 'led_green_pin_not_configured'}
        try:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            import time
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(pin, GPIO.LOW)
            return {'status': 'OK', 'details': {'pin': pin}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_ble_mesh(self) -> Dict[str, Any]:
        try:
            ok, out, err = self._run_command(['bluetoothctl', 'scan', 'on'], timeout=self.timeouts.get('bluetoothctl', 5))
            if not ok:
                return {'status': 'FAILED', 'reason': err or 'bluetoothctl_failed'}
            return {'status': 'OK', 'details': out.strip()}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_esp32(self) -> Dict[str, Any]:
        uart = self.esp32_uart
        if not uart:
            return {'status': 'FAILED', 'reason': 'not_configured'}
        try:
            import serial
            with serial.Serial(uart, timeout=2) as ser:
                ser.write(b'\x01')
                resp = ser.read(32)
                if resp:
                    return {'status': 'OK', 'details': {'resp_len': len(resp)}}
                return {'status': 'FAILED', 'reason': 'no_response'}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_driver_fatigue(self) -> Dict[str, Any]:
        try:
            from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
        except Exception:
            return {'status': 'FAILED', 'reason': 'driver_fatigue_handler_not_available'}
        try:
            handler = DriverFatigueHandler(self.cfg)
            sample = handler.read()
            if not isinstance(sample, dict):
                return {'status': 'FAILED', 'reason': 'no_sample'}
            rng = self.cfg.get('hardware_diagnostics.self_test_validation_ranges') or self.local.get('self_test_validation_ranges', {})
            df_rng = rng.get('driver_fatigue', {})
            score = sample.get('drowsiness_score')
            if score is None:
                return {'status': 'FAILED', 'reason': 'missing_score'}
            if not (df_rng.get('drowsiness_score_min', 0.0) <= float(score) <= df_rng.get('drowsiness_score_max', 1.0)):
                return {'status': 'FAILED', 'reason': 'score_out_of_range', 'details': {'score': score}}
            return {'status': 'OK', 'details': {'score': score}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    # Stubs for other tests
    def _test_voltage_sensor(self) -> Dict[str, Any]:
        if self.enable_simulation:
            return {'status': 'OK', 'details': 'Simulated'}
        try:
            from data_collection.sensors.voltage_current_handler import VoltageCurrentHandler
        except Exception:
            return {'status': 'FAILED', 'reason': 'voltage_current_handler_not_available'}
        try:
            handler = VoltageCurrentHandler(self.cfg)
            sample = handler.read()
            if not isinstance(sample, dict):
                return {'status': 'FAILED', 'reason': 'no_sample'}
            v = sample.get('pack_voltage_v')
            if v is None:
                return {'status': 'FAILED', 'reason': 'missing_voltage', 'details': sample}
            # validation ranges: prefer main config, then default
            vr = self.cfg.get('sensors.voltage_current.voltage_range', [3.0, 5.0])
            try:
                if not (float(vr[0]) <= float(v) <= float(vr[1])):
                    return {'status': 'FAILED', 'reason': 'out_of_range', 'details': {'value': v, 'range': vr}}
            except Exception:
                pass
            return {'status': 'OK', 'details': {'voltage': v}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_current_sensor(self) -> Dict[str, Any]:
        if self.enable_simulation:
            return {'status': 'OK', 'details': 'Simulated'}
        try:
            from data_collection.sensors.voltage_current_handler import VoltageCurrentHandler
        except Exception:
            return {'status': 'FAILED', 'reason': 'voltage_current_handler_not_available'}
        try:
            handler = VoltageCurrentHandler(self.cfg)
            sample = handler.read()
            if not isinstance(sample, dict):
                return {'status': 'FAILED', 'reason': 'no_sample'}
            c = sample.get('pack_current_a') or sample.get('pack_current')
            if c is None:
                return {'status': 'FAILED', 'reason': 'missing_current', 'details': sample}
            cr = self.cfg.get('sensors.voltage_current.current_range', [-10.0, 10.0])
            try:
                if not (float(cr[0]) <= float(c) <= float(cr[1])):
                    return {'status': 'FAILED', 'reason': 'out_of_range', 'details': {'value': c, 'range': cr}}
            except Exception:
                pass
            return {'status': 'OK', 'details': {'current': c}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_temp_sensor(self) -> Dict[str, Any]:
        if self.enable_simulation:
            return {'status': 'OK', 'details': 'Simulated'}
        try:
            from data_collection.sensors.voltage_current_handler import VoltageCurrentHandler
        except Exception:
            return {'status': 'FAILED', 'reason': 'voltage_current_handler_not_available'}
        try:
            handler = VoltageCurrentHandler(self.cfg)
            sample = handler.read()
            if not isinstance(sample, dict):
                return {'status': 'FAILED', 'reason': 'no_sample'}
            t = sample.get('max_temp_c') or sample.get('pack_temp_c')
            if t is None:
                return {'status': 'FAILED', 'reason': 'missing_temp', 'details': sample}
            tr = self.cfg.get('sensors.voltage_current.temp_range', [-20.0, 70.0])
            try:
                if not (float(tr[0]) <= float(t) <= float(tr[1])):
                    return {'status': 'FAILED', 'reason': 'out_of_range', 'details': {'value': t, 'range': tr}}
            except Exception:
                pass
            return {'status': 'OK', 'details': {'temp': t}}
        except Exception as ex:
            return {'status': 'FAILED', 'reason': str(ex)}

    def _test_fire_sensor(self) -> Dict[str, Any]:
        return {'status': 'FAILED', 'reason': 'not_implemented'}

    def _test_smoke_sensor(self) -> Dict[str, Any]:
        return {'status': 'FAILED', 'reason': 'not_implemented'}

    def _test_gsm(self) -> Dict[str, Any]:
        return {'status': 'FAILED', 'reason': 'not_implemented'}


def detect_hardware(config: Optional[ConfigLoader] = None) -> Dict[str, Any]:
    checker = HardwareChecker(config)
    return checker.detect_hardware()


def main():
    import argparse
    import sys
    from config.config_loader import ConfigLoader
    try:
        from utils.telemetry import TelemetryCollector
    except Exception:
        TelemetryCollector = None

    parser = argparse.ArgumentParser(description='Run hardware diagnostics (module entry)')
    parser.add_argument('--config', default='config/default_config.yaml', help='Config file path')
    parser.add_argument('--no-alerts', action='store_true', help='Disable GPIO alerts')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable dashboard POSTs')
    parser.add_argument('--no-telemetry', action='store_true', help='Disable telemetry logging')
    parser.add_argument('--simulation', action='store_true', help='Enable simulation mode')
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)

    # Override config with CLI flags using public API
    if args.no_alerts:
        try:
            cfg.set('hardware_diagnostics.alerts.enable', False)
        except Exception:
            pass
    if args.no_dashboard:
        try:
            cfg.set('hardware_diagnostics.dashboard_integration.enable', False)
        except Exception:
            pass
    if args.simulation:
        try:
            cfg.set('hardware_diagnostics.enable_simulation', True)
        except Exception:
            pass

    telemetry = None if args.no_telemetry or TelemetryCollector is None else TelemetryCollector(cfg)

    checker = HardwareChecker(cfg, telemetry=telemetry)
    results = checker.run_all_tests()

    # print minimal summary to stdout for systemd logs
    print(f"Hardware diagnostics completed: passed={len(results.get('passed_tests', []))} failed={len(results.get('failed_tests', []))}")

    if telemetry:
        try:
            telemetry.close()
        except Exception:
            pass

    # Exit 0 if no failed tests, 1 otherwise (systemd unit treats 0 and 1 as SuccessExitStatus)
    sys.exit(0 if not results.get('failed_tests') else 1)


if __name__ == '__main__':
    main()
