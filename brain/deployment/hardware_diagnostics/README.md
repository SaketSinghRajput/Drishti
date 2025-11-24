# Hardware Diagnostics

This module performs boot-time hardware detection for EV Safety Guardian.

Overview

The `hardware_diagnostics` package detects connected I2C, SPI, UART, GPIO, CAN,
camera devices and BLE adapters and compares them against the configured
`expected_sensors` defined in `config/default_config.yaml`.

Files

- `checker.py` — `HardwareChecker` implementation with `detect_hardware()`.
- `config.yaml` — detection timeouts and hardware-to-sensor mappings.
- `tests.py` — unit tests that mock subprocess calls and hardware.

Usage

```python
from deployment.hardware_diagnostics import HardwareChecker
from config.config_loader import ConfigLoader

cfg = ConfigLoader('config/default_config.yaml')
checker = HardwareChecker(cfg)
status = checker.detect_hardware()
print(status)
```

Logging

Results are written to `/home/pi/system_logs/hardware_check.log` by default and
summaries are written as JSON files into the same directory.

Testing

Run the test module with pytest (mocks are used so no hardware required):

```bash
pytest deployment/hardware_diagnostics/tests.py
```

Roadmap

Phase 1: Detection (completed) — non-intrusive scans and mapping.
Phase 2: Self-tests (added) — functional checks, range validation, GPIO blink tests.
Phase 3: Alerts and dashboard integration.
Phase 4: systemd integration and scheduled re-checks.

Phase 3: Alerts and Dashboard Integration

Phase 3 adds alert actuation, status exposure, dashboard integration, telemetry logging, and fail-safe flags.

Features

1. **GPIO Alerts**: On test completion, trigger buzzer/LED based on results:
	- **Failure**: Continuous slow buzzer beeps (3x 0.5s), RED LED blink (GPIO 17)
	- **Success**: GREEN LED blink (GPIO 27, 3x 0.3s)
	- Uses `_DummyGPIO` fallback for dev/CI (no hardware required)

2. **Status Exposure**:
	- `get_system_health_status() -> Dict`: Returns JSON health dict with overall_status, components, disabled_modules
	- Optional Flask `/health` endpoint (port 5002) via `health_api.py`

3. **Dashboard Integration**:
	- POSTs health status to main dashboard (`http://localhost:5000/api/hardware_health`)
	- POSTs to monitoring dashboard (`http://localhost:5003/api/hardware_health`)
	- Graceful failure handling (logs warnings, continues)

4. **Telemetry Logging**:
	- Integrates with `utils/telemetry.py` via `log_hardware_check()`
	- Logs summary (passed/failed/skipped tests) with anonymization

5. **Fail-Safe Flags**:
	- `get_disabled_modules() -> List[str]`: Returns list of failed/skipped components
	- `is_module_enabled(component: str) -> bool`: Check if component is functional
	- Other modules can query to gracefully degrade (e.g., skip camera if disabled)

Configuration

Add to `config/default_config.yaml`:

```yaml
hardware_diagnostics:
  alerts:
	 enable: true
	 gpio_buzzer_pin: 18
	 gpio_led_red_pin: 17
	 gpio_led_green_pin: 27
  dashboard_integration:
	 enable: true
	 main_dashboard_url: http://localhost:5000
	 monitoring_dashboard_url: http://localhost:5003
  telemetry:
	 enable: true
  health_api:
	 enable: false  # Optional standalone API
	 port: 5002
```

Example Usage

```python
from deployment.hardware_diagnostics import HardwareChecker
from config.config_loader import ConfigLoader
from utils.telemetry import TelemetryCollector

cfg = ConfigLoader('config/default_config.yaml')
telemetry = TelemetryCollector(cfg)
checker = HardwareChecker(cfg, telemetry=telemetry)

# Run tests (triggers alerts, POSTs to dashboards, logs to telemetry)
results = checker.run_all_tests()

# Get health status
health = checker.get_system_health_status()
print(f"Overall: {health['overall_status']}")
print(f"Disabled: {health['disabled_modules']}")

# Check if module is enabled
if not checker.is_module_enabled('camera'):
	 print("Camera disabled, using mock data")

telemetry.close()
```

Optional Health API

Start standalone Flask app:

```bash
python -m deployment.hardware_diagnostics.health_api
# Access: http://localhost:5002/health
```

## Phase 4: Systemd Integration and Scheduled Re-checks

Phase 4 enables automatic hardware checks on Raspberry Pi boot and periodic re-checks via systemd.

### Features

1. **Boot-Time Service**: `hardware-diagnostics.service` runs before main inference service to verify hardware. The service invokes the module entrypoint:

```bash
python -m deployment.hardware_diagnostics.checker --config /etc/ev-safety-guardian/default_config.yaml
```
This mirrors the CLI wrapper behavior and accepts the same flags (`--no-alerts`, `--no-dashboard`, `--simulation`).
2. **Periodic Timer**: `hardware-diagnostics.timer` triggers re-checks every 5 minutes (configurable).
3. **Fail-Safe Boot**: Service exits with success even on failures (`SuccessExitStatus=0 1`) to allow boot continuation; failed modules are reported via `get_disabled_modules()`.
4. **Systemd Logging**: Results logged to systemd journal (`journalctl -u hardware-diagnostics`) and to the configured log file.

### Installation

Use the installer to enable services automatically:

```bash
sudo deployment/scripts/install.sh --enable-services
```

Manual installation:

```bash
sudo cp deployment/systemd/hardware-diagnostics.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hardware-diagnostics.service
sudo systemctl enable hardware-diagnostics.timer
sudo systemctl start hardware-diagnostics.timer
```

### Usage & Troubleshooting

Check service status and recent logs:

```bash
sudo systemctl status hardware-diagnostics.service
journalctl -u hardware-diagnostics.service -n 50
```

Disable periodic checks (keep boot-time only):

```bash
sudo systemctl disable hardware-diagnostics.timer
sudo systemctl stop hardware-diagnostics.timer
```

For more details and configuration options see `docs/deployment.md`.

Self-Tests (Phase 2)

The package now supports `self_test(component)` and `run_all_tests()` on `HardwareChecker`.
Self-tests use existing sensor handlers where available (IMU, GPS, Camera, BMS) and perform
short functional checks with a 5s default timeout. For development, enable simulation mode to
return OK for all tests without touching hardware.

Example:

```python
from deployment.hardware_diagnostics import HardwareChecker
from config.config_loader import ConfigLoader
cfg = ConfigLoader('config/default_config.yaml')
checker = HardwareChecker(cfg)
print(checker.self_test('imu'))
print(checker.run_all_tests())
```
