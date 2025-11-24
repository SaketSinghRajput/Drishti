# Deployment Guide

This document provides a concise deployment guide for Raspberry Pi 4 and Jetson-based devices.

Prerequisites
- A Linux host (Raspberry Pi OS / Ubuntu for Jetson)
- Python 3.9 or compatible
- For Coral EdgeTPU: install EdgeTPU runtime packages or use the Docker image that includes them
- For Jetson: use NVIDIA L4T TensorFlow container or native packages

Quickstart (Docker / Pi4)
1. Build image on a machine with aarch64 builder support (or build on-device):
   - `docker build -f deployment/docker/Dockerfile.pi4 -t ev-safety:pi4 .`
2. Run with devices mounted (example):
   - `docker run --privileged --device /dev/can0 --device /dev/video0 --device /dev/gpiomem -v ./models:/models:ro -v ./config:/config:ro -p 5000:5000 ev-safety:pi4`

Systemd (manual)
1. Copy systemd unit files from `deployment/systemd/` to `/etc/systemd/system/`.
2. Copy config to `/etc/ev-safety-guardian/` and edit `config.yaml`.
3. Run `systemctl daemon-reload` then `systemctl enable --now ev-safety-inference`.

### Hardware Diagnostics Service

The repository includes a boot-time hardware diagnostics service that verifies connected
hardware before the main inference service starts. Two units are provided:

- `hardware-diagnostics.service` — a oneshot service that runs the module entrypoint `python -m deployment.hardware_diagnostics.checker` (accepts the same CLI flags as the CLI wrapper).
- `hardware-diagnostics.timer` — an optional timer that re-triggers the service every 5 minutes.

Enable and start them manually:

```bash
sudo systemctl enable hardware-diagnostics.service
sudo systemctl enable hardware-diagnostics.timer
sudo systemctl start hardware-diagnostics.timer
# Check status and logs
sudo systemctl status hardware-diagnostics.service
journalctl -u hardware-diagnostics.service -n 50
```

The installer script `deployment/scripts/install.sh --enable-services` will copy and enable
these units for you when run on the target device (Raspberry Pi). The service is configured
with `SuccessExitStatus=0 1` so a failure during checks will not block boot; failed modules
are reported and can be queried by other components to gracefully degrade functionality.

Alternative: Health Endpoint Polling

If you prefer to poll the diagnostics status from an external monitor instead of using the
systemd timer, enable the health API in the main config and poll the endpoint:

```yaml
hardware_diagnostics:
   health_api:
      enable: true
      port: 5002
```

Then poll `http://localhost:5002/health` from your monitoring system. See
`deployment/hardware_diagnostics/README.md` for details on the health payload and examples.

Notes
- The included `ev-safety-inference.service` assumes a system user `evguardian` exists; the `install.sh` script will attempt to create it.
- The service is conservative with `Restart=on-failure` and `RestartSec=10` to avoid rapid restart loops.

Hardware
- Coral EdgeTPU: copy the `deployment/hardware/99-coral-tpu.rules` to `/etc/udev/rules.d/` and reload udev.
- CAN: use `deployment/hardware/can-setup.sh can0 500000` to bring up a CAN interface.
- Camera: use the `camera-v4l2.conf` snippet to prefer MJPG for reduced CPU.

Scripts
- `deployment/scripts/install.sh` - basic installer (creates venv, installs requirements, copies units). Review before running.
- `deployment/scripts/compile_edgetpu_model.sh` - wrapper around `edgetpu_compiler`.

Cross-compile and Build
- Use `deployment/scripts/cross_compile_onnx.sh` to prepare a buildx/qemu environment for building aarch64 images.

EdgeTPU notes
- Coral EdgeTPU runtime packages (`libedgetpu1-*`) are provided by Coral's apt repos. For Debian Bookworm / Raspberry Pi OS you must add the Coral apt key and repository prior to installing; see Coral documentation for the correct repo and package name for your device.

Troubleshooting
- If Coral is not present, set `inference.use_edgetpu: false` in config to fall back to CPU.
- If camera fails, check `v4l2-ctl --list-formats-ext` and adjust `preferred_pixel_format`.

## Hardware Diagnostics

The project includes a boot-time hardware checker in `deployment/hardware_diagnostics/`.
This module scans I2C, SPI, UART, GPIO, CAN, camera devices and BLE adapters and
compares findings against the configured expected sensors. Results are logged to
`/home/pi/system_logs/hardware_check.log` and a JSON summary is written next to the log.

Usage:

```bash
# Manual check
python -m deployment.hardware_diagnostics.checker

# or in Python
from deployment.hardware_diagnostics import HardwareChecker
from config.config_loader import ConfigLoader
cfg = ConfigLoader('config/default_config.yaml')
checker = HardwareChecker(cfg)
status = checker.detect_hardware()
print(status)
```

Configuration for the checker is available in `deployment/hardware_diagnostics/config.yaml` with
detection mappings and timeouts. The main config's `hardware_diagnostics:` section allows enabling
simulation mode and changing the log location.

### Hardware Self-Tests

The hardware diagnostics module now includes automated self-tests (Phase 2) to verify functional
operation of detected components. Self-tests perform a short functional check (read sample, validate
ranges, blink GPIO, etc.) and are protected by per-test timeouts (default 5s).

Run self-tests programmatically:

```python
from deployment.hardware_diagnostics import HardwareChecker
from config.config_loader import ConfigLoader
cfg = ConfigLoader('config/default_config.yaml')
checker = HardwareChecker(cfg)
results = checker.run_all_tests()
print(results['passed_tests'], results['failed_tests'])
```

Key configuration options are in `deployment/hardware_diagnostics/config.yaml` under
`self_test_timeouts` and `self_test_validation_ranges`. For development without hardware, set
`hardware_diagnostics.enable_simulation: true` in your main config to mark all tests as OK.
