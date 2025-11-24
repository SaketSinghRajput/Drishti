# Deployment

This directory contains deployment artifacts and helpers for edge devices (Raspberry Pi 4, Jetson Nano, x86 dev).

Contents:
- `docker/` - Dockerfiles and docker-compose samples for Pi4, Jetson, and development images.
- `systemd/` - systemd unit files to run services at boot.
- `hardware/` - udev rules and hardware setup scripts (Coral EdgeTPU, CAN, camera V4L2).
- `scripts/` - helper scripts for cross-compilation, model compilation, and install.
- `monitoring_dashboard/` - Flask-based system monitoring dashboard (CPU/GPU/memory, inference stats).

### Hardware Diagnostics (`hardware_diagnostics/`)

Boot-time hardware detection module that scans for connected sensors and peripherals, logs results, and returns status for integration with monitoring dashboards.

Key files:
- `checker.py`: Main `HardwareChecker` class with `detect_hardware()` method
- `config.yaml`: Hardware mappings and detection settings
- `tests.py`: Pytest suite with mocked hardware
- `README.md`: Detailed usage and troubleshooting guide

Current scope (Phase 1): Detection only — scans I2C, SPI, UART, GPIO, CAN, Camera, BLE, ESP32 and compares against expected sensors from config.

Quick start:
```
# Run manual check
python -m deployment.hardware_diagnostics.checker

# View logs
tail -f /home/pi/system_logs/hardware_check.log
```

Quick start (manual):

1. Review `deployment/scripts/install.sh` for prerequisites and platform-specific steps.
2. Adjust `config/default_config.yaml` for your target platform.
3. For Docker-based deployment, build the image for your platform and start via `docker-compose`.

See `docs/deployment.md` for full instructions and troubleshooting.
