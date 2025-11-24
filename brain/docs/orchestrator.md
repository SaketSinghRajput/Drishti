# Orchestrator: Multimodal AI Pipeline Brain

This document describes `main.py`, the root orchestrator that initializes and
coordinates hardware diagnostics, sensor collection, BMS monitoring, ML
inference, dashboard, BLE safety broadcasting, alerts, and telemetry.

See `main.py` for the implementation and `deployment/systemd/ev-safety-orchestrator.service`
for the systemd unit used on production devices.

## Overview

`main.py` performs a boot-time hardware check, then starts long-running
subsystems and a 1-second loop that:

- Collects current sensor window from the DataLogger
- Runs inference via `EdgeInferenceEngine`
- Processes BMS telemetry via `BMSMonitor`
- Builds a BLE safety packet and calls `utils.ble_utils.broadcast_safety_packet()`
- Updates the dashboard state
- Triggers alerts via the AlertManager
- Logs telemetry

The orchestrator is defensive: missing subsystems are tolerated and stubbed in
`main.py` to enable development on non-Pi hosts.

## Running

Development:

```bash
python scripts/run_orchestrator.py --mode dev --config config_examples/dev_sim.yaml
```

Production (systemd):

```bash
sudo systemctl enable ev-safety-orchestrator.service
sudo systemctl start ev-safety-orchestrator.service
sudo journalctl -u ev-safety-orchestrator -f
```

## BLE Safety Packet

See `utils/ble_utils.broadcast_safety_packet()` for the schema and usage. The
orchestrator writes safety packets to `data/ble_safety_outbox.jsonl` (dev-mode)
and attempts BLE advertising when supported.

## Configuration

Key orchestrator config lives under `orchestrator:` in the config file. See
`config_examples/dev_sim.yaml` for a development-ready example.

## Troubleshooting

- If the dashboard fails to start, check for port conflicts and `journalctl` logs.
- If BLE broadcast is not visible, ensure the platform supports advertising and
  that `bleak` or appropriate system utilities are installed.

