# Driver Fatigue Integration

This document explains the integrated outputs from the driver fatigue detector:
BLE broadcast (file outbox), REST API, SQLite logging, and central event push.

## Overview

- BLE Broadcast: write compact JSON packets to an outbox file (`data/ble_fatigue_outbox.json`) for V2V/nearby systems.
- REST API: small Flask app exposing `/status`, `/events`, and `/query` for quick checks.
- SQLite Logging: per-sample persistence to `data/fatigue_events.db` for auditing and export.
- Central Push: optional HTTP POST of events to a fleet endpoint (config-gated).

## BLE Packet Format

Each line in `data/ble_fatigue_outbox.json` is a JSON object:

```
{
  "timestamp": "ISO8601 UTC",
  "vehicle_id": "8-char-hash",
  "status": "FatiguedDriver",
  "state": "DROWSY",
  "score": 0.72,
  "reason": "EAR below threshold"
}
```

Vehicle identifiers are hashed using `ble_utils.hash_node_id()` with the configured salt.

### BLE Backend & Advertising

- `ble_broadcast.ble_backend`: choose `auto`, `bleak`, `pybluez`, or `stub`. `auto` will attempt to use available platform backends.
- `ble_broadcast.tx_power`: transmit power in dBm (integer). Typical range: -30 (min) to +20 (max). Use 0 for default.
- `ble_broadcast.service_uuid`: service UUID advertised in BLE packets for discovery.
- `ble_broadcast.manufacturer_id`: manufacturer-specific ID for compact payloads.

When a real BLE backend is available and configured, the handler will attempt best-effort advertising. If advertising fails due to missing dependencies, permissions, or platform limitations, the file outbox is still written as a fallback for debugging and CI.

## REST API

Endpoints:

- `GET /status` — returns latest `fatigue_score`, `state`, `last_event`, `timestamp`.
- `GET /events?limit=N` — returns recent events (default limit configured).
- `POST /query` — JSON body `{ "check": "drowsy" }` returns `{ "result": true/false }`.

Run the API with `python scripts/run_fatigue_api.py` (default port 5001).

## SQLite Logging

Table `fatigue_events` schema (columns):
- id, timestamp, ear, mar, head_pitch, score, state, event_flag, vehicle_hash, trip_id

Use `FatigueLogger.get_recent_events(limit)` to read recent entries and
`FatigueLogger.export_to_csv()` to write CSV if `enable_csv_export` is true.

## Central Push

If enabled, the handler will POST event JSON to the configured endpoint with
`timeout_s` and `retry_attempts` configured under `sensors.driver_fatigue.central_push`.

## Configuration Keys

See `config/default_config.yaml` under `sensors.driver_fatigue` for defaults and comments.

## Privacy & Security

- Vehicle IDs are hashed before writing to outbox or DB.
- Raw frames are never written by these modules.
- Telemetry integration should handle encryption for cloud sync.

## Troubleshooting

- Port conflicts: change `sensors.driver_fatigue.fatigue_api.port`.
- Missing dependencies: `Flask` and `requests` are required to run the API and central push.
- Central endpoint down: central push is retried per config and failures are logged but not propagated.

