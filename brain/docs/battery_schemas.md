# Battery Schemas

This document is the canonical schema reference for telemetry, alerts, storage, and BLE outbox packets used by the battery monitoring system.

1) BMS telemetry JSON (per-sample)

Fields (typical):
- `timestamp` (string, ISO 8601 UTC) — example: `2025-11-21T10:00:00Z`
- `pack_voltage` (number, V) — total pack voltage, e.g. `48.5`
- `pack_current` (number, A) — pack current, positive = discharge
- `soc` (number, percent 0-100)
- `soh` (number, percent 0-100)
- `cell_voltages` (array[number]) — per-cell voltages in V, e.g. `[3.7, 3.71, ...]`
- `cell_temps` (array[number]) — per-cell or per-module temperatures in °C, e.g. `[25.3, 26.1, ...]`
- `metadata` (object) — optional, e.g. `{ "vehicle_hash": "abc123" }`

Notes: consumers should tolerate missing optional fields. Typical ranges: pack_voltage 0–1000 V (system dependent), cell voltages 0–5 V, temps -40–120 °C.

2) Alert JSON schema (emitted by `BMSMonitor` / `TelemetryCollector`)

Top-level alert object (example):

{
  "source": "rule" | "ml",
  "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "reason": "voltage_diff" | "cell_overtemp" | "ML anomaly" | ...,
  "rule": "r1" (optional),
  "value": 3.2 (numeric, optional),
  "anomaly_score": 0.92 (optional numeric),
  "rul_hours": 12 (optional numeric)
}

Alerts are persisted by `EventLogger` and may be broadcast via BLE for critical events.

3) SQLite schema used by `EventLogger`

Tables:

- `battery_snapshots`:
  - `id` INTEGER PRIMARY KEY AUTOINCREMENT
  - `ts` TEXT NOT NULL (ISO 8601 UTC)
  - `snapshot_json` TEXT NOT NULL — JSON-encoded list of telemetry samples (see per-sample schema)
  - `features_json` TEXT — JSON object with extracted features
  - `scores_json` TEXT — JSON object containing ML scores (e.g., `{ "anomaly_score": 0.9 }`)

- `battery_alerts`:
  - `id` INTEGER PRIMARY KEY AUTOINCREMENT
  - `ts` TEXT NOT NULL (ISO 8601 UTC)
  - `severity` TEXT
  - `reason` TEXT
  - `alert_json` TEXT NOT NULL — JSON-encoded alert object
  - `snapshot_id` INTEGER — foreign key into `battery_snapshots(id)` or `NULL` if not linked

Example snapshot row (conceptual):

{
  "ts": "2025-11-21T10:00:00Z",
  "snapshot_json": "[{\"timestamp\":\"2025-11-21T10:00:00Z\", \"pack_voltage\":48.5, \"cell_voltages\":[3.7,3.8]}]",
  "features_json": "{\"f1\":1.2}",
  "scores_json": "{\"anomaly_score\":0.9}"
}

Example alert row (conceptual):

{
  "ts": "2025-11-21T10:00:01Z",
  "severity": "HIGH",
  "reason": "voltage_diff",
  "alert_json": "{\"source\":\"rule\",\"severity\":\"HIGH\",\"reason\":\"voltage_diff\"}",
  "snapshot_id": 42
}

4) BLE outbox packet schema (newline-delimited JSON written by BLE stub)

Each line is a JSON object with these fields:
- `timestamp` (ISO 8601 UTC)
- `vehicle_id` (string) — hashed or anonymized vehicle id
- `severity` (string)
- `reason` (string)
- `anomaly_score` (number, optional)

Example line:

{"timestamp":"2025-11-21T10:00:01Z","vehicle_id":"a1b2...","severity":"CRITICAL","reason":"cell_overtemp","anomaly_score":0.95}

5) Dashboard REST shapes

- `GET /api/telemetry/latest` — returns the latest sample JSON (same shape as per-sample telemetry) augmented with `computed_metrics`:
  - `computed_metrics.voltage_diff`, `min_cell_voltage`, `max_cell_voltage`, `max_cell_temp` when available.

- `GET /api/telemetry/history` — returns a flattened list of samples with alert metadata:

  {
    "history": [
      { "alert_id": 123, "alert_ts": "2025-...", "sample": { ... } },
      { "alert_id": null, "alert_ts": null, "sample": { ... } }
    ]
  }

  - When `EventLogger` is provided, `alert_id` corresponds to the `battery_alerts.id` value that referenced the snapshot. `alert_ts` is the alert timestamp. When no `EventLogger` is present, the in-memory fallback returns `alert_id: null` for each sample.

Reference: see `docs/battery_dashboard.md` for API usage and examples.
