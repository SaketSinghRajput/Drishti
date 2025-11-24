**Battery Alert System**
- **Purpose**: Persistent storage, actuation, and developer-facing BLE outbox for battery health alerts.

**Components**
- `AlertManager`: orchestrates visual/GPIO indications, BLE broadcast, and DB persistence.
- `BLEBroadcastStub`: development stub that appends JSON packets to a file outbox.
- `EventLogger`: sqlite-backed storage for snapshots and alerts.

**EventLogger schema (canonical)**
- `battery_snapshots` (id, ts, snapshot_json, features_json, scores_json)
  - Stores raw telemetry (`snapshot_json`) alongside extracted features and ML scores.
- `battery_alerts` (id, ts, severity, reason, alert_json, snapshot_id)
  - Stores high-level alert metadata and references a snapshot row containing features/scores.

Rationale: Features/scores are stored with the snapshot so a single snapshot row represents the state (telemetry + derived features + scores) observed when the alert occurred. `battery_alerts` remains lightweight and suitable for indexing and listing.

**BLE outbox schema**
- File: configurable as `battery_monitor.alerts.ble_outbox_path` (default `data/ble_outbox.json`).
- Each line is a JSON object with keys: `timestamp`, `vehicle_id`, `severity`, `reason`, `anomaly_score`.

**ExplanationLogger integration**
- Decision: Battery alerts are not logged to `ExplanationLogger`. The canonical stores are `TelemetryCollector.log_battery_alert()` and the new `EventLogger`. This keeps explanation logs focused on model predictions and their explanations; battery alert data (including features/scores) is retained in `EventLogger` for post-hoc analysis.

**Usage**
- BMSMonitor calls `TelemetryCollector.log_battery_alert()` and then `AlertManager.trigger_alert()` for each alert. `AlertManager` will persist via `EventLogger` and broadcast via `BLEBroadcastStub` for `CRITICAL` severity.

**Config keys** (in `config/default_config.yaml`)
- `battery_monitor.alerts.*`: log_path, shutdown_log_path, ble_outbox_path, ble_broadcast_enabled, gpio_* pins, alert_history_size
- `battery_monitor.storage.*`: db_path, retention_days

**Testing**
- Unit tests for `AlertManager`, `BLEBroadcastStub`, and `EventLogger` are included in `tests/test_battery_monitor/`.
