**Battery Monitoring Dashboard**

This document describes the Flask-based dashboard located at `battery_monitor/dashboard`.

See also `docs/battery_schemas.md` for API and storage schema definitions.

Usage
- Start the dashboard with `python -m battery_monitor.dashboard.app` or use `create_app()` to embed it in another process.
- Default host:port is `0.0.0.0:5002` and configurable via `config/default_config.yaml` under `battery_monitor.dashboard`.

APIs
- `GET /api/telemetry/latest` — latest sample with `computed_metrics`.
- `GET /api/telemetry/history` — historical snapshots (from EventLogger or in-memory state).
- `GET /api/alerts` — recent alerts (optionally filter by `severity`).
- `GET /api/alerts/<id>` — alert details + linked snapshot.
- `POST /api/telemetry/update` — internal endpoint to push latest telemetry into dashboard state.

History response shape
- `GET /api/telemetry/history` returns a JSON object with a flattened `history` array. Each element is an object with the following fields:
	- `alert_id`: integer or `null` — the `battery_alerts.id` that referenced the snapshot (or `null` for in-memory fallback)
	- `alert_ts`: string or `null` — timestamp of the alert that referenced the snapshot
	- `sample`: the original telemetry sample (object) as stored in the snapshot

Example:

```
{
	"history": [
		{ "alert_id": 42, "alert_ts": "2025-11-21T10:00:01Z", "sample": { "timestamp": "2025-11-21T10:00:00Z", "pack_voltage": 48.5 } },
		{ "alert_id": null, "alert_ts": null, "sample": { "timestamp": "2025-11-21T10:00:05Z", "pack_voltage": 48.6 } }
	]
}
```

See `docs/battery_schemas.md` for the canonical schema definitions used by the dashboard and storage layers.

Front-end
- The UI (templates/index.html) uses Chart.js for line and bar charts and polls the APIs every 1s (telemetry) / 5s (alerts).

Integration
- The dashboard reads historical data from `EventLogger` if provided at app creation. To enable, pass an `EventLogger` instance to `create_app(event_logger=...)`.
