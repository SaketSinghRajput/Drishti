# Telemetry System

This document describes the telemetry subsystem: buffered, anonymized JSONL logging of predictions, outcomes and false alarms, optional AES-256-GCM encryption at rest, and an offline-first cloud sync for aggregated summaries.

## Overview
- `utils.telemetry.TelemetryCollector` buffers entries and flushes to `telemetry_YYYYMMDD_HHMMSS.jsonl` files.
- Optional encryption (`telemetry.encrypt`) writes temporary plaintext and encrypts to `.jsonl.enc` files using AES-GCM.
- `utils.cloud_sync.CloudSyncManager` aggregates telemetry files, anonymizes PII, and uploads summaries to a configured endpoint.
- `utils.video_retention.VideoRetentionManager` stores event-triggered camera frames with retention and optional encryption.
- `utils.data_aggregator.DataAggregator` prepares retraining datasets by matching predictions with outcomes and applying balancing.

## JSONL schema
Each line is a JSON object. Examples:
```
{
  "timestamp": "2025-01-01T00:00:00Z",
  "type": "prediction",
  "risk_score": 0.7,
  "alert_level": "high",
  "features": { ... },
  "sensor_health": { ... },
  "metadata": { "vehicle_hash": "<hashed>" }
}
```

## Configuration
Edit `config/default_config.yaml` under the `telemetry`, `privacy`, and `retraining` sections.

Important keys:
- `telemetry.output_dir` — directory for telemetry files
- `telemetry.encrypt` — enable AES-256-GCM encryption at rest
- `telemetry.encryption_key_env` / `telemetry.encryption_key_file` — key sources
- `telemetry.cloud_sync.*` — cloud sync configuration
- `privacy.video_retention_hours` — how long to keep retained frames

## CLI
- `scripts/sync_telemetry.py` — uploads aggregated summaries to the configured endpoint (supports `--dry-run` and overrides)

## Security
- Use `TELEMETRY_ENCRYPTION_KEY` env var or a secure key file. Keys should be rotated periodically.
- TelemetryCollector hashes `vehicle_hash` and `trip_id` when `anonymize_ids` is enabled.

## Examples
See `examples/telemetry_example.py` for a usage sample.
