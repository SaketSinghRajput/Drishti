# Battery Health Monitoring

## Overview

The Battery Health Monitoring module provides real-time BMS telemetry ingestion,
rule-based alarms, ML-based anomaly detection and Remaining Useful Life (RUL)
prediction, a Flask-based dashboard for visualization, and event storage for
post-hoc analysis. This Phase 1 document covers architecture, configuration
and integration points for subsequent implementation phases.

## Architecture

Data flow (text diagram):

```
BMS (CAN) --> CAN Parser --> Rule Engine --> Alerts --> Dashboard / Storage
                        |--> ML Inference (anomaly/RUL) --> Alerts
```

Components:
- ingestion: CAN parser and CSV simulator (`battery_monitor/ingestion`)
- rules: deterministic checks for temp/voltage/current
- ml: anomaly detection and RUL prediction (stubs)
- alerts: GPIO, buzzer, BLE broadcast and event logging
- storage: SQLite-backed event/snapshot storage
- dashboard: Flask app serving real-time graphs and REST API

## External Dependencies

- **LibreSolar bms-c1**: CAN protocol and message formats. See
  https://github.com/LibreSolar/bms-c1 for CAN ID mappings and firmware docs.
- **rpi-battery-monitor**: Dashboard UI components used as a reference:
  https://github.com/kmcallister/rpi-battery-monitor

## Configuration

The module is configured under `battery_monitor` in `config/default_config.yaml`.
Key sections:
- `ingestion`: CAN interface, bitrate, simulation flag and CSV path
- `rules`: thresholds and severity mapping
- `ml`: model path, enable flag, thresholds
- `alerts`: GPIO pins, BLE settings, log path
- `storage`: SQLite DB path and retention
- `dashboard`: Flask host/port and update interval

Example values are provided in `config/default_config.yaml`.

## Data Schema

Telemetry JSON documents must conform to `config/bms_schema.json`. Example:

```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "pack_voltage": 48.12,
  "pack_current": -3.2,
  "soc": 72.5,
  "soh": 99.1,
  "cell_voltages": [3.7, 3.71, 3.69],
  "cell_temps": [25.1, 25.0, 24.8],
  "balancer_status": [0,1,0],
  "fault_codes": []
}
```

## Simulation Mode

Use `battery_monitor/ingestion/csv_simulator.py` to generate CSV-based telemetry
for development without hardware. The simulator produces columns matching the
schema and supports configurable noise and fault injection.

## Integration with Main System

Future integration points:
- Feed battery alerts into the main risk scoring pipeline (e.g., high temp
  increases risk score).
- Log battery events to the central telemetry pipeline with anonymization.
- Expose dashboard endpoints for remote monitoring and diagnostics.

## Hardware Requirements

- Raspberry Pi 4 (recommended) or compatible SBC
- PiCAN2 or USB-CAN adapter
- LibreSolar BMS or compatible CAN-enabled BMS
- Buzzer and LED for local alerts

## Development Roadmap

1. Implement CAN ingestion and mapping to `config/bms_schema.json`.
2. Implement RuleEngine and integration with AlertManager.
3. Integrate ML models for anomaly detection and RUL prediction.
4. Add dashboard templates and static assets from rpi-battery-monitor.

## Implemented Rules (details)

The current RuleEngine implements the following checks (see `battery_monitor/rules/rule_engine.py`):

- `max_cell_temp`
  - Logic: compute `max(cell_temps)` and compare to `battery_monitor.rules.thresholds.max_cell_temp_c` (default 70.0 C).
  - On exceedance, returns a rule alert with `rule_name: max_cell_temp` and `value` equal to the max temperature.

- `voltage_diff`
  - Logic: compute `max(cell_voltages) - min(cell_voltages)` and compare to `voltage_diff_threshold_v` (default 0.25 V).
  - Returns `rule_name: voltage_diff` and `value` equal to the voltage difference.

- `current_spike`
  - Logic: absolute pack current compared to `current_spike_threshold_a` (default 100 A).
  - Returns `rule_name: current_spike` with `value` equal to the absolute current.

- `soh_drop_rate`
  - Logic: maintains a deque of recent SoH samples and computes drop-per-day; compares to `soh_drop_rate_percent_per_day`.
  - This check is opt-in through `battery_monitor.rules.enable_soh_tracking` to avoid memory/compute overhead when not desired.

Severity mapping is computed using exceedance ratios mapped to the `battery_monitor.rules.severities` thresholds.

## ML Features and Inference

Features are computed in `battery_monitor/ml/feature_extractor.py`. Representative features include:

- `cell_voltage_mean`, `cell_voltage_std`, `cell_voltage_min`, `cell_voltage_max`
- `voltage_imbalance = max(cell_voltages) - min(cell_voltages)`
- `cell_temp_mean`, `cell_temp_std`, `temp_gradient = max(cell_temps) - min(cell_temps)`
- `pack_current_rms` (RMS over current samples)
- `soc_mean`, `soh_mean`, `fault_count_total`, `sample_count`

Phase 1 inference (`battery_monitor.ml.inference_stub.BatteryMLInference`) provides a deterministic fallback:
- `predict_anomaly(features)` returns `{'is_anomaly': bool, 'anomaly_score': float}` using a simple heuristic such as combining `voltage_imbalance` and `temp_gradient`.
- `predict_rul(features)` returns a heuristic RUL in hours (e.g., `soh * 10` when `soh < 80`) or `None` when unavailable.

For production, load a trained model via `joblib.load(model_path)` and use `decision_function`/`predict` to compute anomaly scores; normalize outputs into `[0,1]`.

## Alert Integration and Schema

Alerts emitted by `BMSMonitor.process_telemetry()` are logged to telemetry via `TelemetryCollector.log_battery_alert()` and follow this schema (documented earlier):

- `type`: `battery_alert`
- `source`: `rule` or `ml`
- `severity`: severity label
- `reason`: human-readable reason
- `value`: numeric value (rule-specific or anomaly score for ML)
- `anomaly_score`: numeric or null
- `telemetry_snapshot`: list of recent BMS samples

TelemetryCollector anonymizes identifiers in `metadata` (e.g., `vehicle_hash`) and grids locations when `telemetry.anonymize_location` is enabled.

## Testing Guide

Unit tests for the monitor are provided in `tests/test_battery_monitor/test_bms_monitor.py` and assert:

- Rule-only scenarios produce rule alerts and calls to `TelemetryCollector.log_battery_alert()` with `telemetry_snapshot`.
- ML-only scenarios produce ML alerts with `anomaly_score` and `rul_hours` fields and logged telemetry.
- Combined scenarios produce both alerts and multiple log calls.
- Edge cases: empty samples, disabled ML, missing fields are covered.

Run the specific tests:

```
pytest -q tests/test_battery_monitor -q
pytest -q tests/test_utils -q
```

## Configuration Details

Main keys (see `config/default_config.yaml` and `config/bms_rules.yaml`):

- `battery_monitor.ingestion.*` (CAN interface and CSV simulator)
- `battery_monitor.ml.enable`, `battery_monitor.ml.feature_window_s`, `battery_monitor.ml.model_path`, `battery_monitor.ml.anomaly_threshold`
- `battery_monitor.rules.enable`, `battery_monitor.rules.enable_soh_tracking`, `battery_monitor.rules.soh_history_window_samples`, `battery_monitor.rules.thresholds.*`, `battery_monitor.rules.severities.*`

### Power Bus Monitoring

- Pack-level monitoring via `VoltageCurrentHandler` (INA219 + ADS1115) provides `pack_voltage_v`, `pack_current_a`, `max_temp_c`, and computed `power_w`.
- The `RuleEngine` and ML feature extractor have been extended to incorporate these pack-level signals (pack voltage/current/temp, power statistics, voltage-drop and current-spike flags).

### Pack-level rule thresholds

- Pack-level thresholds (e.g. `pack_voltage_min_v`, `pack_voltage_max_v`, `pack_current_spike_a`, `pack_temp_max_c`, `voltage_drop_threshold_v`) are now configurable in `config/bms_rules.yaml` under the `thresholds` mapping. These values are read by the `RuleEngine` and used to decide pack-level alerts. Update `config/bms_rules.yaml` or your deployment config to tune behavior for your hardware.

## Notes

- The `RuleEngine` is thread-safe for SoH history via a `Lock` and deque.
- `ConfigLoader` merges values from `config/bms_rules.yaml` at startup so rule thresholds are available via `battery_monitor.rules.thresholds`.
- Extend rules by adding new check methods in `rule_engine.py` and adding calls in `evaluate()`.
