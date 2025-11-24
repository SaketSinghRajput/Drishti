# Battery Health Monitoring Module

Real-time battery management system (BMS) monitoring for EV Safety Guardian.

## Features
- CAN bus ingestion from LibreSolar bms-c1
- Rule-based alarms (temp, voltage, current)
- ML anomaly detection and RUL prediction (stubs)
- Flask dashboard with live graphs
- BLE mesh broadcast for critical alerts
- SQLite event logging

## Structure
```
battery_monitor/
├── ingestion/       # CAN parsing and CSV simulation
├── rules/           # Rule-based alarm engine
├── ml/              # ML inference for anomaly/RUL
├── alerts/          # Alert manager and BLE broadcast
├── storage/         # SQLite event logger
└── dashboard/       # Flask web dashboard
```

## Quick Start
See `docs/battery_monitoring.md` for full documentation.

## External References
- https://github.com/LibreSolar/bms-c1
- https://github.com/kmcallister/rpi-battery-monitor

## Development Status
Phase 1: Directory structure, schema, config, dependencies

## Phase 2: CAN Ingestion and Data Parsing

The following components are provided in Phase 2:

- `battery_monitor/ingestion/can_ingestion.py` - CAN bus interface using `python-can` (scaffold)
- `battery_monitor/ingestion/bms_parser.py` - BMSCANParser for parsing LibreSolar-style CAN frames
- `battery_monitor/ingestion/csv_simulator.py` - enhanced CSV simulator with streaming reads
- `data_collection/sensors/bms_handler.py` - `BMSHandler` sensor integrating simulation and CAN ingestion

Data flow: CAN/CSV -> Parser -> Handler -> DataLogger -> WindowManager

See `docs/bms_integration.md` and `examples/bms_example.py` for usage and testing.
