# System Architecture

Overview of the multimodal accident prediction system. Data flows from sensors into the data collection layer, then into feature engineering and model inference. Telemetry, logging, and privacy-preserving storage complete the pipeline.

Components:

- data_collection: Sensor adapters for GPS, IMU, camera, BLE mesh, and driver-facing camera
- feature_engineering: Windowed aggregation, derived features (jerk, steering rate), interaction features
- models: Training pipelines for XGBoost baseline and future deep learning models
- inference: Edge inference engine, model loaders (ONNX/TFLite), explainability
- utils: Helpers for logging, encryption, validation

Design notes:
- Latency target: <200ms
- Edge-first and privacy-preserving defaults
- Feature-level fusion to keep latency low

Battery monitoring integration:

- `battery_monitor`: BMS telemetry ingestion, rule-based alarms, ML anomaly detection, EventLogger for snapshots, and a Flask dashboard for live visualization and REST APIs.
- Data flow: BMS telemetry → BMSMonitor (rules + ML) → EventLogger (SQLite) → Dashboard API / BLE outbox → EdgeInferenceEngine (feature fusion or risk adjustment).
- Integration patterns: feature-level fusion, shared SystemState, or risk score adjustment (see `docs/battery_schemas.md` and `docs/battery_dashboard.md`).
