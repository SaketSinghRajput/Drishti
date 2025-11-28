# Multimodal Accident Prediction System

Real-time accident risk prediction using driver fatigue, BLE mesh, camera, weather, and road data. This repository contains the project skeleton and configuration for an edge-first, privacy-preserving prediction system.

Key features:
- Edge deployment targets (Raspberry Pi, Jetson Nano)
- Explainable AI (SHAP) for model transparency
- Privacy defaults (data anonymization, encryption at rest)
 - Real-time object detection (MobileNet-SSD, YOLOv5, ONNX) with metadata-only storage
 - Boot-time hardware verification via systemd (Phase 4)
- Runtime explainability with SHAP: human-readable alert messages showing top risk factors
- Optional LLM (Gemini) augmentation for edge-case explanations (configurable, gated, rate-limited)
- Two-stage alert system: visual+audio warnings escalating to haptic feedback for persistent risks
- Explanation logging for debugging and continuous model improvement
- Weather and road condition monitoring with risk scoring

## Labeling and Dataset Preparation

The repository includes a labeling subsystem to produce accident and near-miss labels from stored window files. Use `scripts/label_windows.py` to run automated labeling, manual review, balancing (SMOTE), and temporal train/test splitting. See `docs/labeling_guide.md` for full details.

Project structure:

- `data_collection/`: Sensor ingestion and logging
- `feature_engineering/`: Feature extraction and transformation
- `models/`: ML training and evaluation pipelines
- `inference/`: Real-time edge inference engine
- `utils/`: Shared utilities and helpers
- `config/`: YAML configs and JSON schemas
- `tests/`: Unit and integration tests
- `scripts/`: Utility scripts (validation, sample data)

### Feature Engineering

The feature engineering pipeline transforms raw sensor windows into ML-ready feature vectors.

```python
from feature_engineering.extractor import FeatureExtractor

config = ConfigLoader()
extractor = FeatureExtractor(config)

# Extract features from a window
features = extractor.extract(window)

# Batch process multiple windows to DataFrame
df = extractor.extract_to_dataframe(windows)
df.to_csv('features.csv', index=False)
```

## Dashboard

The **EV Safety Guardian Dashboard** provides a unified web interface for real-time monitoring, incident review, annotation, and analytics.

### Features
- **Real-time Monitoring**: Live risk scores, sensor health, alert history, and SHAP explanations
- **Incident Review**: Video playback with synchronized sensor timelines for flagged high-risk events
- **Human Annotation**: Label incidents as TP/FP/TN/FN with notes for model improvement
- **Analytics**: False alarm rates, detection latency, feature importance trends, sensor reliability

### Quick Start
```bash
# Start the dashboard
python scripts/run_dashboard.py

# Access in browser
open http://localhost:5004
```

### Configuration
Add dashboard settings to `config/default_config.yaml` (the file now contains a `dashboard:` section with defaults).

See `docs/dashboard.md` for full documentation.

## Examples

The `brain/examples/` directory contains runnable scripts demonstrating system features, grouped by function. See `brain/docs/examples_guide.md` for a full guide.

Run all examples (sequential):

```powershell
python brain/scripts/run_all_examples.py --timeout 60
```

The examples runner writes `data/example_test_results.json` with a pass/fail summary.

Feature categories produced by the extractor include:
- Raw sensor aggregations (GPS speed/heading, IMU accelerations, vehicle controls)
- Derived features (jerk, steering rate, speed deltas, lane offset trends)
- Interaction features (drowsiness×speed, visibility×speed, curvature×speed, wet_road×speed)
- Adapter features (driver_fatigue, BLE mesh, camera, environment)

Installation:

1. Create and activate a virtual environment (Python 3.9+):

   python -m venv venv
   .\venv\Scripts\Activate.ps1  # PowerShell on Windows

2. Install dependencies:

   pip install -r requirements.txt

Optional camera dependencies (for production or local testing with real cameras):

   pip install opencv-python Pillow ultralytics

Configuration:

- Copy and customize `config/default_config.yaml`. Environment-specific overrides can be added later.

Quick start (placeholders for future implementation):

- Data collection: scripts to start sensor logging (to be implemented)
See `examples/ble_mesh_example.py` for a BLE mesh neighbor tracking demonstration.
See `examples/camera_example.py` for a camera simulation and adapter demonstration.
See `examples/environment_example.py` for weather and road data collection demonstration.
 
Running the Full System
-----------------------

The root orchestrator `main.py` coordinates hardware checks, sensors, BMS monitoring,
inference, dashboard, BLE safety broadcasts, alerts, and telemetry. For development:

```bash
python scripts/run_orchestrator.py --mode dev --config config_examples/dev_sim.yaml
```

On a production device (Raspberry Pi) the orchestrator is managed by systemd via
the `ev-safety-orchestrator.service` unit (see `docs/orchestrator.md`).
See `docs/llm_integration.md` for details on enabling and configuring the Gemini-based LLM augmentation.

## Driver Fatigue Integration

This project includes an integrated Driver Fatigue subsystem that exposes:

- BLE mesh fatigue broadcasting (file outbox)
- A REST API for querying latest fatigue status (`/status`, `/events`, `/query`)
- Persistent SQLite logging of per-sample fatigue events
- Optional central event push for fleet monitoring

See `docs/driver_fatigue_integration.md` for full usage and configuration.

Quick start:

```powershell
# Run the fatigue API server
python scripts/run_fatigue_api.py

# Run the integration example to exercise detect->alert->broadcast->log->api flows
python examples/fatigue_integration_example.py
```
- Training: training pipeline scripts will be added under `models/`
- Inference: edge engine will be added under `inference/`
 - Edge Inference: real-time prediction engine for low-latency deployments (see `docs/edge_inference.md`)

## Model Training

The repository includes an XGBoost-based training pipeline for accident risk prediction.

Quick start examples:

```bash
# Train a model for 10-second prediction horizon
python scripts/train_model.py --data-dir data/labeled --horizon 10 --use-smote

# Evaluate a trained model
python scripts/evaluate_model.py --model-path models/trained/h10/model.json --data-dir data/test
```

Key features:
- Multi-horizon training (1s, 5s, 10s)
- Class imbalance handling: sample weights, SMOTE (if supported), focal loss
- GroupKFold cross-validation by `trip_id` to prevent leakage
- PR-AUC primary metric, calibration and Brier score reporting
- SHAP explainability and ONNX export for edge deployment

See `docs/xgboost_training.md` for full details and examples.

## Battery Health Monitoring

The project includes a Battery Health Monitoring subsystem with the following capabilities:

- Real-time telemetry ingestion (pack voltage, current, SoC, SoH, cell voltages and temperatures)
- Rule-based alarms and ML anomaly detection (RUL predictions are supported by the ML stub)
- Event snapshot logging to SQLite for persistent storage and analysis
- BLE outbox (development stub) to broadcast critical alerts
- Flask Dashboard with live graphs and REST APIs for visualization and programmatic access

The dashboard provides a lightweight, edge-compatible UI and APIs. See `docs/battery_dashboard.md` for the dashboard API reference and `docs/battery_schemas.md` for canonical schemas.

Quick start:
```powershell
# Run dashboard (simulation mode)
python -m battery_monitor.dashboard.app
# Access the dashboard at http://localhost:5002
```

Note: Production deployments reading CAN require appropriate CAN hardware (PiCAN, USB-CAN). Development can use CSV simulation.

## Telemetry & Privacy

The project includes a telemetry subsystem for production monitoring and retraining. Key points:

- Buffered anonymized JSONL logs of predictions, outcomes and false alarms are written to `data/telemetry`.
- IDs are hashed and locations can be coarsened by grid to protect privacy.
- Optional AES-256-GCM encryption at rest is supported (env var `TELEMETRY_ENCRYPTION_KEY`).
- Cloud sync aggregates summaries and uploads them via `scripts/sync_telemetry.py`.
- Event-triggered video frames are retained via `utils.video_retention.VideoRetentionManager` with retention and optional encryption.

See `docs/telemetry.md` and `examples/telemetry_example.py` for details and examples.

## Deep Learning Models

This project also provides deep-learning trainers for sequence and multimodal models:

- `models/train_lstm.py`: LSTM/GRU sequence trainer
- `models/train_cnn.py`: 1D-CNN temporal trainer
- `models/train_multimodal.py`: camera + sensor + tabular fusion trainer

Use `scripts/train_dl_model.py` to train and export models (TFLite), and `scripts/compare_models.py` to compare models. See `docs/deep_learning_training.md` for details.

See `config/schema/sensor_window_schema.json` and `config/schema/label_schema.json` for data format details.

Development:

- Code style: use Black and Flake8
- Type hints: use MyPy
- Tests: run `pytest`

License: TBD
