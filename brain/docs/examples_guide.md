# Examples Guide

This guide documents the runnable examples under `brain/examples/`.

## Quick Start

Prerequisites:
- Activate the `.venv-win` virtual environment
- Ensure `PYTHONPATH` includes the repo root or run from `f:/iit`
- Create data directories:

```powershell
mkdir data
mkdir data/windows
mkdir data/telemetry
mkdir examples/output
```

Run a single example:

```powershell
python brain/examples/basic_logger_example.py
```

## Example Categories

- Data Collection: `basic_logger_example.py`, `ble_mesh_example.py`, `environment_example.py`
- Sensors / CV: `camera_example.py`, `driver_fatigue_cv_example.py`, `hand_pose_example.py`
- Training: `xgboost_training_example.py`, `lstm_training_example.py`, `cnn_training_example.py`, `multimodal_training_example.py`
- Inference: `edge_inference_example.py`, `explainability_example.py`, `llm_integration_example.py`
- Dashboards: `dashboard_example.py`, `battery_dashboard_example.py`
- Full System: `orchestrator_example.py`, `run_fatigue_cv_demo.py`

## Common Issues & Fixes

- Missing models: run `scripts/verify_mediapipe_models.py` or provide model files in the expected paths.
- Long latency: update `config/default_config.yaml` to lower resolutions and sample rates for edge.
- Schema validation errors: see `config/schema/sensor_window_schema.json` for the canonical schema.
- Webcam access on Windows: grant camera permissions to your terminal and use `--camera 0`.

## Running All Examples

To run all examples sequentially (recommended in CI), use the runner script:

```powershell
python brain/scripts/run_all_examples.py --timeout 60
```

Use `--stop-on-error` to halt on the first failing example and inspect logs.

## Per-Example Notes

Each example may require small setup (data directories, model files). See the header comments in each file for required args and flags.
