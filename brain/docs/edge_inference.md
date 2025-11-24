# Edge Inference System

The edge inference system enables real-time accident risk prediction on vehicle-grade edge devices with <200ms latency. It supports multiple model formats (XGBoost, TensorFlow Lite, ONNX) and provides configurable alerting with hysteresis to prevent false alarms.

## Architecture

### Components
1. **Model Loaders** (`inference/model_loaders.py`): Unified interfaces for loading XGBoost, TFLite, and ONNX models with lazy imports and runtime selection.
2. **Sliding Window Buffer** (`inference/sliding_window_buffer.py`): Lightweight, thread-safe buffer maintaining the last W seconds of sensor data for real-time feature extraction.
3. **Decision Unit** (`inference/decision_unit.py`): Converts risk scores to actionable alerts with hysteresis and context checks.
4. **Edge Inference Engine** (`inference/edge_inference.py`): Main orchestrator with multi-threaded sensor ingestion, feature extraction, model inference, and monitoring.

## Usage

See `examples/edge_inference_example.py` and `scripts/run_inference.py` for runnable examples.

## Configuration

Key parameters in `config/default_config.yaml`:

```yaml
inference:
  latency_budget_ms: 200
  inference_interval_s: 1.0
  model_path: 'models/latest_model.xgb'
  prediction_horizon_s: 10
  alert_hysteresis_margin: 0.05
  alert_hysteresis_dwell_s: 2.0
  critical_sensors: [gps, imu, vehicle_controls]
```

## Model Formats

- XGBoost (.xgb)
- TensorFlow Lite (.tflite)
- ONNX (.onnx)

## Performance Optimization

- Use XGBoost for lowest latency
- Use int8 quantized TFLite models when available
- Reduce enabled features in `feature_engineering` if extraction is slow

## API Reference

See inline docstrings in the modules under `inference/`.
