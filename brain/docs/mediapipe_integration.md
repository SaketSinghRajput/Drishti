# MediaPipe Integration Guide

This document summarizes the project's MediaPipe TFLite integration, configuration, and troubleshooting tips.

## Architecture

- TFLite models exported from MediaPipe are detected automatically by `brain.utils.model_loader_tflite.TFLiteDetectionModel`.
- When a model appears to be a MediaPipe task, the `MediaPipeTaskLoader` factory attempts to instantiate a task-specific loader from `brain.utils.mediapipe_loaders` (face, pose, iris, hand wrappers).
- Hand/pose landmark extraction and auxiliary enrichment (grip detection, pedestrian intent) are exposed via sensor handlers and adapters under `brain/data_collection`.

## Configuration

Example `config/default_config.yaml` snippets (relevant keys):

- Enable MediaPipe globally:

```yaml
mediapipe:
  enabled: true
  tflite_num_threads: 2
```

- Driver fatigue sensor toggles:

```yaml
sensors:
  driver_fatigue:
    use_mediapipe_tasks: true
    hand_landmark:
      enable: true
      grip_detection_threshold: 0.5
      enable_grip_features: true
```

- Camera sensor toggles:

```yaml
sensors:
  camera:
    enable_pose_detection: true
    enable_pedestrian_intent: true
    enable_cyclist_lean: true
```

## Performance & Benchmarking

- Use `scripts/benchmark_fatigue_latency.py --mediapipe-suite --tflite-model <path>` to run a basic per-frame timing loop using the `TFLiteDetectionModel` wrapper.
- For detailed comparisons between the Python MediaPipe API and TFLite task models, run `examples/mediapipe_benchmark_example.py`.
- Aim for per-stage inference latency < 50ms for real-time deployment; actual results depend on hardware and interpreter threading.

## Troubleshooting

- If the auto-detection does not classify your model correctly, check the model output shapes and tensor names. Exports vary; adjust `auto_detect_mediapipe_task()` heuristics in `brain/utils/model_loader_tflite.py` or add an explicit loader in `brain/utils/mediapipe_loaders.py`.
- Missing TensorFlow or `tflite_runtime`: the loader falls back to a synthetic or generic path. Install `tflite_runtime` for lightweight runtimes on embedded devices.
- If a MediaPipe loader fails during runtime, the `TFLiteDetectionModel` falls back to generic TFLite interpreter inference to preserve behavior.

## Development Tips

- Add new MediaPipe task wrappers in `brain/utils/mediapipe_loaders.py` and expose a minimal `detect()` or `predict()` API so `MediaPipeTaskLoader` can instantiate and delegate to it.
- Unit tests for loader auto-detection live under `brain/tests/test_utils/test_model_loader_tflite.py` (mocked interpreters allowed).

