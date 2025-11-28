# Production Validation Runbook

This document outlines the minimal steps to validate a release candidate for
production. The validation focuses on three areas:

- Dependency and environment checks
- Static typing / linter checks (optional)
- Test suite execution with strict warnings-as-errors

Quick steps

- Install Python and required packages: `pytest`, `numpy`, and any runtime
  dependencies (TensorFlow, tflite-runtime) for full validation.

- Run the validation script (recommended):

```powershell
python -m brain.scripts.validate_production
```

What the script does

- Verifies a small set of required packages are importable.
- Attempts to run `mypy` if available.
- Runs `pytest` with `-W error::DeprecationWarning --strict-markers`.

Key configuration keys for Grip/MediaPipe

- `mediapipe.hand_landmark.enabled` (recommended canonical toggle)
- `mediapipe.hand_landmark.model_path` (path to the TFLite hand-landmark model)
- legacy override: `sensors.driver_fatigue.hand_landmark.enable` or
  `sensors.driver_fatigue.hand_landmark.enabled`

To enable grip monitoring in production ensure either the `mediapipe` tree or
`sensors.driver_fatigue.hand_landmark` is set to `enabled: true` (or `enable: true`)
and the model path points to a valid TFLite model. The `make_grip_monitor_from_config`
factory respects both locations to avoid accidental misconfiguration.

Tuning inference performance

- `sensors.driver_fatigue.input_resolution`: optional early downscale resolution
- `sensors.driver_fatigue.tflite_frame_skip`: skip N frames between inferences
- `sensors.driver_fatigue.inference_slow_threshold_s`: threshold to consider
  an inference slow (seconds)
- `sensors.driver_fatigue.inference_slow_limit`: consecutive slow count before
  using cached results

CI Recommendation

- Run `scripts/validate_production.py` as part of PR checks and fail on any
  non-zero exit codes. Ensure all tests marked with `@pytest.mark.production` are
  included in the strict subset.
