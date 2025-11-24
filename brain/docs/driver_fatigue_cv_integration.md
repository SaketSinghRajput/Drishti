# Driver Fatigue CV Integration

This document describes the CV-based driver fatigue detection integration.

Overview
- Uses MediaPipe Face Mesh by default for landmark extraction and computes
  EAR (eye aspect ratio), MAR (mouth aspect ratio), head pose estimates,
  blink and yawn counts, PERCLOS, and a combined drowsiness score.

Installation
- Install dependencies: `pip install -r requirements.txt` (ensure `mediapipe` and `imutils` are installed).
- Optionally clone external research repo into `data_collection/sensors/driver_drowsiness/external_repo`.

Configuration
- See `config/default_config.yaml` under `sensors.driver_fatigue` for parameters.

Algorithms
- EAR: standard 6-point formula. Lower EAR indicates eye closure.
- MAR: ratio of mouth opening to width; used for yawn detection.
- Head pose: estimated via landmark geometry; large tilt contributes to drowsiness score.

Usage
- Run example: `python examples/driver_fatigue_cv_example.py --video sample.mp4`

Testing
- Unit tests mock MediaPipe and camera I/O. See `tests/test_data_collection/test_driver_fatigue_cv.py`.
