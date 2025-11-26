# MediaPipe TFLite Integration

This document describes the MediaPipe TFLite task integration used by the driver fatigue
and camera pipelines. The code supports a 3-stage face pipeline for the driver camera:

1. Face detection (`face_detection_short_range.tflite`)
2. Face landmarking (`face_landmark_lite.tflite`)
3. Iris landmarking (`iris_landmark.tflite`)

And a pose pipeline for external cameras:

1. Person detection (`pose_detection.tflite`)
2. Pose landmarking (`pose_landmark_full.tflite`)

Configuration
-------------
The YAML config (`config/default_config.yaml`) exposes a `mediapipe` section and
sensor flags under `sensors.driver_fatigue` and `sensors.camera` to enable/disable
TFLite tasks and provide model paths.

Files
-----
- `brain/utils/mediapipe_loaders.py`: Minimal TFLite wrappers for face/landmark/iris/pose.
- `data_collection/sensors/driver_drowsiness/tflite_detector.py`: Uses mediapipe loaders when `use_mediapipe_tasks` is true.
- `data_collection/sensors/camera_handler.py`: Can enrich person detections with `pose_data` when `enable_pose_detection` is true.

Troubleshooting
---------------
- If models are not found, the loaders return safe defaults and the pipeline falls back to heuristics.
- If TFLite interpreter is not available, the code falls back to the Python MediaPipe API (if installed), otherwise Haar/heuristics.

