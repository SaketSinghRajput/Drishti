# Hand Pose & Enhanced Pose Integration

This document describes the hand landmark (grip) integration and enhanced pose
features (pedestrian intent and cyclist lean) introduced to the sensor
pipeline.

## Overview

- Hand landmark detection provides a steering-wheel `grip_confidence` value
  derived from the MediaPipe hand-landmark models (21 keypoints).
- Pedestrian intent scoring uses consecutive pose keypoints to compute a
  velocity-like delta; larger negative vertical deltas indicate a person
  approaching the camera (vehicle).
- Cyclist lean detection computes the shoulder->hip tilt angle; angles above
  15° indicate maneuvering/turning risk.

## Architecture

- The primary driver fatigue pipeline remains unchanged for face/iris EAR/MAR
  computation.
- A parallel `GripMonitor` thread can be started to run hand-landmark
  inference without blocking the main loop.
- Pose enrichment runs in the camera handler and augments person/bicycle
  objects with `pose_data` fields: `pedestrian_intent_score`,
  `cyclist_lean_angle` and `cyclist_maneuvering_flag`.

## Hand Landmark Detection

API: `MediaPipeHandLandmarker(model_path)`

- `detect_hand_landmarks(frame, bbox)` -> `List[(x,y)]` normalized to full frame
- Keypoint indices: 0=wrist, 4=thumb tip, 8=index tip, 12=middle tip, 16=ring tip, 20=pinky tip

Grip heuristic (default): wrist present and >=3 fingertips visible with
vertical clustering and horizontal spread -> high confidence.

## Pedestrian Intent Scoring

- Uses shoulder (indices 11,12) and hip (23,24) midpoints.
- Computes vertical delta across frames; maps delta through a sigmoid to
  produce score in 0..1.
- Thresholds: >0.7 = approaching, ~0.5 stationary, <0.3 receding.

## Cyclist Lean Detection

- Shoulder and hip midpoints define a vector; angle from vertical in degrees
  quantifies lean. >15° flags `cyclist_maneuvering_flag`.

## Configuration

Key toggles in `brain/config/default_config.yaml`:

- `mediapipe.hand_landmark.enabled` (default: false)
- `sensors.driver_fatigue.hand_landmark.enable` (default: false)
- `sensors.camera.enable_pedestrian_intent` (default: false)
- `sensors.camera.enable_cyclist_lean` (default: false)
- `feature_engineering.enable_grip_interactions` (default: false)
- `feature_engineering.enable_pedestrian_intent_interactions` (default: false)

## Testing

- Unit tests for the hand landmarker are provided under
  `brain/tests/test_utils/test_mediapipe_hand_loader.py` (mocked interpreter).
- Integration tests for grip features and pedestrian/cyclist features are
  included under `brain/tests/test_data_collection/` and
  `brain/tests/test_feature_engineering/`.

## Troubleshooting

- If the hand landmarker reports unavailable, ensure TFLite runtime is
  installed (`tflite-runtime` or `tensorflow`) and the model path points to a
  valid `.tflite` file.
- If parallel thread blocks, check that the platform's thread count or GIL
  interactions are not exhausted; reduce `mediapipe.tflite_num_threads` if
  necessary.

## Performance

- Lite hand landmark models typically run <20ms on modern ARM/Intel CPUs;
  full models are heavier (~30ms).
- Parallel hand inference reduces blocking on the main pipeline but adds
  background CPU usage. Use `mediapipe.hand_landmark.parallel_inference` to
  toggle this behavior.
