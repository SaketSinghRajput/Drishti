# Data Collection

This module implements a simulated, thread-safe data collection pipeline for core vehicle sensors including GPS, IMU, vehicle controls and driver fatigue monitoring. It provides a `DataLogger` that orchestrates background sensor polling, sliding window aggregation, and writing windowed JSON samples to disk with retention-based cleanup.

See `examples/basic_logger_example.py` for a simple usage example.

## Driver Fatigue Feature Enrichment

The `DriverFatigueAdapter` post-processes raw driver fatigue samples within each time window to extract higher-level features for accident prediction models. It computes:

### Rolling Statistics
- Mean, Std, Min, Max for `drowsiness_score`, `blink_rate`, and `perclos`.

### Sudden Change Detection
- Detects rapid increases in drowsiness (e.g., >0.3 increase in <3 seconds)
- Flags potential micro-sleep events or attention lapses
- Configurable thresholds via `sensors.driver_fatigue.sudden_drowsiness_threshold`

### Trend Analysis
- Determines if drowsiness/blink_rate is increasing, decreasing, or stable
- Uses linear regression slope to classify trends

### Event Counting
- Aggregates total yawns and micro-sleep events across the window

### Head Pose Aggregation
- Computes mean pitch, yaw, roll from head pose data

### Integration

The adapter is invoked by `DataLogger` during the window flush loop. Enriched features are added to each window under the `driver_fatigue_features` key before writing to disk. This ensures downstream feature engineering and model training have access to both raw samples and computed features.

### Configuration

See `config/default_config.yaml` under `sensors.driver_fatigue` for tunable parameters.

## Camera Sensor

The camera sensor captures visual scene information through object detection, providing data for collision avoidance and pedestrian protection. The system uses a simulation-first `CameraHandler` for development and supports TFLite/ONNX/YOLOv5 models for production.

Camera in expected sensors: the default configuration now includes `camera` in
`data_collection.expected_sensors`. If you prefer camera to be optional for your
deployment, remove it from that list in `config/default_config.yaml` — doing so
will make health checks treat the camera as optional.

### Data Collected
- Detected objects with bounding boxes, estimated distances and relative speeds
- Lane center offset and simple lane clearance flags
- Obstacle-in-path flags and pedestrian near-crossing flags

### Adapter
`CameraAdapter` computes window-level aggregates (distance stats, objects-in-path ratios, pedestrian risk, relative motion summaries) and stores results under `camera_features` in each window.

See `docs/camera_integration.md` for full integration details and configuration options.
