# Feature Engineering Guide

This document describes the `FeatureExtractor` which converts sensor windows
into ML-ready feature vectors.

## Overview

Feature engineering consolidates raw sensor aggregates, derived temporal
features, interaction features, and adapter-provided metrics into a single
flat feature vector suitable for training and inference.

## Usage

```python
from config.config_loader import ConfigLoader
from feature_engineering import FeatureExtractor

cfg = ConfigLoader()
fe = FeatureExtractor(cfg)
features = fe.extract(window)
df = fe.extract_to_dataframe(windows)
```

## Feature Categories

### Raw Sensor Aggregations
- GPS: `speed_mean`, `speed_std`, `speed_min`, `speed_max`, `speed_kph_delta_*`
- IMU: `ax_mean`, `ay_mean`, `az_mean`, `yaw_rate_mean`, `total_acceleration_*`
- Vehicle controls: `steer_mean`, `brake_mean`, `throttle_mean`, `brake_flag_ratio`

### Derived Features
- Jerk (axis-wise and magnitude): `jerk_magnitude_mean`, `jerk_magnitude_max`
- Steering rate: `steering_rate_mean`
- Speed deltas: `speed_kph_delta_total`, `speed_kph_delta_mean`, `speed_kph_delta_max`
- Lane offset trend slope: `lane_offset_trend_slope`

### Interaction Features
- `drowsiness_speed_interaction`, `visibility_speed_interaction`, `curvature_speed_interaction`, `wet_road_speed_interaction`, `brake_drowsiness_interaction`, `nearest_object_speed_interaction`

### Adapter Features
Adapter features from `driver_fatigue_features`, `camera_features`, `environment_features` are flattened and included. Prefixes can be enabled in config to avoid name collisions.

## Configuration

See `config/default_config.yaml` under `feature_engineering` for toggles and missing-data handling.

## Missing Data Handling

- Sentinel values (`-999.0` by default) are used for empty sensor arrays.
- Availability flags (`gps_available`, `imu_available`, `vc_available`) are emitted.

## Performance

- Batch extraction via `extract_to_dataframe` uses pandas when available.
- Most computations use pure Python fallbacks if `numpy`/`pandas` are not installed.

## Examples

See `examples/feature_extraction_example.py` for a minimal example and
`scripts/extract_features_batch.py` for a CLI batch extraction tool.
