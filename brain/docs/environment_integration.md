# Environment Integration

This document describes the environment subsystem: weather and road handlers and the EnvironmentAdapter.

## Overview

The environment subsystem provides two sensor handlers (`WeatherHandler`, `RoadHandler`) that emit simulated samples for weather and road geometry, and an `EnvironmentAdapter` which aggregates samples into window-level features such as weather/road risk scores and time-of-day categorization.

(See `examples/environment_example.py` for a runnable demonstration.)

## Configuration

Relevant config lives under `sensors.weather`, `sensors.road`, and `environment_features` in `config/default_config.yaml`.

### New/important keys

sensors.weather (defaults shown):
- `sample_rate`: 0.1
- `enable_simulation`: true
- `default_visibility_m`: 5000
- `simulation_transition_probability`: 0.05
- `rain_visibility_reduction`: 0.5
- `snow_visibility_reduction`: 0.3

sensors.road (defaults shown):
- `sample_rate`: 1
- `enable_simulation`: true
- `default_curvature_radius_m`: 1000.0
- `default_speed_limit_kph`: 50
- `simulation_segment_duration_s`: 30
- `intersection_probability`: 0.1
- `sharp_turn_curvature_threshold_m`: 50.0
- `sharp_turn_probability`: 0.5

environment_features (weights & thresholds):
- `weather_risk_weights`: mapping for precip/visibility/surface (example in default config)
- `road_risk_weights`: mapping for curvature/turn/intersection/slope
- `steep_slope_threshold_percent`: used to determine slope risk

## Notes

- `enable_lane_detection` remains a placeholder in `CameraHandler` (object-derived lane offset only).
- `CameraHandler` enforces `max_retained_frames_per_trip` before passing raw frames to the retention hook.

