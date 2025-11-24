# Data Schema

This document describes the sensor window schema used throughout the project. The authoritative JSON Schema files are located in `config/schema/`.

Key fields:
- `timestamp` (ISO 8601)
- `vehicle_hash` (anonymized ID)
- `gps`, `imu`, `vehicle_controls`, `driver_fatigue` (required)
- Optional fields: `ble_neighbors`, `camera`, `weather`, `road`, `label`, `metadata`

Example minimal window (see `config/schema/sensor_window_schema.json` for full details):

```
{
  "timestamp": "2025-01-01T12:00:00Z",
  "vehicle_hash": "vh_example",
  "gps": {"lat": 37.0, "lon": -122.0, "speed_kph": 50.0, "heading": 180},
  "imu": {"yaw_rate": 0.1, "pitch": 0.0, "roll": 0.0, "ax": 0.0, "ay": 0.0, "az": 9.8},
  "vehicle_controls": {"steering_angle": 0.0, "brake_pressure": 0.0, "brake_flag": false, "throttle_position": 0.2},
  "driver_fatigue": {"drowsiness_score": 0.1, "blink_rate": 10, "perclos": 0.1, "head_pose": {"pitch": 0, "yaw": 0, "roll": 0, "confidence": 1.0}, "yawning_count": 0, "micro_sleep_flag": false}
}
```
