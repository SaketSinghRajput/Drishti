# Fake Datasets for Accident Prediction

This folder contains tools and a small sample dataset for experimenting with accident prediction models.

Files:
- `generate_fake_accidents.py`: Script to generate synthetic telemetry rows with an `accident` label (0/1).
- `sample_accident_dataset.csv`: A small CSV sample with example rows.

Suggested fields (columns):
- `timestamp`: ISO 8601 timestamp
- `vehicle_speed_kph`: vehicle speed in km/h
- `steering_angle_deg`: steering wheel angle in degrees
- `brake_pressure`: 0-1 normalized brake pressure
- `accel_x`, `accel_y`, `accel_z`: accelerometer readings (m/s^2)
- `lane_offset_m`: lateral offset from lane center in meters
- `weather`: one of `clear`, `rain`, `fog`, `snow`
- `light_condition`: one of `day`, `night`, `dawn`, `dusk`
- `driver_drowsiness`: 0-1 score (higher -> more drowsy)
- `distraction_level`: 0-1 score
- `nearby_vehicles`: integer count
- `road_type`: one of `highway`, `urban`, `rural`
- `accident`: label (0 = no accident, 1 = accident)

Usage

Generate a CSV with 10k rows:

```powershell
python generate_fake_accidents.py --n 10000 --out my_accidents.csv
```

The generator uses simple heuristics to increase accident probability when speed is high, drowsiness is high, braking is extreme, or weather is bad. Use the generated data as a starting point for model experiments; it is synthetic and not suitable for deployment without real data validation.
