import json
import pytest
try:
    from jsonschema import validate, ValidationError
except Exception:
    pytest.skip("jsonschema not installed, skipping schema validation tests", allow_module_level=True)


def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_valid_sensor_window(tmp_path, sample_sensor_window):
    schema = load_schema("config/schema/sensor_window_schema.json")
    # should not raise
    validate(instance=sample_sensor_window, schema=schema)


def test_missing_required_fields(tmp_path, sample_sensor_window):
    schema = load_schema("config/schema/sensor_window_schema.json")
    bad = dict(sample_sensor_window)
    bad.pop("timestamp", None)
    with pytest.raises(ValidationError):
        validate(instance=bad, schema=schema)


def test_label_schema_valid(tmp_path):
    schema = load_schema("config/schema/label_schema.json")
    good = {
        "window_id": "w_1",
        "timestamp": "2020-01-01T00:00:00Z",
        "annotation_timestamp": "2020-01-01T00:00:01Z",
        "is_accident": False,
    }
    validate(instance=good, schema=schema)


def test_label_schema_missing_required():
    schema = load_schema("config/schema/label_schema.json")
    bad = {"window_id": "w1"}
    with pytest.raises(ValidationError):
        validate(instance=bad, schema=schema)


def test_sensor_window_out_of_range(tmp_path, sample_sensor_window):
    schema = load_schema("config/schema/sensor_window_schema.json")
    bad = dict(sample_sensor_window)
    # steering out of allowed range
    bad["vehicle_controls"]["steering_angle"] = 2000
    with pytest.raises(ValidationError):
        validate(instance=bad, schema=schema)


def test_sensor_window_type_errors(tmp_path, sample_sensor_window):
    schema = load_schema("config/schema/sensor_window_schema.json")
    bad = dict(sample_sensor_window)
    # lat should be number
    bad["gps"]["lat"] = "not-a-number"
    with pytest.raises(ValidationError):
        validate(instance=bad, schema=schema)


def test_large_sensor_window_performance(tmp_path):
    schema = load_schema("config/schema/sensor_window_schema.json")
    # build a large window with many BLE neighbors and camera objects
    large = {
        "timestamp": "2025-01-01T00:00:00Z",
        "vehicle_hash": "vh_big",
        "gps": {"lat": 0.0, "lon": 0.0, "speed_kph": 10.0, "heading": 0},
        "imu": {"yaw_rate": 0.0, "pitch": 0.0, "roll": 0.0, "ax": 0.0, "ay": 0.0, "az": 9.8},
        "vehicle_controls": {"steering_angle": 0.0, "brake_pressure": 0.0, "brake_flag": False, "throttle_position": 0.0},
        "driver_fatigue": {"drowsiness_score": 0.0, "blink_rate": 0, "perclos": 0.0, "head_pose": {"pitch": 0, "yaw": 0, "roll": 0, "confidence": 1.0}, "yawning_count": 0, "micro_sleep_flag": False},
        "ble_neighbors": {"mesh_density": 100, "neighbors": []},
        "camera": {"objects": []},
    }
    for i in range(100):
        large["ble_neighbors"]["neighbors"].append({"id_hash": f"n{i}", "rssi": -50, "dist_m": 5.0, "rel_heading": 0, "rel_speed_mps": 0, "last_seen_ms": 0})
        large["camera"]["objects"].append({"class": "car", "bbox_pixels": [0, 0, 10, 10], "distance_m": 5.0, "rel_speed_mps": 0, "heading": 0})

    # should validate without raising
    validate(instance=large, schema=schema)
