from utils.validation import validate_sensor_reading, compute_data_quality_score


def test_validate_gps_reading():
    ok, errs = validate_sensor_reading("gps", {"lat": 0.0, "lon": 0.0})
    assert ok
    bad, errs = validate_sensor_reading("gps", {"lat": 999, "lon": "x"})
    assert not bad


def test_compute_data_quality_score():
    w = {"gps": {}, "imu": {}}
    score = compute_data_quality_score(w, ["gps", "imu", "vehicle_controls"])
    assert 0 <= score <= 1
