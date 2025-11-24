from data_collection.sensors.driver_drowsiness import utils


def test_adjust_thresholds_for_ir_low_light():
    ear, mar = utils.adjust_thresholds_for_ir(0.25, 0.6, ambient_light_level=10.0, ir_ear_offset=0.05, ir_mar_offset=-0.05, ambient_light_low_threshold=30.0)
    assert ear >= 0.25
    assert mar <= 0.6


def test_adjust_thresholds_for_ir_bright_light():
    ear, mar = utils.adjust_thresholds_for_ir(0.25, 0.6, ambient_light_level=200.0, ir_ear_offset=0.05, ir_mar_offset=-0.05, ambient_light_low_threshold=30.0)
    # thresholds should be unchanged in bright ambient
    assert abs(ear - 0.25) < 1e-6
    assert abs(mar - 0.6) < 1e-6
