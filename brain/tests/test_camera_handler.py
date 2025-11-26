from data_collection.sensors.camera_handler import CameraHandler


def test_camera_simulation_read():
    cfg = {"sensors": {"camera": {"enable_simulation": True}}}
    ch = CameraHandler(cfg)
    data = ch.read()
    assert isinstance(data, dict)
    assert "objects" in data
    assert "timestamp" in data
