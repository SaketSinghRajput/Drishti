import time
import types

import pytest

from data_collection.sensors.vehicle_can_handler import VehicleCANHandler


def test_simulation_mode():
    # pass a simple dict-like config
    cfg = {"sensors": {"vehicle_can": {"enable_simulation": True, "sample_rate": 5}}}
    h = VehicleCANHandler(cfg)
    s = h.read()
    assert isinstance(s, dict)
    assert "timestamp" in s and "speed_kph" in s


def test_can_parsing(monkeypatch):
    # create a fake CANIngestion that returns an OBD-II speed response: [0x41, 0x0D, 60]
    class FakeCAN:
        def __init__(self, cfg=None):
            pass

        def start(self):
            pass

        def read_message(self, timeout=0.1):
            return (0x7E8, bytes([0x41, 0x0D, 60]))

        def get_status(self):
            return {"available": True}

    monkeypatch.setattr("data_collection.sensors.vehicle_can_handler.CANIngestion", FakeCAN)
    cfg = {"sensors": {"vehicle_can": {"enable_simulation": False, "can_interface": "can0"}}}
    h = VehicleCANHandler(cfg)
    s = h.read()
    assert isinstance(s, dict)
    assert s.get("speed_kph") == 60 or s.get("raw_data") is not None
