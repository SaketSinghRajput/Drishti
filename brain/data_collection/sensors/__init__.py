"""Sensor handlers package.

Exports:
- SensorHandler (base class)
- GPSHandler, IMUHandler, VehicleControlsHandler, DriverFatigueHandler

Handlers are designed to be thread-safe and return dicts suitable for WindowManager.
"""

from .base import SensorHandler
from .gps_handler import GPSHandler
from .imu_handler import IMUHandler
from .vehicle_controls_handler import VehicleControlsHandler
from .driver_fatigue_handler import DriverFatigueHandler
from .ble_mesh_handler import BLEMeshHandler
from .camera_handler import CameraHandler
from .weather_handler import WeatherHandler
from .road_handler import RoadHandler
from .bms_handler import BMSHandler
from .vehicle_can_handler import VehicleCANHandler
from .gsm_handler import GSMHandler

__all__ = [
    "SensorHandler",
    "GPSHandler",
    "IMUHandler",
    "VehicleControlsHandler",
    "DriverFatigueHandler",
    "BLEMeshHandler",
    "CameraHandler",
    "WeatherHandler",
    "RoadHandler",
    "BMSHandler",
    "VehicleCANHandler",
    "GSMHandler",
]
