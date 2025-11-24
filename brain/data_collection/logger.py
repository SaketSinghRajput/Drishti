"""DataLogger orchestrates sensor handlers, window management, and storage."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Optional

from config.config_loader import ConfigLoader
from data_collection.buffer import CircularBuffer
from data_collection.window_manager import WindowManager
from data_collection.sensors import (
    GPSHandler,
    IMUHandler,
    VehicleControlsHandler,
    DriverFatigueHandler,
    BLEMeshHandler,
)
from data_collection.sensors import CameraHandler
from data_collection.sensors import WeatherHandler, RoadHandler
from data_collection.sensors import BMSHandler
from data_collection.sensors.vehicle_can_handler import VehicleCANHandler
from data_collection.sensors.voltage_current_handler import VoltageCurrentHandler
from data_collection.environment_adapter import EnvironmentAdapter
from data_collection.driver_fatigue_adapter import DriverFatigueAdapter
from data_collection.ble_mesh_adapter import BLEMeshAdapter
from data_collection.camera_adapter import CameraAdapter
from utils.video_retention import VideoRetentionManager

logger = logging.getLogger(__name__)


class DataLogger:
    def __init__(self, config: ConfigLoader, vehicle_hash: Optional[str] = None):
        self.config = config
        self.vehicle_hash = vehicle_hash or uuid.uuid4().hex
        self.window_manager = WindowManager(config)
        self.buffer = CircularBuffer(config)
        # driver fatigue feature adapter
        self.fatigue_adapter = DriverFatigueAdapter(config)
        self.ble_adapter = BLEMeshAdapter(config)
        self.camera_adapter = CameraAdapter(config)
        self.environment_adapter = EnvironmentAdapter(config)
        # instantiate handlers
        self.sensors = [
            GPSHandler(config),
            IMUHandler(config),
            VehicleControlsHandler(config),
            DriverFatigueHandler(config),
            BLEMeshHandler(config),
            CameraHandler(config),
            WeatherHandler(config),
            RoadHandler(config),
        ]
        # Optionally add Vehicle CAN handler if enabled in config
        try:
            if bool(self.config.get("sensors.vehicle_can.enable", False)):
                self.sensors.append(VehicleCANHandler(config))
        except Exception:
            logger.exception("Failed to initialize VehicleCANHandler")
        # Try to add BMS handler but do not fail startup if unavailable
        try:
            bms = BMSHandler(config)
            self.sensors.append(bms)
        except Exception:
            logger.warning('BMS handler unavailable; continuing without BMS')
        # Optionally add VoltageCurrentHandler for power bus monitoring
        try:
            if bool(self.config.get('sensors.voltage_current.enable', False)):
                try:
                    vch = VoltageCurrentHandler(config)
                    self.sensors.append(vch)
                    logger.info('VoltageCurrentHandler initialized')
                except Exception as e:
                    logger.warning('VoltageCurrentHandler init failed: %s', e)
        except Exception:
            pass
        # If we created both BMS and VoltageCurrentHandler, wire the VCH into BMSHandler
        try:
            bms_handler = next((s for s in self.sensors if getattr(s, 'get_name', lambda: None)() == 'bms'), None)
            vch_handler = next((s for s in self.sensors if getattr(s, 'get_name', lambda: None)() == 'voltage_current'), None)
            if bms_handler and vch_handler:
                # attach as a best-effort reference; BMSHandler will call vch.read() when normalizing CAN frames
                try:
                    setattr(bms_handler, '_vch', vch_handler)
                except Exception:
                    pass
        except Exception:
            pass
        # persistent trip id for this logging session
        self.trip_id = uuid.uuid4().hex
        # health snapshot protected by a lock
        self._health_snapshot = {}
        self._health_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads = []
        # Video retention manager
        try:
            if bool(self.config.get('sensors.camera.enable_frame_retention', True)):
                self.video_retention_manager = VideoRetentionManager(config)
                try:
                    self.video_retention_manager.start_cleanup()
                except Exception:
                    pass
            else:
                self.video_retention_manager = None
        except Exception:
            self.video_retention_manager = None
        # attach camera flagged hook
        try:
            for s in self.sensors:
                if isinstance(s, CameraHandler):
                    # set the on_frame_flagged hook
                    s.on_frame_flagged = self._handle_flagged_frame
        except Exception:
            pass

    def start(self) -> None:
        self._stop_event.clear()
        # start sensor threads
        for s in self.sensors:
            t = threading.Thread(target=self._sensor_loop, args=(s,), daemon=True)
            t.start()
            self._threads.append(t)

        # start window flush loop
        t = threading.Thread(target=self._window_flush_loop, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1.0)
        # stop video retention cleanup thread
        try:
            if hasattr(self, 'video_retention_manager') and self.video_retention_manager:
                self.video_retention_manager.stop_cleanup()
        except Exception:
            logger.exception("Failed to stop video retention manager")

    def _handle_flagged_frame(self, metadata: dict, frame: Optional[Any] = None) -> None:
        try:
            event_type = 'high_risk' if float(metadata.get('risk_score', 0) or 0) > 0.6 else 'triggered'
            trip_id = getattr(self, 'trip_id', 'unknown')
            if hasattr(self, 'video_retention_manager') and self.video_retention_manager:
                if frame is not None:
                    self.video_retention_manager.save_frame(frame, metadata, event_type, trip_id)
                else:
                    try:
                        self.video_retention_manager.save_metadata_only(metadata, event_type, trip_id)
                    except Exception:
                        # fallback to save metadata file in output dir
                        pass
        except Exception:
            logger.exception("Failed handling flagged frame")

    def _sensor_loop(self, handler) -> None:
        rate = getattr(handler, "sample_rate", 1)
        interval = 1.0 / max(1.0, float(rate))
        while not self._stop_event.is_set():
            data = handler.read()
            if data:
                self.window_manager.add_reading(handler.get_name(), data)
                # update health snapshot for this sensor
                try:
                    with self._health_lock:
                        self._health_snapshot[handler.get_name()] = handler.get_health()
                except Exception:
                    # non-fatal; continue
                    pass
            time.sleep(interval)

    def _window_flush_loop(self) -> None:
        interval = float(self.config.get("data_collection.window_interval_s", 1))
        while not self._stop_event.is_set():
            try:
                if self.window_manager.should_flush():
                    # snapshot health
                    with self._health_lock:
                        health_snapshot = dict(self._health_snapshot)
                    window = self.window_manager.get_current_window(sensor_health_snapshot=health_snapshot)
                    # enrich with identifiers
                    window["vehicle_hash"] = self.vehicle_hash
                    window["trip_id"] = self.trip_id

                    # ensure expected sensors are present in the window (avoid schema validation failures)
                    try:
                        for s in getattr(self.window_manager, 'expected_sensors', []):
                            window.setdefault(s, [])
                    except Exception:
                        pass

                    # Enrich with driver fatigue features if samples exist
                    try:
                        df_samples = window.get("driver_fatigue")
                        if isinstance(df_samples, list) and df_samples:
                            # attempt to retrieve events from the handler if available
                            fatigue_handler = next((s for s in self.sensors if s.get_name() == 'driver_fatigue'), None)
                            events = None
                            try:
                                if fatigue_handler and hasattr(fatigue_handler, 'get_events'):
                                    events = fatigue_handler.get_events()
                            except Exception:
                                events = None
                            fatigue_features = self.fatigue_adapter.analyze(df_samples, events=events)
                            if fatigue_features:
                                window["driver_fatigue_features"] = fatigue_features
                    except Exception as e:
                        logger.warning("Failed to compute driver fatigue features: %s", e)

                    # Enrich with BLE mesh features if samples exist
                    try:
                        ble_samples = window.get("ble_mesh")
                        if isinstance(ble_samples, list) and ble_samples:
                            ble_features = self.ble_adapter.analyze(ble_samples)
                            if ble_features:
                                window["ble_mesh_features"] = ble_features
                    except Exception as e:
                        logger.warning("Failed to compute BLE mesh features: %s", e)

                    # Enrich with camera features if samples exist
                    try:
                        camera_samples = window.get("camera")
                        if isinstance(camera_samples, list) and camera_samples:
                            camera_features = self.camera_adapter.analyze(camera_samples)
                            if camera_features:
                                window["camera_features"] = camera_features
                    except Exception as e:
                        logger.warning("Failed to compute camera features: %s", e)

                    # Enrich with environment features if samples exist
                    try:
                        weather_samples = window.get("weather")
                        road_samples = window.get("road")
                        if (isinstance(weather_samples, list) and weather_samples) or (isinstance(road_samples, list) and road_samples):
                            env_features = self.environment_adapter.analyze(weather_samples or [], road_samples or [])
                            if env_features:
                                window["environment_features"] = env_features
                    except Exception as e:
                        logger.warning("Failed to compute environment features: %s", e)

                    # write
                    path = self.buffer.write_window(window)
                    logger.debug("Wrote window to %s", path)
                    # cleanup periodically
                    self.buffer.cleanup_old_files()
                time.sleep(interval)
            except Exception as e:
                logger.exception("Error in window flush loop: %s", e)

    def get_status(self) -> dict:
        status = {"vehicle_hash": self.vehicle_hash, "sensors": {}}
        for s in self.sensors:
            status["sensors"][s.get_name()] = s.get_health()
        status["buffer"] = {"count": self.buffer.get_file_count(), "size_mb": self.buffer.get_total_size_mb()}
        return status

    def get_current_window(self) -> dict:
        """Return the current window (proxy to WindowManager) enriched like the flush loop.

        This provides a read-only snapshot similar to what the flush loop writes
        so callers (like the orchestrator) can access recent sensor samples
        without duplicating WindowManager internals.
        """
        # snapshot health
        try:
            with self._health_lock:
                health_snapshot = dict(self._health_snapshot)
        except Exception:
            health_snapshot = {}

        try:
            window = self.window_manager.get_current_window(sensor_health_snapshot=health_snapshot)
        except Exception:
            window = {}

        # enrich with identifiers
        try:
            window["vehicle_hash"] = self.vehicle_hash
            window["trip_id"] = self.trip_id
        except Exception:
            pass

        # Best-effort enrichment with adapters (mirror logic used when flushing)
        try:
            df_samples = window.get("driver_fatigue")
            if isinstance(df_samples, list) and df_samples:
                fatigue_handler = next((s for s in self.sensors if getattr(s, 'get_name', lambda: None)() == 'driver_fatigue'), None)
                events = None
                try:
                    if fatigue_handler and hasattr(fatigue_handler, 'get_events'):
                        events = fatigue_handler.get_events()
                except Exception:
                    events = None
                try:
                    fatigue_features = self.fatigue_adapter.analyze(df_samples, events=events)
                    if fatigue_features:
                        window["driver_fatigue_features"] = fatigue_features
                except Exception:
                    pass
        except Exception:
            pass

        try:
            ble_samples = window.get("ble_mesh")
            if isinstance(ble_samples, list) and ble_samples:
                try:
                    ble_features = self.ble_adapter.analyze(ble_samples)
                    if ble_features:
                        window["ble_mesh_features"] = ble_features
                except Exception:
                    pass
        except Exception:
            pass

        try:
            camera_samples = window.get("camera")
            if isinstance(camera_samples, list) and camera_samples:
                try:
                    camera_features = self.camera_adapter.analyze(camera_samples)
                    if camera_features:
                        window["camera_features"] = camera_features
                except Exception:
                    pass
        except Exception:
            pass

        try:
            weather_samples = window.get("weather")
            road_samples = window.get("road")
            if (isinstance(weather_samples, list) and weather_samples) or (isinstance(road_samples, list) and road_samples):
                try:
                    env_features = self.environment_adapter.analyze(weather_samples or [], road_samples or [])
                    if env_features:
                        window["environment_features"] = env_features
                except Exception:
                    pass
        except Exception:
            pass

        return window
