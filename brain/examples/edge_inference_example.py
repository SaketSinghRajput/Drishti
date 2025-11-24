#!/usr/bin/env python3
"""Example: Real-time edge inference for accident prediction.

Demonstrates loading a model, starting the inference engine, simulating sensor
inputs, and reading real-time risk predictions.
"""
import time
import logging
from config.config_loader import ConfigLoader
from inference.edge_inference import EdgeInferenceEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    config = ConfigLoader('config/default_config.yaml')
    engine = EdgeInferenceEngine(config, vehicle_hash='demo_vehicle_001', trip_id='demo_trip_001', _simulation_mode=True)

    logger.info('Starting inference engine...')
    try:
        engine.start()
    except Exception as e:
        logger.error('Failed to start engine: %s', e)
        return

    logger.info('Simulating sensor data...')
    try:
        for i in range(30):
            gps_data = {'timestamp': time.time(), 'lat': 12.9716 + i * 0.0001, 'lon': 77.5946 + i * 0.0001, 'speed_kph': 60.0 + (i % 10)}
            engine.add_sensor_reading('gps', gps_data)
            imu_data = {'timestamp': time.time(), 'ax': 0.1, 'ay': -0.2, 'az': 9.8}
            engine.add_sensor_reading('imu', imu_data)
            vc_data = {'timestamp': time.time(), 'steering_angle': 5.0, 'brake_pressure': 0.0, 'throttle_position': 0.3}
            engine.add_sensor_reading('vehicle_controls', vc_data)

            if i % 5 == 0:
                pred = engine.get_latest_prediction()
                if pred:
                    logger.info('Prediction at t=%ds: Risk=%.3f, Alert=%s, Uncertain=%s', i, pred['risk_score'], pred['alert_level'], pred['uncertainty_flag'])
            time.sleep(1.0)
    finally:
        logger.info('Stopping engine...')
        engine.stop()


if __name__ == '__main__':
    main()
