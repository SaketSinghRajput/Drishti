"""Example demonstrating runtime explainability with EdgeInferenceEngine."""
from config.config_loader import ConfigLoader
from inference.edge_inference import EdgeInferenceEngine
import time


def run_demo():
    cfg = ConfigLoader('config/default_config.yaml')
    engine = EdgeInferenceEngine(cfg, _simulation_mode=True, vehicle_hash='demo', trip_id='demo_trip')
    engine.start()
    try:
        for i in range(3):
            engine.add_sensor_reading('gps', {'timestamp': time.time(), 'speed_kph': 60 + i})
            engine.add_sensor_reading('driver_fatigue', {'timestamp': time.time(), 'drowsiness_score': 0.6 + i*0.05})
            time.sleep(1.1)
            pred = engine.get_latest_prediction()
            if pred:
                print(f"Risk: {pred.get('risk_score')}, Level: {pred.get('alert_level')}")
                print('Reason:', pred.get('reason'))
                if pred.get('explanation'):
                    print('Explanation message:', pred['explanation'].get('message'))
    finally:
        engine.stop()


if __name__ == '__main__':
    run_demo()
