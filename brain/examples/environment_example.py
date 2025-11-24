"""Example demonstrating WeatherHandler, RoadHandler, and EnvironmentAdapter."""
from config.config_loader import ConfigLoader
from data_collection.sensors.weather_handler import WeatherHandler
from data_collection.sensors.road_handler import RoadHandler
from data_collection.environment_adapter import EnvironmentAdapter
import time
import json


def main():
    cfg = ConfigLoader('config/default_config.yaml')
    wh = WeatherHandler(cfg)
    rh = RoadHandler(cfg)
    ea = EnvironmentAdapter(cfg)

    weather_samples = []
    road_samples = []
    for _ in range(5):
        weather_samples.append(wh.read())
        road_samples.append(rh.read())
        time.sleep(0.1)

    features = ea.analyze(weather_samples, road_samples)
    print(json.dumps(features, indent=2))


if __name__ == '__main__':
    main()
