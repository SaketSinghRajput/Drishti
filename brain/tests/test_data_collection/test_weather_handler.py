from config.config_loader import ConfigLoader
from data_collection.sensors.weather_handler import WeatherHandler


def test_weather_handler_read():
    cfg = ConfigLoader('config/default_config.yaml')
    wh = WeatherHandler(cfg)
    r = wh.read()
    assert isinstance(r, dict)
    assert 'precip_type' in r
    assert 'visibility_m' in r
    assert 'road_surface' in r


def test_weather_handler_simulation_values():
    cfg = ConfigLoader('config/default_config.yaml')
    wh = WeatherHandler(cfg)
    for _ in range(5):
        r = wh.read()
        assert r['visibility_m'] >= 0
        assert r['precip_intensity_mm_hr'] >= 0
        assert r['precip_type'] in ('none', 'rain', 'snow')
