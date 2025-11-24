from config.config_loader import ConfigLoader
from data_collection.sensors.road_handler import RoadHandler


def test_road_handler_read():
    cfg = ConfigLoader('config/default_config.yaml')
    rh = RoadHandler(cfg)
    r = rh.read()
    assert isinstance(r, dict)
    assert 'curvature_radius_m' in r
    assert 'speed_limit_kph' in r
    assert 'upcoming_turn' in r


def test_road_handler_simulation_values():
    cfg = ConfigLoader('config/default_config.yaml')
    rh = RoadHandler(cfg)
    for _ in range(5):
        r = rh.read()
        assert r['curvature_radius_m'] >= 0
        assert r['speed_limit_kph'] >= 0
        assert r['upcoming_turn'] in ('none', 'left', 'right', 'sharp')


def test_road_handler_sharp_turn_emitted(monkeypatch):
    # Force a segment update with a small curvature and predictable randomness
    cfg = ConfigLoader('config/default_config.yaml')
    rh = RoadHandler(cfg)
    # force the segment to be considered expired
    rh._segment_start = 0

    # make gaussian curvature small (below sharp threshold) and random.random small
    monkeypatch.setattr('random.gauss', lambda mu, sigma: 10.0)
    monkeypatch.setattr('random.random', lambda: 0.0)

    r = rh.read()
    assert r['upcoming_turn'] == 'sharp'
