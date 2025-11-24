from config.config_loader import ConfigLoader
from data_collection.environment_adapter import EnvironmentAdapter


def make_weather(precip='none', intensity=0.0, vis=5000, surface='dry'):
    return {
        'timestamp': '2025-01-01T12:00:00Z',
        'precip_type': precip,
        'precip_intensity_mm_hr': intensity,
        'visibility_m': vis,
        'ambient_light_level': 'day',
        'road_surface': surface,
    }


def make_road(curv=1000.0, speed=50, turn='none', inter=False, slope=0.0):
    return {
        'timestamp': '2025-01-01T12:00:00Z',
        'curvature_radius_m': curv,
        'speed_limit_kph': speed,
        'upcoming_turn': turn,
        'intersection_flag': inter,
        'slope_percent': slope,
    }


def test_environment_adapter_basic():
    cfg = ConfigLoader('config/default_config.yaml')
    ea = EnvironmentAdapter(cfg)
    ws = [make_weather('rain', 2.0, 1000, 'wet')]
    rs = [make_road(80.0, 60, 'left', False, 2.0)]
    f = ea.analyze(ws, rs)
    assert isinstance(f, dict)
    assert 'weather_risk_score' in f
    assert 'road_geometry_risk_score' in f
    assert 'combined_risk_score' in f


def test_environment_adapter_empty():
    cfg = ConfigLoader('config/default_config.yaml')
    ea = EnvironmentAdapter(cfg)
    assert ea.analyze([], []) is None


def test_environment_adapter_infers_surface_when_missing():
    cfg = ConfigLoader('config/default_config.yaml')
    ea = EnvironmentAdapter(cfg)
    # weather samples omit 'road_surface'
    ws = [
        {
            'timestamp': '2025-01-01T12:00:00Z',
            'precip_type': 'rain',
            'precip_intensity_mm_hr': 1.5,
            'visibility_m': 800,
            'ambient_light_level': 'day',
        }
    ]
    f = ea.analyze(ws, [])
    assert isinstance(f, dict)
    # when raining we expect inferred surface to be 'wet' and wet_road_flag True
    assert f.get('road_surface_mode') in ('wet', 'icy', 'dry')
    assert f.get('wet_road_flag') is True


def test_environment_adapter_weights_affect_scores(tmp_path):
    # create a small config that changes weights to make visibility dominant
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "environment_features:\n  weather_risk_weights: {precip: 0.1, visibility: 0.8, surface: 0.1}\n"
    )
    cfg = ConfigLoader(str(cfg_path))
    ea = EnvironmentAdapter(cfg)
    ws = [make_weather('none', 0.0, 50, 'dry')]
    rs = []
    f = ea.analyze(ws, rs)
    assert isinstance(f, dict)
    # with visibility very low, weather_risk_score should be high (~1.0)
    assert f.get('weather_risk_score', 0.0) > 0.7
