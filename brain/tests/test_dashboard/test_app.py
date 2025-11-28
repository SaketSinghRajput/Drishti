import pytest
from dashboard.app import create_app


def test_app_health_minimal():
    cfg = {"annotations_db_path": ":memory:", "telemetry_dir": "", "explanation_dir": ""}
    app = create_app(cfg)
    app.testing = True
    client = app.test_client()
    r = client.get('/health')
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, dict)
