import json
import pytest

from deployment.monitoring_dashboard.app import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config.update({"TESTING": True})
    return app


def test_system_stats_endpoint(app):
    client = app.test_client()
    r = client.get('/api/system_stats')
    # If psutil not installed, endpoint returns 503 with error
    if r.status_code == 503:
        assert r.get_json().get("error")
    else:
        js = r.get_json()
        assert 'cpu_percent' in js
        assert 'memory' in js
