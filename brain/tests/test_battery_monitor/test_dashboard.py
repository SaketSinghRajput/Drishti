import json
from unittest.mock import Mock, patch

from battery_monitor.dashboard.app import create_app


@patch('battery_monitor.dashboard.app.DashboardState')
def test_telemetry_latest_returns_sample(MockState):
    state = Mock()
    state.get_latest_telemetry.return_value = {
        'timestamp': '2025-11-21T10:00:00Z',
        'pack_voltage': 48.6,
        'soc': 87.3,
    }
    MockState.return_value = state
    app = create_app(config={'battery_monitor': {'dashboard': {}}}, event_logger=None)
    client = app.test_client()
    resp = client.get('/api/telemetry/latest')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['pack_voltage'] == 48.6


def test_alerts_endpoint_uses_event_logger():
    mock_el = Mock()
    mock_el.get_recent_alerts.return_value = [{'id': 1, 'severity': 'HIGH'}]
    app = create_app(config={'battery_monitor': {'dashboard': {}}}, event_logger=mock_el)
    client = app.test_client()
    resp = client.get('/api/alerts?limit=10')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'alerts' in data and len(data['alerts']) == 1
    mock_el.get_recent_alerts.assert_called_once()


def test_telemetry_update_changes_state():
    state = Mock()
    state.get_latest_telemetry.return_value = None
    app = create_app(config={'battery_monitor': {'dashboard': {}}}, event_logger=None)
    app.dashboard_state = state
    client = app.test_client()
    sample = {'pack_voltage': 48.0}
    r = client.post('/api/telemetry/update', json=sample)
    assert r.status_code == 200
    state.update_latest.assert_called_once()


def test_telemetry_history_inmemory():
    # in-memory fallback should return flattened history with alert_id null
    state = Mock()
    samples = [
        {'timestamp': '2025-11-21T10:00:00Z', 'pack_voltage': 48.5},
        {'timestamp': '2025-11-21T10:00:01Z', 'pack_voltage': 48.6},
    ]
    state.get_history.return_value = samples
    app = create_app(config={'battery_monitor': {'dashboard': {}}}, event_logger=None)
    app.dashboard_state = state
    client = app.test_client()
    r = client.get('/api/telemetry/history')
    assert r.status_code == 200
    data = r.get_json()
    assert 'history' in data and isinstance(data['history'], list)
    assert data['history'][0]['alert_id'] is None


def test_metrics_summary_with_history():
    state = Mock()
    state.get_history.return_value = [
        {'timestamp': '2025-11-21T10:00:00Z', 'soc': 80.0, 'soh': 95.0},
        {'timestamp': '2025-11-21T10:00:01Z', 'soc': 79.5, 'soh': 94.9},
    ]
    state.get_metrics_summary.return_value = {"sample_count": 2, "avg_soc": 79.75, "min_soh": 94.9}
    app = create_app(config={'battery_monitor': {'dashboard': {}}}, event_logger=None)
    app.dashboard_state = state
    client = app.test_client()
    r = client.get('/api/metrics/summary')
    assert r.status_code == 200
    data = r.get_json()
    assert data.get('sample_count') == 2


def test_history_event_logger_error_path():
    # if EventLogger raises, endpoint should return 500
    mock_el = Mock()
    mock_el.get_recent_alerts.side_effect = Exception('db error')
    app = create_app(config={'battery_monitor': {'dashboard': {}}}, event_logger=mock_el)
    client = app.test_client()
    r = client.get('/api/telemetry/history')
    assert r.status_code == 500
