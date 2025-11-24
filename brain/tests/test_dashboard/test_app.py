import tempfile
import os
import json
import time

import pytest

from dashboard.app import create_app


@pytest.fixture
def temp_dirs(tmp_path):
    td = tmp_path / "telemetry"
    ed = tmp_path / "explanations"
    td.mkdir()
    ed.mkdir()
    return str(td), str(ed)


@pytest.fixture
def app_client(tmp_path, temp_dirs):
    telemetry_dir, explanation_dir = temp_dirs
    cfg = {
        "annotations_db_path": ":memory:",
        "telemetry_dir": telemetry_dir,
        "explanation_dir": explanation_dir,
        "default_reviewer_id": "tester",
    }
    app = create_app(cfg)
    app.testing = True
    client = app.test_client()
    return app, client


def test_health_endpoint(app_client):
    app, client = app_client
    r = client.get('/health')
    assert r.status_code == 200
    data = r.get_json()
    assert 'status' in data


def test_incident_and_annotation_flow(app_client):
    app, client = app_client
    # create an incident in the DB
    inc_meta = {
        "incident_id": "inc-1",
        "trip_id": "trip-1",
        "timestamp": int(time.time()),
        "risk_score": 0.85,
        "alert_level": "high",
        "has_video": 0,
        "frame_count": 0,
    }
    app.annotations_db.create_incident(inc_meta["incident_id"], inc_meta)

    # list incidents
    r = client.get('/api/incidents')
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)
    assert any(i.get('incident_id') == 'inc-1' for i in data)

    # get incident detail
    r = client.get('/api/incidents/inc-1')
    assert r.status_code == 200
    d = r.get_json()
    assert d.get('incident_id') == 'inc-1'

    # post an annotation
    payload = {"incident_id": "inc-1", "label": "FP", "notes": "test", "reviewer_id": "tester"}
    r = client.post('/api/annotations', data=json.dumps(payload), content_type='application/json')
    assert r.status_code == 201
    aid = r.get_json().get('annotation_id')
    assert isinstance(aid, int)

    # fetch annotations for incident
    r = client.get('/api/annotations?incident_id=inc-1')
    assert r.status_code == 200
    anns = r.get_json()
    assert isinstance(anns, list) and len(anns) >= 1


def test_annotations_update_delete(app_client):
    app, client = app_client
    aid = app.annotations_db.create_annotation('inc-2', 'TP', 'init', 'u1')
    # update
    r = client.put(f'/api/annotations/{aid}', data=json.dumps({"notes": "updated"}), content_type='application/json')
    assert r.status_code == 200
    assert r.get_json().get('ok') is True
    # delete
    r = client.delete(f'/api/annotations/{aid}')
    assert r.status_code == 200
    assert r.get_json().get('ok') is True


def test_incident_frames_timeline_endpoints(app_client):
    app, client = app_client
    # create incident and set sensor_window_path to a temp json
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    try:
        json.dump({"timestamps": [0,1,2], "series": {"speed": [0,1,2]}}, tmp)
        tmp.close()
        app.annotations_db.create_incident('inc-frames', {"incident_id": "inc-frames", "sensor_window_path": tmp.name, "has_video": 0})
        r = client.get('/api/incidents/inc-frames/timeline')
        assert r.status_code == 200
        t = r.get_json()
        assert 'series' in t or isinstance(t, dict)
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def test_realtime_latest_and_sensor_health_and_alerts(app_client):
    app, client = app_client
    # set latest prediction
    now = int(time.time())
    app.dashboard_state.update_latest_prediction({"timestamp": now, "risk_score": 0.77, "alert_level": "high", "reason": "eye_closed"})
    r = client.get('/api/realtime/latest')
    assert r.status_code == 200
    latest = r.get_json()
    assert latest.get('risk_score') == 0.77

    # sensor health
    sh = {"camera": {"status": "ok"}, "gps": {"status": "degraded"}}
    app.dashboard_state.update_sensor_health(sh)
    r = client.get('/api/realtime/sensor_health')
    assert r.status_code == 200
    assert r.get_json().get('camera', {}).get('status') == 'ok'

    # alerts endpoint should include the alert we pushed earlier
    r = client.get('/api/alerts')
    assert r.status_code == 200
    alerts = r.get_json()
    assert isinstance(alerts, list)
    assert any(a.get('reason') == 'eye_closed' for a in alerts)


def test_analytics_endpoints_and_export(app_client):
    app, client = app_client
    # prepare telemetry and explanation JSONL
    telemetry_dir = app.analytics_engine.telemetry_dir
    explanation_dir = app.analytics_engine.explanation_dir

    # write some telemetry and explanation records
    tpath = os.path.join(telemetry_dir, 'tstats.jsonl')
    ex_path = os.path.join(explanation_dir, 'estats.jsonl')
    with open(tpath, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({"incident_id": "inc-A", "alert_level": "high"}) + '\n')
        fh.write(json.dumps({"incident_id": "inc-B", "alert_level": "low"}) + '\n')

    with open(ex_path, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({"prediction_id": "inc-A", "latency_ms": 120, "shap_values": {"f1": 0.5}}) + '\n')

    # create annotations: mark inc-A as FP
    app.annotations_db.create_annotation('inc-A', 'FP', 'note', 'tester')
    app.annotations_db.create_annotation('inc-B', 'TP', 'note', 'tester')

    # false alarm rate
    r = client.get('/api/analytics/false_alarm_rate')
    assert r.status_code == 200
    data = r.get_json()
    assert 'rate' in data and data.get('fp') >= 0

    # alert distribution
    r = client.get('/api/analytics/alert_distribution')
    assert r.status_code == 200
    ad = r.get_json()
    assert 'total' in ad and 'distribution' in ad

    # sensor reliability (should return dict even if no direct sensor logs)
    r = client.get('/api/analytics/sensor_reliability')
    assert r.status_code == 200
    sr = r.get_json()
    assert isinstance(sr, dict)

    # model performance summary
    r = client.get('/api/analytics/model_performance_summary')
    assert r.status_code == 200
    mp = r.get_json()
    assert 'tp' in mp and 'fp' in mp

    # CSV export for false_alarm_rate
    r = client.get('/api/analytics/export?format=csv&metric=false_alarm_rate')
    assert r.status_code == 200
    assert 'text/csv' in r.headers.get('Content-Type', '')
import tempfile
import os
import json
import threading
import time
import shutil

import pytest

from dashboard.app import create_app


@pytest.fixture
def temp_dirs(tmp_path):
    td = tmp_path / "telemetry"
    ed = tmp_path / "explanations"
    td.mkdir()
    ed.mkdir()
    return str(td), str(ed)


@pytest.fixture
def app_client(tmp_path, temp_dirs):
    telemetry_dir, explanation_dir = temp_dirs
    cfg = {
        "annotations_db_path": ":memory:",
        "telemetry_dir": telemetry_dir,
        "explanation_dir": explanation_dir,
        "default_reviewer_id": "tester",
    }
    app = create_app(cfg)
    app.testing = True
    client = app.test_client()
    return app, client


def test_health_endpoint(app_client):
    app, client = app_client
    r = client.get('/health')
    assert r.status_code == 200
    data = r.get_json()
    assert 'status' in data


def test_incident_and_annotation_flow(app_client):
    app, client = app_client
    # create an incident in the DB
    inc_meta = {
        "incident_id": "inc-1",
        "trip_id": "trip-1",
        "timestamp": int(time.time()),
        "risk_score": 0.85,
        "alert_level": "high",
        "has_video": 0,
        "frame_count": 0,
    }
    app.annotations_db.create_incident(inc_meta["incident_id"], inc_meta)

    # list incidents
    r = client.get('/api/incidents')
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list)
    assert any(i.get('incident_id') == 'inc-1' for i in data)

    # get incident detail
    r = client.get('/api/incidents/inc-1')
    assert r.status_code == 200
    d = r.get_json()
    assert d.get('incident_id') == 'inc-1'

    # post an annotation
    payload = {"incident_id": "inc-1", "label": "FP", "notes": "test", "reviewer_id": "tester"}
    r = client.post('/api/annotations', data=json.dumps(payload), content_type='application/json')
    assert r.status_code == 201
    aid = r.get_json().get('annotation_id')
    assert isinstance(aid, int)

    # fetch annotations for incident
    r = client.get('/api/annotations?incident_id=inc-1')
    assert r.status_code == 200
    anns = r.get_json()
    assert isinstance(anns, list) and len(anns) >= 1


def test_annotations_update_delete(app_client):
    app, client = app_client
    aid = app.annotations_db.create_annotation('inc-2', 'TP', 'init', 'u1')
    # update
    r = client.put(f'/api/annotations/{aid}', data=json.dumps({"notes": "updated"}), content_type='application/json')
    assert r.status_code == 200
    assert r.get_json().get('ok') is True
    # delete
    r = client.delete(f'/api/annotations/{aid}')
    assert r.status_code == 200
    assert r.get_json().get('ok') is True


def test_incident_frames_timeline_endpoints(app_client):
    app, client = app_client
    # create incident and set sensor_window_path to a temp json
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    try:
        json.dump({"timestamps": [0,1,2], "series": {"speed": [0,1,2]}}, tmp)
        tmp.close()
        app.annotations_db.create_incident('inc-frames', {"incident_id": "inc-frames", "sensor_window_path": tmp.name, "has_video": 0})
        r = client.get('/api/incidents/inc-frames/timeline')
        assert r.status_code == 200
        t = r.get_json()
        assert 'series' in t or isinstance(t, dict)
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def test_realtime_latest_and_sensor_health_and_alerts(app_client):
    app, client = app_client
    # set latest prediction
    now = int(time.time())
    app.dashboard_state.update_latest_prediction({"timestamp": now, "risk_score": 0.77, "alert_level": "high", "reason": "eye_closed"})
    r = client.get('/api/realtime/latest')
    assert r.status_code == 200
    latest = r.get_json()
    assert latest.get('risk_score') == 0.77

    # sensor health
    sh = {"camera": {"status": "ok"}, "gps": {"status": "degraded"}}
    app.dashboard_state.update_sensor_health(sh)
    r = client.get('/api/realtime/sensor_health')
    assert r.status_code == 200
    assert r.get_json().get('camera', {}).get('status') == 'ok'

    # alerts endpoint should include the alert we pushed earlier
    r = client.get('/api/alerts')
    assert r.status_code == 200
    alerts = r.get_json()
    assert isinstance(alerts, list)
    assert any(a.get('reason') == 'eye_closed' for a in alerts)


def test_analytics_endpoints_and_export(app_client):
    app, client = app_client
    # prepare telemetry and explanation JSONL
    telemetry_dir = app.analytics_engine.telemetry_dir
    explanation_dir = app.analytics_engine.explanation_dir

    # write some telemetry and explanation records
    tpath = os.path.join(telemetry_dir, 'tstats.jsonl')
    ex_path = os.path.join(explanation_dir, 'estats.jsonl')
    with open(tpath, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({"incident_id": "inc-A", "alert_level": "high"}) + '\n')
        fh.write(json.dumps({"incident_id": "inc-B", "alert_level": "low"}) + '\n')

    with open(ex_path, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({"prediction_id": "inc-A", "latency_ms": 120, "shap_values": {"f1": 0.5}}) + '\n')

    # create annotations: mark inc-A as FP
    app.annotations_db.create_annotation('inc-A', 'FP', 'note', 'tester')
    app.annotations_db.create_annotation('inc-B', 'TP', 'note', 'tester')

    # false alarm rate
    r = client.get('/api/analytics/false_alarm_rate')
    assert r.status_code == 200
    data = r.get_json()
    assert 'rate' in data and data.get('fp') >= 0

    # alert distribution
    r = client.get('/api/analytics/alert_distribution')
    assert r.status_code == 200
    ad = r.get_json()
    assert 'total' in ad and 'distribution' in ad

    # sensor reliability (should return dict even if no direct sensor logs)
    r = client.get('/api/analytics/sensor_reliability')
    assert r.status_code == 200
    sr = r.get_json()
    assert isinstance(sr, dict)

    # model performance summary
    r = client.get('/api/analytics/model_performance_summary')
    assert r.status_code == 200
    mp = r.get_json()
    assert 'tp' in mp and 'fp' in mp

    # CSV export for false_alarm_rate
    r = client.get('/api/analytics/export?format=csv&metric=false_alarm_rate')
    assert r.status_code == 200
    assert 'text/csv' in r.headers.get('Content-Type', '')
import tempfile
import os
import json
import threading
import time
import shutil

import pytest

from dashboard.app import create_app


@pytest.fixture
def temp_dirs(tmp_path):
    td = tmp_path / "telemetry"
    ed = tmp_path / "explanations"
    td.mkdir()
    ed.mkdir()
    return str(td), str(ed)


@pytest.fixture
def app_client(tmp_path, temp_dirs):
    telemetry_dir, explanation_dir = temp_dirs
    cfg = {
        "annotations_db_path": ":memory:",
        "telemetry_dir": telemetry_dir,
        import tempfile
        import os
        import json
        import threading
        import time
        import shutil

        import pytest

        from dashboard.app import create_app


        @pytest.fixture
        def temp_dirs(tmp_path):
            td = tmp_path / "telemetry"
            ed = tmp_path / "explanations"
            td.mkdir()
            ed.mkdir()
            return str(td), str(ed)


        @pytest.fixture
        def app_client(tmp_path, temp_dirs):
            telemetry_dir, explanation_dir = temp_dirs
            cfg = {
                "annotations_db_path": ":memory:",
                "telemetry_dir": telemetry_dir,
                "explanation_dir": explanation_dir,
                "default_reviewer_id": "tester",
            }
            app = create_app(cfg)
            app.testing = True
            client = app.test_client()
            return app, client


        def test_health_endpoint(app_client):
            app, client = app_client
            r = client.get('/health')
            assert r.status_code == 200
            data = r.get_json()
            assert 'status' in data


        def test_incident_and_annotation_flow(app_client):
            app, client = app_client
            # create an incident in the DB
            inc_meta = {
                "incident_id": "inc-1",
                "trip_id": "trip-1",
                "timestamp": int(time.time()),
                "risk_score": 0.85,
                "alert_level": "high",
                "has_video": 0,
                "frame_count": 0,
            }
            app.annotations_db.create_incident(inc_meta["incident_id"], inc_meta)

            # list incidents
            r = client.get('/api/incidents')
            assert r.status_code == 200
            data = r.get_json()
            assert isinstance(data, list)
            assert any(i.get('incident_id') == 'inc-1' for i in data)

            # get incident detail
            r = client.get('/api/incidents/inc-1')
            assert r.status_code == 200
            d = r.get_json()
            assert d.get('incident_id') == 'inc-1'

            # post an annotation
            payload = {"incident_id": "inc-1", "label": "FP", "notes": "test", "reviewer_id": "tester"}
            r = client.post('/api/annotations', data=json.dumps(payload), content_type='application/json')
            assert r.status_code == 201
            aid = r.get_json().get('annotation_id')
            assert isinstance(aid, int)

            # fetch annotations for incident
            r = client.get('/api/annotations?incident_id=inc-1')
            assert r.status_code == 200
            anns = r.get_json()
            assert isinstance(anns, list) and len(anns) >= 1


        def test_annotations_update_delete(app_client):
            app, client = app_client
            aid = app.annotations_db.create_annotation('inc-2', 'TP', 'init', 'u1')
            # update
            r = client.put(f'/api/annotations/{aid}', data=json.dumps({"notes": "updated"}), content_type='application/json')
            assert r.status_code == 200
            assert r.get_json().get('ok') is True
            # delete
            r = client.delete(f'/api/annotations/{aid}')
            assert r.status_code == 200
            assert r.get_json().get('ok') is True


        def test_incident_frames_timeline_endpoints(app_client):
            app, client = app_client
            # create incident and set sensor_window_path to a temp json
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            try:
                json.dump({"timestamps": [0,1,2], "series": {"speed": [0,1,2]}}, tmp)
                tmp.close()
                app.annotations_db.create_incident('inc-frames', {"incident_id": "inc-frames", "sensor_window_path": tmp.name, "has_video": 0})
                r = client.get('/api/incidents/inc-frames/timeline')
                assert r.status_code == 200
                t = r.get_json()
                assert 'series' in t or isinstance(t, dict)
            finally:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass


        def test_realtime_latest_and_sensor_health_and_alerts(app_client):
            app, client = app_client
            # set latest prediction
            now = int(time.time())
            app.dashboard_state.update_latest_prediction({"timestamp": now, "risk_score": 0.77, "alert_level": "high", "reason": "eye_closed"})
            r = client.get('/api/realtime/latest')
            assert r.status_code == 200
            latest = r.get_json()
            assert latest.get('risk_score') == 0.77

            # sensor health
            sh = {"camera": {"status": "ok"}, "gps": {"status": "degraded"}}
            app.dashboard_state.update_sensor_health(sh)
            r = client.get('/api/realtime/sensor_health')
            assert r.status_code == 200
            assert r.get_json().get('camera', {}).get('status') == 'ok'

            # alerts endpoint should include the alert we pushed earlier
            r = client.get('/api/alerts')
            assert r.status_code == 200
            alerts = r.get_json()
            assert isinstance(alerts, list)
            assert any(a.get('reason') == 'eye_closed' for a in alerts)


        def test_analytics_endpoints_and_export(app_client):
            app, client = app_client
            # prepare telemetry and explanation JSONL
            telemetry_dir = app.analytics_engine.telemetry_dir
            explanation_dir = app.analytics_engine.explanation_dir

            # write some telemetry and explanation records
            tpath = os.path.join(telemetry_dir, 'tstats.jsonl')
            ex_path = os.path.join(explanation_dir, 'estats.jsonl')
            with open(tpath, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps({"incident_id": "inc-A", "alert_level": "high"}) + '\n')
                fh.write(json.dumps({"incident_id": "inc-B", "alert_level": "low"}) + '\n')

            with open(ex_path, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps({"prediction_id": "inc-A", "latency_ms": 120, "shap_values": {"f1": 0.5}}) + '\n')

            # create annotations: mark inc-A as FP
            app.annotations_db.create_annotation('inc-A', 'FP', 'note', 'tester')
            app.annotations_db.create_annotation('inc-B', 'TP', 'note', 'tester')

            # false alarm rate
            r = client.get('/api/analytics/false_alarm_rate')
            assert r.status_code == 200
            data = r.get_json()
            assert 'rate' in data and data.get('fp') >= 0

            # alert distribution
            r = client.get('/api/analytics/alert_distribution')
            assert r.status_code == 200
            ad = r.get_json()
            assert 'total' in ad and 'distribution' in ad

            # sensor reliability (should return dict even if no direct sensor logs)
            r = client.get('/api/analytics/sensor_reliability')
            assert r.status_code == 200
            sr = r.get_json()
            assert isinstance(sr, dict)

            # model performance summary
            r = client.get('/api/analytics/model_performance_summary')
            assert r.status_code == 200
            mp = r.get_json()
            assert 'tp' in mp and 'fp' in mp

            # CSV export for false_alarm_rate
            r = client.get('/api/analytics/export?format=csv&metric=false_alarm_rate')
            assert r.status_code == 200
            assert 'text/csv' in r.headers.get('Content-Type', '')

