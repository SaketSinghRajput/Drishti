import threading
import time
import threading
import time
import tempfile
import json
import os

import pytest

from dashboard.app import create_app


@pytest.mark.slow
def test_e2e_polling_incident_annotation_and_analytics(tmp_path):
    # prepare dirs
    td = tmp_path / 'telemetry'
    ed = tmp_path / 'explanations'
    td.mkdir()
    ed.mkdir()

    cfg = {
        "annotations_db_path": ":memory:",
        "telemetry_dir": str(td),
        "explanation_dir": str(ed),
        "default_reviewer_id": "int_tester",
    }
    app = create_app(cfg)
    app.testing = True
    client = app.test_client()

    stop = threading.Event()

    def poller_sim():
        # simulate poller that produces a high-risk prediction and creates an incident
        i = 0
        while not stop.is_set() and i < 10:
            # push prediction
            pred = {"timestamp": int(time.time()), "risk_score": 0.9, "alert_level": "high", "incident_id": f"auto-{i}"}
            app.dashboard_state.update_latest_prediction(pred)
            # create incident in DB
            meta = {"incident_id": f"auto-{i}", "timestamp": pred['timestamp'], "risk_score": pred['risk_score'], "alert_level": pred['alert_level']}
            app.annotations_db.create_incident(meta['incident_id'], meta)
            i += 1
            time.sleep(0.1)

    t = threading.Thread(target=poller_sim, daemon=True)
    t.start()

    # wait for poller to create at least one incident
    found = False
    for _ in range(20):
        r = client.get('/api/incidents')
        assert r.status_code == 200
        incs = r.get_json()
        if incs:
            found = True
            break
        time.sleep(0.1)
    assert found, "No incidents created by poller"

    # annotate the first incident as FP
    inc0 = incs[0].get('incident_id')
    r = client.post('/api/annotations', json={"incident_id": inc0, "label": "FP", "notes": "integ test", "reviewer_id": "int_tester"})
    assert r.status_code == 201

    # allow analytics to pick up (AnalyticsEngine reads files/DB on demand)
    time.sleep(0.2)
    r = client.get('/api/analytics/model_performance_summary')
    assert r.status_code == 200
    mp = r.get_json()
    # should include at least one FP or TP depending on timing
    assert isinstance(mp.get('fp'), int)

    stop.set()
    t.join(timeout=1)
