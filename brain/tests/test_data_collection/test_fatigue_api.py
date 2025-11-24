import json

import pytest

from data_collection.sensors.fatigue_api import create_app, get_global_state


def test_get_status_initial():
    cfg = {"sensors": {"driver_fatigue": {"fatigue_api": {"max_events_cache": 10}}}}
    app = create_app(cfg)
    client = app.test_client()
    r = client.get("/status")
    assert r.status_code == 200
    data = r.get_json()
    assert "fatigue_score" in data


def test_update_and_query():
    cfg = {"sensors": {"driver_fatigue": {"fatigue_api": {"max_events_cache": 10}}}}
    app = create_app(cfg)
    # update state
    app.state.update_status(0.85, "MICROSLEEP", {"reason": "test"})
    client = app.test_client()
    r = client.get("/status")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("state") == "MICROSLEEP"

    # query
    r2 = client.post("/query", json={"check": "microsleep"})
    assert r2.status_code == 200
    assert r2.get_json().get("result") is True


def test_events_limit():
    cfg = {"sensors": {"driver_fatigue": {"fatigue_api": {"max_events_cache": 5}}}}
    app = create_app(cfg)
    for i in range(6):
        app.state.update_status(0.5 + i * 0.01, "DROWSY", {"i": i})
    client = app.test_client()
    r = client.get("/events?limit=3")
    assert r.status_code == 200
    evs = r.get_json().get("events")
    assert len(evs) == 3
