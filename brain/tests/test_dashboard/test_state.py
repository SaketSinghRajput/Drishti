import time
from dashboard.state import DashboardState
import threading


def test_update_and_get_latest():
    s = DashboardState(max_alerts=5)
    s.update_latest_prediction({"timestamp": int(time.time()), "risk_score": 0.2, "alert_level": "low"})
    latest = s.get_latest_prediction()
    assert latest is not None
    assert "risk_score" in latest


def test_alert_history_rollover():
    s = DashboardState(max_alerts=3)
    for i in range(5):
        s.update_latest_prediction({"timestamp": i, "risk_score": i * 0.1, "alert_level": "high"})
    history = s.get_alert_history()
    assert len(history) == 3


def test_concurrent_updates_and_gets():
    s = DashboardState(max_alerts=1000)

    def writer():
        for i in range(200):
            s.update_latest_prediction({"timestamp": i, "risk_score": i * 0.01, "alert_level": "medium" if i % 2 == 0 else "low"})

    def reader(results):
        for _ in range(200):
            lv = s.get_latest_prediction()
            if lv is not None:
                results.append(lv.get('timestamp'))

    readers = []
    results = []
    w = threading.Thread(target=writer)
    for _ in range(4):
        t = threading.Thread(target=reader, args=(results,))
        readers.append(t)
    w.start()
    for t in readers:
        t.start()
    w.join()
    for t in readers:
        t.join()

    # expect some reads captured timestamps
    assert len(results) > 0
import time
from dashboard.state import DashboardState
import threading


def test_update_and_get_latest():
    s = DashboardState(max_alerts=5)
    s.update_latest_prediction({"timestamp": int(time.time()), "risk_score": 0.2, "alert_level": "low"})
    latest = s.get_latest_prediction()
    assert latest is not None
    assert "risk_score" in latest


def test_alert_history_rollover():
    s = DashboardState(max_alerts=3)
    for i in range(5):
        s.update_latest_prediction({"timestamp": i, "risk_score": i * 0.1, "alert_level": "high"})
    history = s.get_alert_history()
    assert len(history) == 3


def test_concurrent_updates_and_gets():
    s = DashboardState(max_alerts=1000)
    stop = False

    def writer():
        for i in range(200):
            s.update_latest_prediction({"timestamp": i, "risk_score": i * 0.01, "alert_level": "medium" if i % 2 == 0 else "low"})

    def reader(results):
        for _ in range(200):
            lv = s.get_latest_prediction()
            if lv is not None:
                results.append(lv.get('timestamp'))

    readers = []
    results = []
    w = threading.Thread(target=writer)
    for _ in range(4):
        t = threading.Thread(target=reader, args=(results,))
        readers.append(t)
    w.start()
    for t in readers:
        t.start()
    w.join()
    for t in readers:
        t.join()

    # expect some reads captured timestamps
    assert len(results) > 0
