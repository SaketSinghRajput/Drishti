import os
import json
import tempfile
import time
from datetime import datetime, timedelta

import pytest

from dashboard.analytics import AnalyticsEngine
from dashboard.annotations_db import AnnotationsDB


def write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as fh:
        for r in records:
            fh.write(json.dumps(r) + '\n')


@pytest.fixture
def sample_dirs(tmp_path):
    td = tmp_path / 'telemetry'
    ed = tmp_path / 'explanations'
    td.mkdir()
    ed.mkdir()
    return str(td), str(ed)


def test_false_alarm_rate_and_latency_and_feature_importance(sample_dirs):
    telemetry_dir, explanation_dir = sample_dirs
    # create telemetry jsonl with incident ids
    tfile = os.path.join(telemetry_dir, 't1.jsonl')
    telem = []
    # create 3 FP and 7 TP incidents
    for i in range(10):
        telem.append({"incident_id": f"inc{i}", "timestamp": int(time.time()), "alert_level": "high", "risk_score": 0.9 if i<3 else 0.2})
    write_jsonl(tfile, telem)

    # explanations with latency and shap values
    efile = os.path.join(explanation_dir, 'e1.jsonl')
    ex = []
    for i in range(10):
        ex.append({"prediction_id": f"inc{i}", "latency_ms": 50 + i, "shap_values": {"f1": i*0.1, "f2": (10-i)*0.2}})
    write_jsonl(efile, ex)
import os
import json
import tempfile
import time
from datetime import datetime, timedelta

import pytest

from dashboard.analytics import AnalyticsEngine
from dashboard.annotations_db import AnnotationsDB


def write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as fh:
        for r in records:
            fh.write(json.dumps(r) + '\n')


@pytest.fixture
def sample_dirs(tmp_path):
    td = tmp_path / 'telemetry'
    ed = tmp_path / 'explanations'
    td.mkdir()
    ed.mkdir()
    return str(td), str(ed)


def test_false_alarm_rate_and_latency_and_feature_importance(sample_dirs):
    telemetry_dir, explanation_dir = sample_dirs
    # create telemetry jsonl with incident ids
    tfile = os.path.join(telemetry_dir, 't1.jsonl')
    telem = []
    # create 3 FP and 7 TP incidents
    for i in range(10):
        telem.append({"incident_id": f"inc{i}", "timestamp": int(time.time()), "alert_level": "high", "risk_score": 0.9 if i<3 else 0.2})
    write_jsonl(tfile, telem)

    # explanations with latency and shap values
    efile = os.path.join(explanation_dir, 'e1.jsonl')
    ex = []
    for i in range(10):
        ex.append({"prediction_id": f"inc{i}", "latency_ms": 50 + i, "shap_values": {"f1": i*0.1, "f2": (10-i)*0.2}})
    write_jsonl(efile, ex)

    # annotations DB: mark first 3 as FP, others TP
    db_path = tempfile.mktemp(suffix='.db')
    ann = AnnotationsDB(db_path)
    for i in range(10):
        label = 'FP' if i < 3 else 'TP'
        ann.create_annotation(f'inc{i}', label, '', 'r1')

    ae = AnalyticsEngine(telemetry_dir, explanation_dir, cache_ttl_s=1)

    # false alarm rate
    far = ae.compute_false_alarm_rate(None, None, annotations_db=ann)
    assert 'rate' in far
    assert far['fp'] == 3
    assert far['tp'] == 7

    # latency stats
    lat = ae.compute_detection_latency_stats(None, None)
    assert lat['count'] == 10
    assert 'p95' in lat and 'p99' in lat

    # feature importance
    fi = ae.compute_feature_importance_trends(None, None, top_n=2)
    assert 'top_features' in fi
    assert len(fi['top_features']) <= 2

    ann.close()


def test_empty_logs_return_defaults(sample_dirs):
    telemetry_dir, explanation_dir = sample_dirs
    ae = AnalyticsEngine(telemetry_dir, explanation_dir)
    far = ae.compute_false_alarm_rate(None, None, annotations_db=None)
    assert isinstance(far, dict)
    lat = ae.compute_detection_latency_stats(None, None)
    assert lat.get('count', 0) == 0


def test_alert_distribution_and_sensor_reliability_and_model_perf(sample_dirs):
    telemetry_dir, explanation_dir = sample_dirs
    # telemetry: produce various alert levels and sensor_health samples
    tfile = os.path.join(telemetry_dir, 'td.jsonl')
    records = [
        {"incident_id": "i1", "alert_level": "high", "sensor_health": {"camera": {"status": "ok"}}},
        {"incident_id": "i2", "alert_level": "medium", "sensor_health": {"camera": {"status": "ok"}}},
        {"incident_id": "i3", "alert_level": "low", "sensor_health": {"camera": {"status": "error"}}},
    ]
    write_jsonl(tfile, records)

    # explanations: latencies and shap
    efile = os.path.join(explanation_dir, 'ed.jsonl')
    ex = [
        {"prediction_id": "i1", "latency_ms": 30, "shap_values": {"s1": 0.3}},
        {"prediction_id": "i2", "latency_ms": 40, "shap_values": {"s1": 0.4}},
    ]
    write_jsonl(efile, ex)

    # annotations DB for model performance
    db_path = tempfile.mktemp(suffix='.db')
    ann = AnnotationsDB(db_path)
    ann.create_annotation('i1', 'TP', '', 'r')
    ann.create_annotation('i2', 'FP', '', 'r')
    ann.create_annotation('i3', 'FN', '', 'r')

    ae = AnalyticsEngine(telemetry_dir, explanation_dir, cache_ttl_s=2)

    # alert distribution
    dist = ae.compute_alert_distribution(None, None)
    assert dist['total'] == 3
    assert 'distribution' in dist and 'high' in dist['distribution']

    # sensor reliability
    sr = ae.compute_sensor_reliability(None, None)
    assert 'camera' in sr and 'availability_pct' in sr['camera']

    # model performance summary
    mp = ae.compute_model_performance_summary(ann, None, None)
    assert mp['tp'] == 1 and mp['fp'] == 1 and mp['fn'] == 1

    # cache behavior: call again and ensure cached value returned (we can't easily introspect cache timestamps, so rely on same values)
    dist2 = ae.compute_alert_distribution(None, None)
    assert dist2 == dist
    # expire cache
    time.sleep(2.1)
    dist3 = ae.compute_alert_distribution(None, None)
    assert dist3 == dist

    ann.close()


def test_malformed_and_date_filters(sample_dirs):
    telemetry_dir, explanation_dir = sample_dirs
    # create one malformed line and one valid
    tfile = os.path.join(telemetry_dir, 'bad.jsonl')
    with open(tfile, 'w', encoding='utf-8') as fh:
        fh.write('not a json\n')
        fh.write(json.dumps({"incident_id": "x1", "alert_level": "low"}) + '\n')

    ae = AnalyticsEngine(telemetry_dir, explanation_dir)
    dist = ae.compute_alert_distribution(None, None)
    assert dist['total'] == 1
