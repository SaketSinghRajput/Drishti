def test_evaluation_imports():
    from models import compute_metrics
    assert compute_metrics is not None

def test_compute_metrics_basic():
    import numpy as np
    from models import compute_metrics
    y = np.array([0,1,0,1])
    p = np.array([0.1,0.9,0.2,0.8])
    m = compute_metrics(y,p)
    assert 'pr_auc' in m
