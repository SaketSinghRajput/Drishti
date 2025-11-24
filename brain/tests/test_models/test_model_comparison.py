import numpy as np
from models.model_comparison import ModelComparator


def test_compare_metrics_aggregates():
    # create two mock models
    def pred_a(X):
        return np.zeros(len(X))
    def pred_b(X):
        return np.ones(len(X)) * 0.5
    models = [
        {'name': 'a', 'predict_fn': pred_a, 'type': 'xgboost', 'path': None},
        {'name': 'b', 'predict_fn': pred_b, 'type': 'lstm', 'path': None}
    ]
    X = np.zeros((10,5))
    y = np.array([0,1]*5)
    inputs = {'xgboost': (X,y), 'sequence': (X,y), 'multimodal': ((X,X,X), y)}
    comp = ModelComparator(models, inputs)
    df, roc, pr = comp.compare_metrics()
    assert 'model' in df.columns


def test_benchmark_edge_deployment_runs():
    def pred_a(X):
        return np.zeros(len(X))
    models = [
        {'name': 'a', 'predict_fn': pred_a, 'type': 'xgboost', 'path': None}
    ]
    X = np.zeros((5,3))
    y = np.array([0,1,0,1,0])
    inputs = {'xgboost': (X,y)}
    comp = ModelComparator(models, inputs)
    df = comp.benchmark_edge_deployment()
    assert isinstance(df, type(__import__('pandas').DataFrame()))


def test_compare_metrics_multimodal_predict_call():
    called = {'ok': False}

    def pred_multi(X):
        # Should receive a tuple of three arrays (camera, sensor, tabular)
        assert isinstance(X, tuple)
        assert len(X) == 3
        # each element should be an ndarray with first dimension = number of samples
        import numpy as _np
        assert all(hasattr(arr, 'shape') for arr in X)
        called['ok'] = True
        # return zeros matching sample count
        return _np.zeros(X[0].shape[0])

    models = [
        {'name': 'm', 'predict_fn': pred_multi, 'type': 'multimodal', 'path': None}
    ]
    X = np.zeros((10, 5))
    y = np.array([0, 1] * 5)
    inputs = {'multimodal': ((X, X, X), y)}
    comp = ModelComparator(models, inputs)
    df, roc, pr = comp.compare_metrics()
    assert called['ok']


def test_load_model_predictor_xgboost_monkeypatch(monkeypatch, tmp_path):
    # monkeypatch the XGBoostTrainer used by the script to avoid external deps
    from scripts import compare_models as cmp_mod

    class FakeXGB:
        def __init__(self, config, horizon=None):
            self._loaded = False

        def load_model(self, path):
            self._loaded = True

        def predict_proba(self, X):
            import numpy as _np
            # return 0.1 for all samples
            n = len(X) if hasattr(X, '__len__') else 1
            return _np.full(n, 0.1)

    monkeypatch.setattr('models.train_xgboost.XGBoostTrainer', FakeXGB, raising=False)

    fake_cfg = object()
    pred = cmp_mod.load_model_predictor('fake_path', 'xgboost', config=fake_cfg)
    # should be callable and return probabilities
    import numpy as _np
    arr = _np.zeros((5, 3))
    p = pred(arr)
    assert isinstance(p, _np.ndarray)
    assert p.shape[0] == 5
