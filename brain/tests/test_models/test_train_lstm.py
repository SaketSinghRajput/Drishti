import pytest

try:
    import tensorflow as tf
    tf_available = True
except Exception:
    tf_available = False

from models.train_lstm import LSTMTrainer


def test_build_model_shape(tmp_path):
    # Fake ConfigLoader with dict-style .get() and .as_dict()
    class FakeConfig:
        def __init__(self):
            self._data = {
                'training': {
                    'deep_learning': {
                        'lstm': {
                            'layers': [{'units': 8, 'dropout': 0.0}],
                            'dense_layers': [4],
                            'optimizer': 'adam',
                            'learning_rate': 0.001
                        }
                    }
                }
            }

        def get(self, key, default=None):
            parts = key.split('.') if key else []
            cur = self._data
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return default
            return cur

        def as_dict(self):
            return dict(self._data)

    cfg = FakeConfig()
    trainer = LSTMTrainer(cfg, horizon=5)
    if not tf_available:
        with pytest.raises(RuntimeError):
            trainer.build_model((10,3))
        return
    model = trainer.build_model((10,3))
    assert model.input_shape[1:] == (10,3)


def test_train_emits_tflite_when_export_requested(tmp_path, monkeypatch):
    # Setup fake trainer with a model that has a fit() method
    from models.train_lstm import LSTMTrainer
    import numpy as _np

    class FakeModel:
        def fit(self, X, y, **kwargs):
            class H:
                history = {'loss': [0.1]}

            return H()

    class FakeConfig:
        def __init__(self):
            self._data = {'training': {'deep_learning': {'lstm': {}}, 'model_compression': {'quantization': {'enable': False}}}}

        def get(self, key, default=None):
            parts = key.split('.') if key else []
            cur = self._data
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return default
            return cur

        def as_dict(self):
            return dict(self._data)

    cfg = FakeConfig()
    trainer = LSTMTrainer(cfg, horizon=5)
    trainer.model = FakeModel()

    called = {}

    def fake_export(path, quantize=False):
        # write a small file to simulate export
        with open(path, 'wb') as fh:
            fh.write(b'tflite')
        called['path'] = path

    monkeypatch.setattr(trainer, 'export_tflite', fake_export)

    X_train = _np.zeros((4, 10, 3))
    y_train = _np.array([0, 1, 0, 1])
    # call train with export_tflite=True, quantization disabled in config
    trainer.train(X_train, y_train, X_val=None, y_val=None, output_dir=str(tmp_path), apply_pruning=False, apply_qat=False, export_tflite=True)
    outp = str(tmp_path / f"lstm_h5.tflite")
    # our fake_export will have written to a path based on horizon default; check any file created in tmp_path
    assert any(p.suffix == '.tflite' for p in tmp_path.iterdir())
