import pytest

try:
    import tensorflow as tf
    tf_available = True
except Exception:
    tf_available = False

from models.train_cnn import CNNTrainer


def test_build_model_shape():
    # Create a minimal fake ConfigLoader with dict-style .get() used by trainers
    class FakeConfig:
        def __init__(self):
            self._data = {
                'training': {
                    'deep_learning': {
                        'cnn': {
                            'conv_layers': [{'filters': 8, 'kernel_size': 3}],
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
    trainer = CNNTrainer(cfg, horizon=5)
    if not tf_available:
        with pytest.raises(RuntimeError):
            trainer.build_model((10,3))
        return
    model = trainer.build_model((10,3))
    assert model.input_shape[1:] == (10,3)
