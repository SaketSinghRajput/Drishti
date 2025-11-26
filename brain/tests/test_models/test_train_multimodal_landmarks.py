import numpy as np

try:
    import tensorflow as tf
    tf_available = True
except Exception:
    tf_available = False

from brain.models.train_multimodal import MultimodalTrainer


class DummyConfig:
    def __init__(self):
        self._d = {
            'training': {
                'deep_learning': {
                    'multimodal': {
                        'use_landmark_sequences': True,
                        'pose_branch': {'lstm_layers': [{'units': 16}]},
                        'hand_branch': {'lstm_layers': [{'units': 16}]}
                    }
                }
            }
        }

    def get(self, k, default=None):
        return self._d.get(k, default)

    def as_dict(self):
        return self._d


def test_build_multimodal_with_landmarks():
    if not tf_available:
        return
    cfg = DummyConfig()
    trainer = MultimodalTrainer(cfg)
    # camera_shape: (seq_len, channels)
    camera_shape = (10, 8)
    sensor_shape = (10, 6)
    tabular_shape = (12,)
    pose_shape = (10, 33 * 3)
    hand_shape = (10, 21 * 3)
    model = trainer.build_model(camera_shape, sensor_shape, tabular_shape, pose_shape=pose_shape, hand_shape=hand_shape)
    assert model is not None
    # run a single forward pass with random data
    import numpy as _np
    batch = 2
    cam = _np.random.randn(batch, *camera_shape).astype(float)
    sens = _np.random.randn(batch, *sensor_shape).astype(float)
    tab = _np.random.randn(batch, *tabular_shape).astype(float)
    pose = _np.random.randn(batch, *pose_shape).astype(float)
    hand = _np.random.randn(batch, *hand_shape).astype(float)
    preds = model.predict([cam, sens, tab, pose, hand])
    assert preds.shape[0] == batch
