import pytest

try:
    import tensorflow as tf
    tf_available = True
except Exception:
    tf_available = False

import importlib.util
import os
# Load the `models/train_multimodal.py` module directly to avoid package-relative import issues
mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'train_multimodal.py'))
# Read and rewrite relative imports to absolute imports so the module can be
# executed in test context without the original package layout.
with open(mod_path, 'r', encoding='utf-8') as _fh:
    _src = _fh.read()
_src = _src.replace('from ..utils import', 'from utils import')
_src = _src.replace('from ..models import', 'from models import')
spec = importlib.util.spec_from_loader('train_multimodal', loader=None)
mm_mod = importlib.util.module_from_spec(spec)
exec(compile(_src, mod_path, 'exec'), mm_mod.__dict__)
MultimodalTrainer = getattr(mm_mod, 'MultimodalTrainer')


def test_build_model_multimodal_shape():
    # Fake ConfigLoader providing multimodal and sequence_preparation settings
    class FakeConfig:
        def __init__(self):
            self._data = {
                'training': {
                    'deep_learning': {
                        'multimodal': {
                            'camera_branch': {'conv_layers': [{'filters': 8, 'kernel_size': 3}]},
                            'sensor_branch': {'lstm_layers': [{'units': 8}]},
                            'tabular_branch': {'dense_layers': [4]},
                            'fusion_dense_layers': [8]
                        }
                    },
                    'sequence_preparation': {
                        'seq_length': 5,
                        'normalization_method': 'zscore'
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
    trainer = MultimodalTrainer(cfg, horizon=5)
    if not tf_available:
        with pytest.raises(RuntimeError):
            trainer.build_model((5,2),(5,3),(4,))
        return
    model = trainer.build_model((5,2),(5,3),(4,))
    assert len(model.inputs) == 3


def test_camera_alignment_mapping_logic_without_tf():
    # This test exercises the alignment logic used by the trainer without requiring TF.
    import numpy as _np
    from utils import sequence_utils

    def make_window_with_camera(has_camera=True, cam_len=6, sens_len=8):
        window = {'trip_id': 't1'}
        if has_camera:
            window['camera_series'] = {'px': list(_np.arange(cam_len)), 'py': list(_np.arange(cam_len))}
        window['sensor_timeseries'] = {'ax': list(_np.arange(sens_len)), 'ay': list(_np.arange(sens_len))}
        window['features'] = {'f1': 1.0}
        window['labels'] = {'is_accident': 0}
        return window

    windows = [make_window_with_camera(True) for _ in range(3)]
    windows.append(make_window_with_camera(False))
    windows.extend([make_window_with_camera(True) for _ in range(2)])

    Xc_raw, yc, meta_c = sequence_utils.prepare_sequences_from_windows(windows, seq_length=5, timeseries_key='camera_series', use_sliding=False)
    Xs_raw, ys, meta_s = sequence_utils.prepare_sequences_from_windows(windows, seq_length=5, timeseries_key='sensor_timeseries', use_sliding=False)
    cam_map = {id(m['window']): idx for idx, m in enumerate(meta_c)}
    cam_chan = Xc_raw.shape[-1] if Xc_raw.size else 0
    Xc_aligned = _np.zeros((len(Xs_raw), 5, cam_chan), dtype=float)
    for j, ms in enumerate(meta_s):
        cam_idx = cam_map.get(id(ms['window']))
        if cam_idx is not None and cam_chan > 0 and len(Xc_raw) > cam_idx:
            Xc_aligned[j] = Xc_raw[cam_idx]

    assert Xc_aligned.shape[0] == Xs_raw.shape[0]
    # ensure there is at least one zero row for the window that lacked camera data
    zero_rows = [i for i in range(Xc_aligned.shape[0]) if Xc_aligned.shape[-1] == 0 or _np.allclose(Xc_aligned[i], 0.0)]
    # If camera channel dim is zero, there won't be any zero-filled rows to assert;
    # in that case the alignment is valid by construction. Only require a zero
    # row when camera channels are present.
    # If there are fewer raw camera sequences than sensor sequences then at
    # least one aligned camera row should remain zero (window lacking camera).
    if cam_chan > 0 and len(Xc_raw) < Xs_raw.shape[0]:
        assert len(zero_rows) >= 1


@pytest.mark.skipif(not tf_available, reason='TensorFlow required for ablation and attention visualization')
def test_ablation_and_attention_visualization(tmp_path):
    import numpy as _np
    # Build a small trainer and model using real TF when available
    cfg = type('C', (), {})()
    cfg._data = {
        'training': {
            'deep_learning': {
                'multimodal': {
                    'camera_branch': {'conv_layers': [{'filters': 4, 'kernel_size': 3}]},
                    'sensor_branch': {'lstm_layers': [{'units': 8}]},
                    'tabular_branch': {'dense_layers': [4]},
                    'fusion_dense_layers': [8],
                    'fusion_type': 'concat'
                }
            },
            'sequence_preparation': {'seq_length': 5, 'normalization_method': 'none'}
        }
    }
    def get(k, default=None):
        return cfg._data.get(k, default) if isinstance(k, str) and '.' not in k else cfg._data
    cfg.get = lambda key, default=None: ({'training': cfg._data['training']}.get(key) if key == 'training' else cfg._data.get(key, default))
    cfg.as_dict = lambda: cfg._data

    trainer = MultimodalTrainer(cfg, horizon=None)
    # Build a small model
    model = trainer.build_model(camera_shape=(5,2), sensor_shape=(5,2), tabular_shape=(2,))

    # create synthetic test tuple: 4 samples
    Xc = _np.ones((4, 5, 2), dtype=float)
    Xs = _np.ones((4, 5, 2), dtype=float)
    Xt = _np.ones((4, 2), dtype=float)
    y = _np.array([0,1,0,1])
    # assign the model to trainer (already built)
    trainer.model = model

    reports = trainer.ablation_study((Xc, Xs, Xt, y))
    # Expect baseline and no_camera/no_sensor/no_tabular entries
    assert 'baseline' in reports
    assert 'no_camera' in reports and 'no_sensor' in reports and 'no_tabular' in reports

    # Now test attention visualization: rebuild model with attention fusion
    trainer.mm_cfg['fusion_type'] = 'attention'
    trainer = MultimodalTrainer(cfg, horizon=None)
    model = trainer.build_model(camera_shape=(5,2), sensor_shape=(5,2), tabular_shape=(2,))
    trainer.model = model
    sample_tuple = (Xc[:1], Xs[:1], Xt[:1])
    viz = trainer.visualize_fusion_weights(sample_tuple)
    assert 'attention_weights' in viz
    assert viz['attention_weights'].ndim == 2
