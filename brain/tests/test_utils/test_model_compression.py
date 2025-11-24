import numpy as np

from utils.model_compression import create_representative_dataset


def test_create_representative_dataset():
    X = np.random.randn(10,5,3)
    gen = create_representative_dataset(X, batch_size=2)
    it = gen()
    b = next(it)
    assert isinstance(b, list)
    assert b[0].shape[0] == 2


def test_measure_model_size_tmpfile(tmp_path):
    p = tmp_path / 'm.bin'
    p.write_bytes(b'0' * 1024 * 16)
    from utils.model_compression import measure_model_size
    size_mb = measure_model_size(str(p))
    assert size_mb > 0


def test_create_representative_dataset_multi_inputs():
    import numpy as _np
    a = _np.zeros((8, 2), dtype=float)
    b = _np.ones((8, 3), dtype=float)
    gen = create_representative_dataset([a, b], batch_size=3)
    batches = list(gen())
    assert len(batches) == int(_np.ceil(8 / 3))
    first = batches[0]
    assert isinstance(first, list) and len(first) == 2


def _tf_available():
    try:
        import tensorflow as tf
        return True
    except Exception:
        return False


import pytest
from utils import model_compression as mc


@pytest.mark.skipif(not _tf_available(), reason='TensorFlow required for compression tests')
def test_pruning_and_strip(tmp_path):
    import tensorflow as tf
    # small sequential model
    inp = tf.keras.Input(shape=(5,2))
    x = tf.keras.layers.Conv1D(4, 3, padding='same', activation='relu')(inp)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    # apply pruning if tfmot available, otherwise just ensure functions raise predictably
    if getattr(mc, 'tfmot_available', False):
        pr = mc.apply_magnitude_pruning(model, initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=10, frequency=1)
        assert pr is not None
        stripped = mc.strip_pruning(pr)
        assert stripped is not None
    else:
        with pytest.raises(RuntimeError):
            mc.apply_magnitude_pruning(model)


@pytest.mark.skipif(not _tf_available(), reason='TensorFlow not available for TFLite tests')
def test_quantize_and_qat_wrapper(tmp_path):
    import tensorflow as tf
    # trivial model
    inp = tf.keras.Input(shape=(5,2))
    x = tf.keras.layers.GlobalAveragePooling1D()(inp)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    outp = str(tmp_path / 'm.tflite')
    # post-training quantization (float) should produce a tflite file
    try:
        mc.quantize_model_post_training(model, outp, int8=False, representative_dataset=None)
        assert (tmp_path / 'm.tflite').exists()
    except RuntimeError:
        pytest.skip('TFLite conversion not available in this environment')

    # QAT wrapper requires tfmot
    if getattr(mc, 'tfmot_available', False):
        qmodel = mc.quantization_aware_training_wrapper(model, mixed_precision=False)
        assert qmodel is not None
    else:
        with pytest.raises(RuntimeError):
            mc.quantization_aware_training_wrapper(model)


@pytest.mark.skipif(not _tf_available(), reason='TensorFlow required for distillation tests')
def test_distillation_trainer_changes_student_weights():
    import tensorflow as tf
    import numpy as _np
    # small synthetic dataset
    N = 32
    # Single-input case
    X = _np.random.randn(N, 4).astype(_np.float32)
    y = (_np.sum(X, axis=1) > 0).astype(_np.int32)

    # Teacher: slightly larger network
    inp = tf.keras.Input(shape=(4,))
    t = tf.keras.layers.Dense(16, activation='relu')(inp)
    tout = tf.keras.layers.Dense(1, activation='sigmoid')(t)
    teacher = tf.keras.Model(inputs=inp, outputs=tout)
    teacher.compile(optimizer='adam', loss='binary_crossentropy')
    # train teacher briefly to produce non-trivial soft labels
    teacher.fit(X, y, epochs=3, batch_size=8, verbose=0)

    # Student: smaller network
    sinp = tf.keras.Input(shape=(4,))
    s = tf.keras.layers.Dense(8, activation='relu')(sinp)
    sout = tf.keras.layers.Dense(1, activation='sigmoid')(s)
    student = tf.keras.Model(inputs=sinp, outputs=sout)
    student.compile(optimizer='adam', loss='binary_crossentropy')

    # record initial weight norm
    def weight_norm(m):
        total = 0.0
        for v in m.trainable_weights:
            try:
                arr = v.numpy()
            except Exception:
                arr = _np.asarray(v)
            total += float(_np.sum(_np.abs(arr)))
        return total

    init_norm = weight_norm(student)

    dt = mc.DistillationTrainer(teacher, student, temperature=2.0, alpha=0.5)
    history = dt.train(X, y, X_val=None, y_val=None, epochs=2, batch_size=8)
    # history should contain at least one loss entry
    assert isinstance(history, dict) or hasattr(history, 'history')
    # Accept both dict with 'loss' or Keras-style History
    if isinstance(history, dict):
        assert 'loss' in history and len(history['loss']) > 0
    else:
        assert hasattr(history, 'history') and 'loss' in history.history and len(history.history['loss']) > 0

    post_norm = weight_norm(student)
    assert post_norm != init_norm

    # Multi-input case: teacher and student accepting two inputs
    X1 = _np.random.randn(N, 2).astype(_np.float32)
    X2 = _np.random.randn(N, 2).astype(_np.float32)
    y2 = (_np.sum(X1, axis=1) + _np.sum(X2, axis=1) > 0).astype(_np.int32)

    a = tf.keras.Input(shape=(2,))
    b = tf.keras.Input(shape=(2,))
    concat = tf.keras.layers.concatenate([a, b])
    t2 = tf.keras.layers.Dense(8, activation='relu')(concat)
    tout2 = tf.keras.layers.Dense(1, activation='sigmoid')(t2)
    teacher2 = tf.keras.Model(inputs=[a, b], outputs=tout2)
    teacher2.compile(optimizer='adam', loss='binary_crossentropy')
    teacher2.fit([X1, X2], y2, epochs=2, batch_size=8, verbose=0)

    sa = tf.keras.Input(shape=(2,))
    sb = tf.keras.Input(shape=(2,))
    sconcat = tf.keras.layers.concatenate([sa, sb])
    s2 = tf.keras.layers.Dense(4, activation='relu')(sconcat)
    sout2 = tf.keras.layers.Dense(1, activation='sigmoid')(s2)
    student2 = tf.keras.Model(inputs=[sa, sb], outputs=sout2)
    student2.compile(optimizer='adam', loss='binary_crossentropy')

    init_norm2 = weight_norm(student2)
    dt2 = mc.DistillationTrainer(teacher2, student2, temperature=1.0, alpha=0.7)
    history2 = dt2.train([X1, X2], y2, epochs=2, batch_size=8)
    # check training produced loss entries
    if isinstance(history2, dict):
        assert 'loss' in history2 and len(history2['loss']) > 0
    else:
        assert hasattr(history2, 'history') and 'loss' in history2.history and len(history2.history['loss']) > 0
    post_norm2 = weight_norm(student2)
    assert post_norm2 != init_norm2
