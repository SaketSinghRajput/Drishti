"""Model compression utilities: pruning, QAT wrapper, TFLite conversion helpers.

These are utility wrappers around TensorFlow Model Optimization Toolkit
and TFLite converter. The functions are defensive: if TF or tfmot
aren't available they raise informative errors.
"""
import os
import tempfile
import numpy as np
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow_model_optimization as tfmot
    tf_available = True
    tfmot_available = True
except Exception:
    tf = None
    keras = None
    tfmot = None
    tf_available = False
    tfmot_available = False


def apply_magnitude_pruning(model, initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=1000, frequency=100):
    if not tfmot_available:
        raise RuntimeError('tensorflow_model_optimization is required for pruning')
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                                 final_sparsity=final_sparsity,
                                                                 begin_step=begin_step,
                                                                 end_step=end_step,
                                                                 frequency=frequency)
    }
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    return pruned_model


def strip_pruning(pruned_model):
    if not tfmot_available:
        raise RuntimeError('tensorflow_model_optimization is required for stripping pruning wrappers')
    return tfmot.sparsity.keras.strip_pruning(pruned_model)


def quantize_model_post_training(keras_model, output_path: str, int8: bool = False, representative_dataset=None):
    if not tf_available:
        raise RuntimeError('TensorFlow required for TFLite conversion')
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # set input/output types to int8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    return output_path


def export_tflite_with_metadata(keras_model, output_path: str, metadata: dict = None, quantize: bool = False, representative_dataset=None):
    """Export a Keras model to TFLite and optionally write a small metadata JSON.

    This is a convenience wrapper used by trainers to keep model artifact
    + metadata together for deployment.
    """
    path = None
    if quantize:
        path = quantize_model_post_training(keras_model, output_path, int8=True, representative_dataset=representative_dataset)
    else:
        if not tf_available:
            raise RuntimeError('TensorFlow required for TFLite export')
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        path = output_path
    # write metadata JSON if requested
    if metadata is not None:
        try:
            import json
            meta_path = output_path + '.metadata.json'
            with open(meta_path, 'w') as fh:
                json.dump(metadata, fh)
        except Exception:
            pass
    return path


def compare_model_accuracy(y_true, preds_a, preds_b):
    """Return a small dict comparing two prediction arrays using basic metrics.

    Uses sklearn if available, otherwise falls back to simple numpy computations.
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return {
            'roc_auc_a': float(roc_auc_score(y_true, preds_a)),
            'roc_auc_b': float(roc_auc_score(y_true, preds_b)),
            'pr_auc_a': float(average_precision_score(y_true, preds_a)),
            'pr_auc_b': float(average_precision_score(y_true, preds_b)),
        }
    except Exception:
        import numpy as _np
        def approx_pr_auc(y, p):
            # crude PR-AUC approximation by trapezoid on sorted precision-recall points
            idx = _np.argsort(p)[::-1]
            p_sorted = p[idx]
            tp = _np.cumsum(y[idx])
            fp = _np.cumsum(1 - y[idx])
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp[-1] if tp[-1] > 0 else 1)
            return float(_np.trapz(prec, rec))
        def approx_roc_auc(y, p):
            # trapezoidal approximation of ROC
            from math import fabs
            idx = _np.argsort(p)
            fpr = _np.cumsum(1 - y[idx]) / (len(y) - _np.sum(y) + 1e-8)
            tpr = _np.cumsum(y[idx]) / (_np.sum(y) + 1e-8)
            return float(_np.trapz(tpr, fpr))
        return {
            'roc_auc_a': approx_roc_auc(y_true, preds_a),
            'roc_auc_b': approx_roc_auc(y_true, preds_b),
            'pr_auc_a': approx_pr_auc(y_true, preds_a),
            'pr_auc_b': approx_pr_auc(y_true, preds_b),
        }


def create_representative_dataset(X_samples, batch_size: int = 1):
    """Create a representative dataset generator for TFLite conversion.

    Accepts either a single NumPy array (N, ...) or a tuple/list of arrays
    for multi-input models. Yields lists of input arrays per TFLite requirements.
    """
    import numpy as _np

    # handle single-array case
    if not isinstance(X_samples, (list, tuple)):
        X_samples = [X_samples]

    # ensure all inputs have same first-dimension length
    lengths = [len(x) for x in X_samples]
    if len(set(lengths)) != 1:
        raise ValueError('All input arrays must have the same number of samples')
    total = lengths[0]

    def gen():
        for i in range(0, total, batch_size):
            batch_list = []
            for arr in X_samples:
                batch = arr[i:i+batch_size]
                batch_list.append(batch.astype(_np.float32))
            yield batch_list

    return gen


def measure_model_size(path: str) -> float:
    size = os.path.getsize(path) / (1024.0 * 1024.0)
    return size


def benchmark_inference_latency(tflite_path: str, sample_input, runs: int = 100):
    try:
        import time
        import tflite_runtime.interpreter as tflite_rt
    except Exception:
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
        except Exception as exc:
            raise RuntimeError('TFLite runtime not available for benchmarking') from exc
    else:
        interpreter = tflite_rt.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Guard: sample_input must contain at least one sample per input. If empty, return NaNs.
    try:
        import numpy as _np
        def _is_empty(x):
            try:
                # numpy arrays have .size
                return hasattr(x, 'size') and int(x.size) == 0
            except Exception:
                try:
                    return len(x) == 0
                except Exception:
                    return False

        if isinstance(sample_input, (list, tuple)):
            empty_any = any(_is_empty(el) for el in sample_input)
            if empty_any:
                return float('nan'), float('nan')
        else:
            if _is_empty(sample_input):
                return float('nan'), float('nan')
    except Exception:
        # if any error while checking emptiness, continue and let normal flow attempt to run
        pass
    # set input
    import time
    # warmup
    for _ in range(5):
        for i, inp in enumerate(sample_input if isinstance(sample_input, (list, tuple)) else [sample_input]):
            interpreter.set_tensor(input_details[i]['index'], inp.astype(input_details[i]['dtype']))
        interpreter.invoke()
    # timed runs
    times = []
    for _ in range(runs):
        t0 = time.time()
        for i, inp in enumerate(sample_input if isinstance(sample_input, (list, tuple)) else [sample_input]):
            interpreter.set_tensor(input_details[i]['index'], inp.astype(input_details[i]['dtype']))
        interpreter.invoke()
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def quantization_aware_training_wrapper(keras_model, mixed_precision: bool = False):
    """Wrap a Keras model for quantization-aware training (QAT).

    This function uses `tfmot.quantization.keras.quantize_model` to insert
    fake-quant ops for weights/activations. Returns the wrapped model which
    should be compiled and trained as usual. If `mixed_precision` is True,
    the global policy will be set to `mixed_float16` before wrapping.
    """
    if not tf_available or not tfmot_available:
        raise RuntimeError('TensorFlow and tensorflow_model_optimization are required for QAT')
    # optional mixed precision: preserve and restore previous global policy
    prev_policy = None
    mp_module = None
    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as mp_module
            prev_policy = mp_module.global_policy()
            mp_module.set_global_policy('mixed_float16')
        except Exception:
            # if setting mixed precision fails, proceed without changing policy
            prev_policy = None

    # simple quantize_model wrapper; accept optional quantize_config if present
    try:
        quantize_model = tfmot.quantization.keras.quantize_model
        # prefer explicit quantize_config if the caller provides one via attribute
        # (some tfmot versions accept quantize_config kwarg)
        try:
            q_model = quantize_model(keras_model)
        except TypeError:
            # fallback in case signature differs
            q_model = quantize_model(keras_model)
        return q_model
    except Exception as exc:
        raise RuntimeError('QAT wrapping failed') from exc
    finally:
        # restore previous mixed-precision policy if we changed it
        if mixed_precision and prev_policy is not None and mp_module is not None:
            try:
                mp_module.set_global_policy(prev_policy)
            except Exception:
                # best-effort restore; if it fails, don't raise from here
                pass


__all__ = [
    'apply_magnitude_pruning', 'strip_pruning', 'quantize_model_post_training',
    'create_representative_dataset', 'measure_model_size', 'benchmark_inference_latency',
    'quantization_aware_training_wrapper'
]

LOGGER = logging.getLogger(__name__)


def get_optimizer(opt_name: str, lr: float = 1e-3):
    """Resolve optimizer name to a Keras optimizer instance.

    Accepts common aliases (case-insensitive). Falls back to Adam and logs a
    warning when the name is unrecognized.
    """
    if not tf_available:
        raise RuntimeError('TensorFlow required to construct optimizers')
    name = (opt_name or '').strip().lower() if opt_name is not None else ''
    mapping = {
        'adam': keras.optimizers.Adam,
        'sgd': keras.optimizers.SGD,
        'rmsprop': keras.optimizers.RMSprop,
        'adamw': getattr(keras.optimizers, 'AdamW', keras.optimizers.Adam),
        'adagrad': keras.optimizers.Adagrad,
        'adadelta': keras.optimizers.Adadelta,
        'nadam': keras.optimizers.Nadam,
        'adamax': keras.optimizers.Adamax,
    }
    cls = mapping.get(name)
    if cls is None:
        LOGGER.warning('Unknown optimizer name "%s"; defaulting to Adam', opt_name)
        cls = keras.optimizers.Adam
    try:
        return cls(learning_rate=lr)
    except Exception:
        # last-resort fallback
        return keras.optimizers.Adam(learning_rate=lr)


class DistillationTrainer:
    """Helper to distill a teacher model into a smaller student model.

    This is a lightweight helper that supports Keras models when TensorFlow
    is available. It trains the student using a combination of supervised loss
    and a KL-divergence-based soft-target loss from the teacher.
    """
    def __init__(self, teacher, student, temperature: float = 2.0, alpha: float = 0.5):
        self.teacher = teacher
        self.student = student
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 5, batch_size: int = 32, optimizer=None):
        if not tf_available:
            raise RuntimeError('TensorFlow required for distillation')
        # obtain teacher soft targets
        teacher_preds = self.teacher.predict(X_train)
        import numpy as _np
        teacher_preds = _np.asarray(teacher_preds).reshape((-1,))

        # prepare optimizer
        if optimizer is None:
            optimizer = keras.optimizers.Adam()

        bce_loss = keras.losses.BinaryCrossentropy()
        kld_loss = keras.losses.KLDivergence()

        # convert training data to numpy-friendly structure
        X_is_multi = isinstance(X_train, (list, tuple))
        N = len(teacher_preds)

        eps = 1e-7
        T = float(self.temperature)
        alpha = float(self.alpha)

        # precompute teacher logits from probabilities for temperature scaling
        # avoid division by zero
        t_preds_clipped = _np.clip(teacher_preds, eps, 1.0 - eps)
        teacher_logits = _np.log(t_preds_clipped / (1.0 - t_preds_clipped + eps))

        # training loop
        history = {'loss': []}
        for ep in range(int(epochs)):
            # simple shuffling
            idx = _np.arange(N)
            _np.random.shuffle(idx)
            ep_losses = []
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                if X_is_multi:
                    Xb = [(_np.asarray(x)[batch_idx]) for x in X_train]
                else:
                    Xb = _np.asarray(X_train)[batch_idx]
                yb = _np.asarray(y_train)[batch_idx]
                tb_logits = _np.asarray(teacher_logits)[batch_idx]

                # convert to tensors
                Xb_tf = [keras.backend.constant(x, dtype='float32') for x in Xb] if X_is_multi else keras.backend.constant(Xb, dtype='float32')
                yb_tf = keras.backend.constant(yb, dtype='float32')
                tb_logits_tf = keras.backend.constant(tb_logits, dtype='float32')

                with tf.GradientTape() as tape:
                    # forward pass student
                    student_out = self.student(Xb, training=True)
                    student_probs = tf.reshape(student_out, (-1,))
                    # clip and compute logits for student
                    sp_clipped = tf.clip_by_value(student_probs, eps, 1.0 - eps)
                    student_logits = tf.math.log(sp_clipped / (1.0 - sp_clipped + eps))

                    # soft targets with temperature
                    soft_student = tf.sigmoid(student_logits / T)
                    soft_teacher = tf.sigmoid(tb_logits_tf / T)

                    # losses
                    hard_loss = bce_loss(yb_tf, student_probs)
                    distill_loss = kld_loss(soft_teacher, soft_student)
                    # follow requested scaling: divide distillation term by T**2
                    total_loss = alpha * hard_loss + (1.0 - alpha) * (distill_loss / (T ** 2))

                # gradients and step
                grads = tape.gradient(total_loss, self.student.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
                ep_losses.append(float(total_loss.numpy()))

            history['loss'].append(float(_np.mean(ep_losses) if ep_losses else 0.0))

        return history
