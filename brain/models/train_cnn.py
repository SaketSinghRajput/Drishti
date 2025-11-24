"""1D-CNN trainer for temporal pattern detection.

Skeleton implementation mirroring LSTM trainer: build/train/eval/export.
"""
from typing import Optional, Tuple
import os
import json
import logging
import numpy as np

from ..utils import sequence_utils
from ..utils import dataset_utils
from ..utils import model_compression
from ..models import evaluation
from ..models import losses as model_losses
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow import keras
    tf_available = True
except Exception:
    tf = None
    keras = None
    tf_available = False

LOGGER = logging.getLogger(__name__)


class CNNTrainer:
    def __init__(self, config, horizon: Optional[int] = None):
        # Expect ConfigLoader
        self.config_loader = config
        self.training_cfg = self.config_loader.get('training', {})
        self.dl_cfg = self.training_cfg.get('deep_learning', {})
        self.cnn_cfg = self.dl_cfg.get('cnn', {})
        self.config = self.config_loader.as_dict()
        self.horizon = horizon
        self.model = None
        self.history = None

    def load_and_prepare_data(self, data_dir: str):
        labeled = dataset_utils.load_labeled_dataset(data_dir)
        train_w, val_w, test_w = dataset_utils.temporal_train_test_split(
            labeled,
            test_size=self.training_cfg.get('test_size', 0.2),
            val_size=self.training_cfg.get('val_size', 0.1)
        )
        seq_cfg = self.training_cfg.get('sequence_preparation', {})
        seq_length = int(seq_cfg.get('seq_length', 10))
        norm = seq_cfg.get('normalization_method', 'zscore')
        stride = int(seq_cfg.get('seq_stride', 1))
        X_train_raw, y_train, _ = sequence_utils.prepare_sequences_from_windows(train_w,
                                                                            seq_length=seq_length,
                                                                            normalization='none',
                                                                            horizon=self.horizon,
                                                                            stride=stride)
        X_val_raw, y_val, _ = sequence_utils.prepare_sequences_from_windows(val_w,
                                                                        seq_length=seq_length,
                                                                        normalization='none',
                                                                        horizon=self.horizon,
                                                                        stride=stride)
        X_test_raw, y_test, _ = sequence_utils.prepare_sequences_from_windows(test_w,
                                                                          seq_length=seq_length,
                                                                          normalization='none',
                                                                          horizon=self.horizon,
                                                                          stride=stride)
        # handle empty training split
        if X_train_raw.size == 0:
            self.normalization_stats = {}
            X_train = X_train_raw.astype(np.float32)
            X_val = X_val_raw.astype(np.float32)
            X_test = X_test_raw.astype(np.float32)
            return X_train, y_train, X_val, y_val, X_test, y_test
        stats = sequence_utils.compute_normalization_stats(X_train_raw, method=norm)
        self.normalization_stats = stats
        X_train = sequence_utils.apply_normalization(X_train_raw, stats, method=norm)
        X_val = sequence_utils.apply_normalization(X_val_raw, stats, method=norm)
        X_test = sequence_utils.apply_normalization(X_test_raw, stats, method=norm)
        # fail fast on zero-channel inputs
        if X_train.shape[-1] == 0:
            raise RuntimeError('No time-series channels found after sequence preparation; aborting training')
        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_model(self, input_shape: Tuple[int, int]):
        if not tf_available:
            raise RuntimeError('TensorFlow is required to build CNN models')
        cfg = self.cnn_cfg
        inputs = keras.Input(shape=input_shape, name='sensor_seq')
        x = inputs
        # multi-scale parallel convs if configured
        branches = []
        if cfg.get('multi_scale_kernels'):
            for k in cfg.get('multi_scale_kernels', [3,5,7]):
                b = keras.layers.Conv1D(filters=cfg.get('filters', 32), kernel_size=k, activation='relu', padding='same')(x)
                b = keras.layers.GlobalMaxPool1D()(b)
                branches.append(b)
            x = keras.layers.concatenate(branches)
        else:
            for i, layer in enumerate(cfg.get('conv_layers', [{'filters':32,'kernel_size':3}])):
                prev = x
                x = keras.layers.Conv1D(filters=layer.get('filters', 32), kernel_size=layer.get('kernel_size', 3), activation=layer.get('activation','relu'), padding='same')(x)
                pool_k = layer.get('pooling')
                if pool_k:
                    x = keras.layers.MaxPool1D(pool_size=pool_k)(x)
                # optional residual connection
                if cfg.get('use_residual_connections', False):
                    # If pooling was applied, apply the same pooling to the residual branch
                    if pool_k:
                        try:
                            prev = keras.layers.MaxPool1D(pool_size=pool_k)(prev)
                        except Exception:
                            # If shapes cannot be matched via pooling, skip residual for this block
                            prev = None
                    # match channels via 1x1 conv if needed and prev exists
                    if prev is not None:
                        try:
                            prev_chan = int(prev.shape[-1]) if prev.shape[-1] is not None else None
                            curr_chan = int(x.shape[-1]) if x.shape[-1] is not None else None
                        except Exception:
                            prev_chan = None
                            curr_chan = None
                        if prev_chan is not None and curr_chan is not None and prev_chan != curr_chan:
                            prev = keras.layers.Conv1D(filters=curr_chan, kernel_size=1, padding='same')(prev)
                        if prev is not None:
                            x = keras.layers.add([x, prev])
        # optional attention over temporal axis (per-timestep weights)
        if cfg.get('use_attention', False):
            # If multi-scale kernels were used we previously applied GlobalMaxPool1D
            # per-branch which removes the temporal axis. Attention requires a
            # temporal axis to compute per-timestep weights; therefore we ignore
            # attention when multi_scale_kernels is enabled and log a warning.
            if cfg.get('multi_scale_kernels'):
                LOGGER.warning('Configuration requests attention but multi_scale_kernels is enabled; attention will be ignored.')
            else:
                # x shape: (batch, timesteps, channels)
                # compute per-timestep scores and softmax across timesteps
                scores = keras.layers.Dense(1, activation=None, name='att_score')(x)  # (batch, timesteps, 1)
                att_weights = keras.layers.Softmax(axis=1, name='att_softmax')(scores)  # softmax across time
                x = keras.layers.multiply([x, att_weights])
        x = keras.layers.Flatten()(x)
        for d in cfg.get('dense_layers', [64]):
            x = keras.layers.Dense(d, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='sigmoid', name='out')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name=f'cnn_h{self.horizon or "default"}')
        opt_name = cfg.get('optimizer', 'adam')
        lr = cfg.get('learning_rate', 1e-3)
        optimizer = model_compression.get_optimizer(opt_name, lr)
        # choose focal loss if requested
        if cfg.get('use_focal_loss', False):
            loss_fn = model_losses.focal_loss_keras(alpha=cfg.get('focal_loss_alpha', 0.25), gamma=cfg.get('focal_loss_gamma', 2.0))
        else:
            loss_fn = cfg.get('loss', 'binary_crossentropy')
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'])
        self.model = model
        LOGGER.info('Built CNN model: %s', model.summary())
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, output_dir: Optional[str] = None,
              apply_pruning: Optional[bool] = None, apply_qat: Optional[bool] = None, export_tflite: Optional[bool] = None):
        if X_train is None or getattr(X_train, 'size', None) == 0:
            raise RuntimeError('Empty training dataset: aborting training')
        if self.model is None:
            self.build_model(input_shape=X_train.shape[1:])
        cfg = self.cnn_cfg
        if apply_pruning is None:
            apply_pruning = self.training_cfg.get('model_compression', {}).get('pruning', {}).get('enable', False)
        if apply_qat is None:
            apply_qat = self.training_cfg.get('model_compression', {}).get('quantization', {}).get('method', '') == 'qat'
        if export_tflite is None:
            export_tflite = self.training_cfg.get('tflite_export', {}).get('enable', True)
        # optionally apply QAT before pruning
        if apply_qat:
            try:
                # prefer model-specific mixed_precision, fall back to top-level training cfg
                mp = self.cnn_cfg.get('mixed_precision', self.training_cfg.get('mixed_precision', False))
                self.model = model_compression.quantization_aware_training_wrapper(self.model, mixed_precision=mp)
                # recompile
                opt_name = cfg.get('optimizer', 'adam')
                lr = cfg.get('learning_rate', 1e-3)
                optimizer = model_compression.get_optimizer(opt_name, lr)
                loss_fn = model_losses.focal_loss_keras(alpha=cfg.get('focal_loss_alpha', 0.25), gamma=cfg.get('focal_loss_gamma', 2.0)) if cfg.get('use_focal_loss', False) else cfg.get('loss', 'binary_crossentropy')
                self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'])
                self._qat_applied = True
                LOGGER.info('Quantization Aware Training (QAT) enabled for CNN')
            except Exception as exc:
                LOGGER.warning('QAT wrapper failed for CNN, continuing without QAT: %s', exc)
                self._qat_applied = False
        else:
            self._qat_applied = False
        pruning_cfg = self.training_cfg.get('model_compression', {}).get('pruning', {})
        if apply_pruning and pruning_cfg.get('enable', False):
            self.model = model_compression.apply_magnitude_pruning(self.model,
                                                                   initial_sparsity=pruning_cfg.get('initial_sparsity', 0.0),
                                                                   final_sparsity=pruning_cfg.get('final_sparsity', 0.5),
                                                                   begin_step=pruning_cfg.get('begin_step', 0),
                                                                   end_step=pruning_cfg.get('end_step', 1000),
                                                                   frequency=pruning_cfg.get('frequency', 100))
            # recompile
            opt_name = cfg.get('optimizer', 'adam')
            lr = cfg.get('learning_rate', 1e-3)
            optimizer = model_compression.get_optimizer(opt_name, lr)
            loss_fn = model_losses.focal_loss_keras(alpha=cfg.get('focal_loss_alpha', 0.25), gamma=cfg.get('focal_loss_gamma', 2.0)) if cfg.get('use_focal_loss', False) else cfg.get('loss', 'binary_crossentropy')
            self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'])
        callbacks = []
        if tf_available:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.get('early_stopping_patience', 10), restore_best_weights=True)
            callbacks.append(es)
            if output_dir:
                ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, 'model_best.h5'), save_best_only=True)
                callbacks.append(ckpt)
            # optional ReduceLROnPlateau and TensorBoard callbacks
            try:
                if cfg.get('reduce_lr_on_plateau', False) or self.training_cfg.get('reduce_lr_on_plateau', False):
                    rlf = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=cfg.get('reduce_lr_factor', 0.5), patience=cfg.get('reduce_lr_patience', 5))
                    callbacks.append(rlf)
            except Exception:
                pass
            try:
                tb_dir = self.training_cfg.get('tensorboard_log_dir') or cfg.get('tensorboard_log_dir')
                if tb_dir:
                    logdir = tb_dir if os.path.isabs(tb_dir) else os.path.join(output_dir or '.', tb_dir)
                    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir)
                    callbacks.append(tb)
            except Exception:
                pass
        # If pruning was applied, ensure the pruning schedule advances by adding the UpdatePruningStep callback
        if apply_pruning and pruning_cfg.get('enable', False):
            try:
                if getattr(model_compression, 'tfmot_available', False):
                    try:
                        upd = model_compression.tfmot.sparsity.keras.UpdatePruningStep()
                        callbacks.append(upd)
                    except Exception:
                        pass
            except Exception:
                pass
        # compute class weights
        class_weight = None
        if cfg.get('use_class_weight', False):
            classes = np.unique(y_train)
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

        # handle SMOTE-like oversampling
        use_smote = bool(self.training_cfg.get('data_augmentation', {}).get('use_smote', False) or self.config_loader.get('training', {}).get('enable_smote', False))
        if use_smote:
            try:
                target_pos_frac = float(self.training_cfg.get('smote_sampling_strategy', self.config_loader.get('training', {}).get('smote_sampling_strategy', 0.5)))
                pos_idx = np.where(y_train == 1)[0]
                n = len(y_train)
                desired_pos = int(target_pos_frac * n)
                if len(pos_idx) > 0 and desired_pos > len(pos_idx):
                    need = desired_pos - len(pos_idx)
                    rep_idx = np.random.choice(pos_idx, size=need, replace=True)
                    X_train = np.concatenate([X_train, X_train[rep_idx]], axis=0)
                    y_train = np.concatenate([y_train, y_train[rep_idx]], axis=0)
            except Exception:
                LOGGER.warning('SMOTE oversampling failed for CNN; continuing without oversampling')

        # compute sample weights if requested
        sample_weight = None
        use_sample_weights = bool(self.config_loader.get('training', {}).get('use_sample_weights', False) or self.training_cfg.get('use_sample_weights', False))
        if use_sample_weights:
            try:
                counts = np.bincount(y_train.astype(int))
                total = len(y_train)
                w0 = float(total) / (2.0 * counts[0]) if counts[0] > 0 else 1.0
                w1 = float(total) / (2.0 * counts[1]) if len(counts) > 1 and counts[1] > 0 else 1.0
                sample_weight = np.where(y_train == 1, w1, w0).astype(float)
            except Exception:
                LOGGER.warning('Failed to compute sample weights for CNN; proceeding without them')

        # optionally use SequenceDataGenerator for augmentation
        seq_cfg = self.training_cfg.get('sequence_preparation', {})
        aug_cfg = seq_cfg.get('augmentation', {}) if isinstance(seq_cfg, dict) else {}
        if aug_cfg.get('enable', False):
            gen = sequence_utils.SequenceDataGenerator(
                X_train,
                y_train,
                batch_size=cfg.get('batch_size', 32),
                shuffle=True,
                augment=True,
                noise_std=aug_cfg.get('noise_std', 0.0),
                time_jitter_std=aug_cfg.get('time_jitter_std', 0.0)
            )
            steps = len(gen)
            history = self.model.fit(gen,
                                     validation_data=(X_val, y_val) if X_val is not None else None,
                                     epochs=cfg.get('epochs', 50),
                                     steps_per_epoch=steps,
                                     callbacks=callbacks,
                                     class_weight=class_weight)
        else:
            fit_kwargs = {'class_weight': class_weight} if class_weight else {}
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            history = self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val) if X_val is not None else None,
                                     epochs=cfg.get('epochs', 50),
                                     batch_size=cfg.get('batch_size', 32),
                                     callbacks=callbacks,
                                     **fit_kwargs)
        self.history = history.history if hasattr(history, 'history') else history
        if output_dir:
            with open(os.path.join(output_dir, 'training_history.json'), 'w') as fh:
                json.dump(self.history, fh)
            # persist normalization stats if available
            try:
                if hasattr(self, 'normalization_stats') and self.normalization_stats:
                    np.savez(os.path.join(output_dir, 'normalization_stats.npz'), **self.normalization_stats)
            except Exception:
                LOGGER.warning('Failed to save normalization stats')
        if apply_pruning and pruning_cfg.get('enable', False):
            try:
                self.model = model_compression.strip_pruning(self.model)
            except Exception:
                LOGGER.warning('Failed to strip pruning wrappers')
        # save canonical Keras artifact for downstream tasks
        try:
            if output_dir:
                keras_path = os.path.join(output_dir, 'model.h5')
                try:
                    self.model.save(keras_path)
                except Exception:
                    LOGGER.warning('Failed to save canonical Keras model to %s', keras_path)
        except Exception:
            pass
        # export to TFLite if requested (always attempt to write a TFLite file)
        if export_tflite:
            qcfg = self.training_cfg.get('model_compression', {}).get('quantization', {})
            quant_enabled = bool(qcfg.get('enable', False))
            outp = os.path.join(output_dir or '.', f'cnn_h{self.horizon or "default"}.tflite')
            if quant_enabled:
                rep_size = qcfg.get('representative_dataset_size', 100)
                try:
                    metadata = {'horizon': self.horizon, 'input_shape': tuple(X_train.shape[1:])}
                    if getattr(self, '_qat_applied', False):
                        model_compression.export_tflite_with_metadata(self.model, outp, metadata=metadata, quantize=False, representative_dataset=None)
                    else:
                        rep_data = X_val if X_val is not None else X_train
                        rep_gen = model_compression.create_representative_dataset(rep_data[:rep_size])
                        model_compression.export_tflite_with_metadata(self.model, outp, metadata=metadata, quantize=True, representative_dataset=rep_gen)
                except Exception as exc:
                    LOGGER.warning('TFLite quantization failed: %s', exc)
            else:
                try:
                    metadata = {'horizon': self.horizon, 'input_shape': tuple(X_train.shape[1:])}
                    model_compression.export_tflite_with_metadata(self.model, outp, metadata=metadata, quantize=False, representative_dataset=None)
                except Exception as exc:
                    LOGGER.warning('TFLite export failed: %s', exc)
        return self.history

    def visualize_filters(self, out_dir: str):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            LOGGER.warning('matplotlib not available for filter visualization')
            return
        # find first Conv1D layer weights
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Conv1D):
                w = layer.get_weights()[0]  # (kernel_size, in_channels, out_channels)
                break
        else:
            LOGGER.warning('No Conv1D layer found to visualize')
            return
        os.makedirs(out_dir, exist_ok=True)
        ksize, in_ch, out_ch = w.shape
        # plot a subset of filters
        n = min(out_ch, 16)
        fig, axes = plt.subplots(nrows=n//4 or 1, ncols=4, figsize=(12, 3*(n//4 or 1)))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i in range(n):
            filt = w[:, :, i].mean(axis=1)
            axes[i].plot(filt)
            axes[i].set_title(f'filter_{i}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'conv_filters.png'))

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError('Model not built or loaded')
        preds = self.model.predict(X)
        return preds.ravel()

    def evaluate(self, X_test, y_test, output_dir: Optional[str] = None):
        probs = self.predict_proba(X_test)
        model_cfg = self.config_loader.get('model', {})
        thresholds_cfg = model_cfg.get('risk_thresholds') if isinstance(model_cfg, dict) else None
        thresholds = None
        if isinstance(thresholds_cfg, dict):
            ordered = []
            for k in ('low', 'medium', 'high'):
                if k in thresholds_cfg:
                    ordered.append(float(thresholds_cfg[k]))
            if ordered:
                thresholds = ordered
            else:
                try:
                    thresholds = [float(v) for v in thresholds_cfg.values()]
                except Exception:
                    thresholds = None
        elif isinstance(thresholds_cfg, (list, tuple)):
            thresholds = list(thresholds_cfg)
        report = evaluation.generate_evaluation_report(y_test, probs, output_dir=output_dir or '.', model_name='cnn', thresholds=thresholds)
        return report

    def export_tflite(self, output_path: str, quantize: bool = False, representative_data=None):
        if not tf_available:
            raise RuntimeError('TensorFlow required for TFLite export')
        metadata = {'horizon': self.horizon}
        return model_compression.export_tflite_with_metadata(self.model, output_path, metadata=metadata, quantize=quantize, representative_dataset=representative_data)
