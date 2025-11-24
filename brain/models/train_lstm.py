"""LSTM/GRU trainer for sequence-based accident prediction.

This trainer mirrors the project's trainer patterns (config-driven,
load/prepare, build, train, evaluate, export). TensorFlow is used
for model building and export; imports are optional to allow tests
to import the module in environments without TF.
"""
from typing import Optional, Tuple, Dict
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


class LSTMTrainer:
    def __init__(self, config, horizon: Optional[int] = None):
        # Expect a ConfigLoader instance
        self.config_loader = config
        self.training_cfg = self.config_loader.get('training', {})
        self.dl_cfg = self.training_cfg.get('deep_learning', {})
        self.lstm_cfg = self.dl_cfg.get('lstm', {})
        # keep dict view for any legacy uses
        self.config = self.config_loader.as_dict()
        self.horizon = horizon
        self.model = None
        self.history = None

    def load_and_prepare_data(self, data_dir: str):
        # load labeled windows using existing dataset util
        labeled = dataset_utils.load_labeled_dataset(data_dir)
        # temporal split preserves trips
        train_w, val_w, test_w = dataset_utils.temporal_train_test_split(
            labeled,
            test_size=self.training_cfg.get('test_size', 0.2),
            val_size=self.training_cfg.get('val_size', 0.1)
        )
        seq_cfg = self.training_cfg.get('sequence_preparation', {})
        seq_length = int(seq_cfg.get('seq_length', 10))
        norm = seq_cfg.get('normalization_method', 'zscore')
        stride = int(seq_cfg.get('seq_stride', 1))
        # produce raw sequences without normalization so we compute stats on train only
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
        # handle empty training split: avoid computing stats on empty arrays
        if X_train_raw.size == 0:
            self.normalization_stats = {}
            X_train = X_train_raw.astype(np.float32)
            X_val = X_val_raw.astype(np.float32)
            X_test = X_test_raw.astype(np.float32)
            return X_train, y_train, X_val, y_val, X_test, y_test
        # compute normalization stats on training split and apply to all
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
            raise RuntimeError('TensorFlow is required to build LSTM models')
        cfg = self.lstm_cfg
        inputs = keras.Input(shape=input_shape, name='sensor_seq')
        x = inputs
        layers_cfg = cfg.get('layers', [{'units': 64, 'dropout': 0.2}])
        for i, layer_cfg in enumerate(layers_cfg):
            units = layer_cfg.get('units', 64)
            dropout = layer_cfg.get('dropout', 0.0)
            cell_type = layer_cfg.get('cell_type', cfg.get('cell_type', 'lstm'))
            rnn_cls = keras.layers.GRU if cell_type == 'gru' else keras.layers.LSTM
            if layer_cfg.get('bidirectional', False):
                x = keras.layers.Bidirectional(rnn_cls(units, return_sequences=(i < len(layers_cfg) - 1)))(x)
            else:
                x = rnn_cls(units, return_sequences=(i < len(layers_cfg) - 1))(x)
            if dropout and dropout > 0:
                x = keras.layers.Dropout(dropout)(x)
        for units in cfg.get('dense_layers', [64]):
            x = keras.layers.Dense(units, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='sigmoid', name='out')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name=f'lstm_h{self.horizon or "default"}')
        opt_name = cfg.get('optimizer', 'adam')
        lr = cfg.get('learning_rate', 1e-3)
        # resolve optimizer robustly
        optimizer = model_compression.get_optimizer(opt_name, lr)
        # choose loss: focal or binary_crossentropy
        if cfg.get('use_focal_loss', False):
            loss_fn = model_losses.focal_loss_keras(alpha=cfg.get('focal_loss_alpha', 0.25), gamma=cfg.get('focal_loss_gamma', 2.0))
        else:
            loss_fn = cfg.get('loss', 'binary_crossentropy')
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'])
        self.model = model
        LOGGER.info('Built LSTM model: %s', model.summary())
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, output_dir: Optional[str] = None,
              apply_pruning: Optional[bool] = None, apply_qat: Optional[bool] = None, export_tflite: Optional[bool] = None):
        if X_train is None or getattr(X_train, 'size', None) == 0:
            raise RuntimeError('Empty training dataset: aborting training')
        if self.model is None:
            self.build_model(input_shape=X_train.shape[1:])
        cfg = self.lstm_cfg
        if apply_pruning is None:
            apply_pruning = self.training_cfg.get('model_compression', {}).get('pruning', {}).get('enable', False)
        if apply_qat is None:
            apply_qat = self.training_cfg.get('model_compression', {}).get('quantization', {}).get('method', '') == 'qat'
        if export_tflite is None:
            export_tflite = self.training_cfg.get('tflite_export', {}).get('enable', True)
        # optionally apply QAT (wrap model before pruning)
        if apply_qat:
            try:
                mp = self.lstm_cfg.get('mixed_precision', self.training_cfg.get('mixed_precision', False))
                self.model = model_compression.quantization_aware_training_wrapper(self.model, mixed_precision=mp)
                # recompile after wrapping
                opt_name = cfg.get('optimizer', 'adam')
                lr = cfg.get('learning_rate', 1e-3)
                optimizer = model_compression.get_optimizer(opt_name, lr)
                loss_fn = model_losses.focal_loss_keras(alpha=cfg.get('focal_loss_alpha', 0.25), gamma=cfg.get('focal_loss_gamma', 2.0)) if cfg.get('use_focal_loss', False) else cfg.get('loss', 'binary_crossentropy')
                self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'])
                self._qat_applied = True
                LOGGER.info('Quantization Aware Training (QAT) enabled for LSTM')
            except Exception as exc:
                LOGGER.warning('QAT wrapper failed, continuing without QAT: %s', exc)
                self._qat_applied = False
        else:
            self._qat_applied = False
        # optionally apply pruning
        pruning_cfg = self.training_cfg.get('model_compression', {}).get('pruning', {})
        if apply_pruning and pruning_cfg.get('enable', False):
            self.model = model_compression.apply_magnitude_pruning(
                self.model,
                initial_sparsity=pruning_cfg.get('initial_sparsity', 0.0),
                final_sparsity=pruning_cfg.get('final_sparsity', 0.5),
                begin_step=pruning_cfg.get('begin_step', 0),
                end_step=pruning_cfg.get('end_step', 1000),
                frequency=pruning_cfg.get('frequency', 100)
            )
            # recompile after wrapping
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
            # optional ReduceLROnPlateau and TensorBoard callbacks controlled by config
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
        # compute class weights if requested
        class_weight = None
        if cfg.get('use_class_weight', False):
            classes = np.unique(y_train)
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

        # handle SMOTE-like oversampling (simple random oversampling to reach desired positive fraction)
        use_smote = bool(self.training_cfg.get('data_augmentation', {}).get('use_smote', False) or self.config_loader.get('training', {}).get('enable_smote', False))
        if use_smote:
            try:
                target_pos_frac = float(self.training_cfg.get('smote_sampling_strategy', self.config_loader.get('training', {}).get('smote_sampling_strategy', 0.5)))
                pos_idx = np.where(y_train == 1)[0]
                neg_idx = np.where(y_train == 0)[0]
                n = len(y_train)
                desired_pos = int(target_pos_frac * n)
                if len(pos_idx) > 0 and desired_pos > len(pos_idx):
                    need = desired_pos - len(pos_idx)
                    rep_idx = np.random.choice(pos_idx, size=need, replace=True)
                    X_train = np.concatenate([X_train, X_train[rep_idx]], axis=0)
                    y_train = np.concatenate([y_train, y_train[rep_idx]], axis=0)
            except Exception:
                LOGGER.warning('SMOTE oversampling failed; continuing without oversampling')

        # compute sample weights if requested (per-sample weights passed to model.fit)
        sample_weight = None
        use_sample_weights = bool(self.config_loader.get('training', {}).get('use_sample_weights', False) or self.training_cfg.get('use_sample_weights', False))
        if use_sample_weights:
            try:
                counts = np.bincount(y_train.astype(int))
                total = len(y_train)
                # avoid division by zero
                w0 = float(total) / (2.0 * counts[0]) if counts[0] > 0 else 1.0
                w1 = float(total) / (2.0 * counts[1]) if len(counts) > 1 and counts[1] > 0 else 1.0
                sample_weight = np.where(y_train == 1, w1, w0).astype(float)
            except Exception:
                LOGGER.warning('Failed to compute sample weights; proceeding without them')

        # optionally use SequenceDataGenerator for augmentation if enabled
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
            # when using a generator, sample_weight is not supported unless the generator yields it.
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
        # save history
        if output_dir:
            with open(os.path.join(output_dir, 'training_history.json'), 'w') as fh:
                json.dump(self.history, fh)
            # persist normalization stats if available so comparison scripts can reuse them
            try:
                if hasattr(self, 'normalization_stats') and self.normalization_stats:
                    np.savez(os.path.join(output_dir, 'normalization_stats.npz'), **self.normalization_stats)
            except Exception:
                LOGGER.warning('Failed to save normalization stats')
        # strip pruning wrappers before final save/export
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
        # optionally export to TFLite (always emit a TFLite file if requested)
        if export_tflite:
            qcfg = self.training_cfg.get('model_compression', {}).get('quantization', {})
            quant_enabled = bool(qcfg.get('enable', False))
            outp = os.path.join(output_dir or '.', f'lstm_h{self.horizon or "default"}.tflite')
            if quant_enabled:
                rep_size = qcfg.get('representative_dataset_size', 100)
                try:
                    # Build metadata and let helper handle conversion. For QAT-applied
                    # models we request a normal export so that fake-quant nodes are
                    # preserved by the converter; for post-training quantization use
                    # a representative dataset.
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
                # emit a float32 TFLite model
                try:
                    metadata = {'horizon': self.horizon, 'input_shape': tuple(X_train.shape[1:])}
                    model_compression.export_tflite_with_metadata(self.model, outp, metadata=metadata, quantize=False, representative_dataset=None)
                except Exception as exc:
                    LOGGER.warning('TFLite export failed: %s', exc)
        return self.history

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
        report = evaluation.generate_evaluation_report(y_test, probs, output_dir=output_dir or '.', model_name='lstm', thresholds=thresholds)
        return report

    def export_tflite(self, output_path: str, quantize: bool = False, representative_data=None):
        if not tf_available:
            raise RuntimeError('TensorFlow required for TFLite export')
        # Delegate to helper which also writes optional metadata
        metadata = {'horizon': self.horizon}
        return model_compression.export_tflite_with_metadata(self.model, output_path, metadata=metadata, quantize=quantize, representative_dataset=representative_data)
