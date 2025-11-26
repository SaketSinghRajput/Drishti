"""Multimodal trainer combining camera, sensor sequences, and tabular inputs.

This module provides a skeleton MultiModalTrainer that follows the
project's trainer conventions. It builds a multi-input Keras model
and exposes train/evaluate/export methods.
"""
from typing import Optional, Dict, Tuple
import os
import json
import logging
import numpy as np

from ..utils import sequence_utils
from ..utils import dataset_utils
from ..utils import model_compression
from ..models import evaluation
from ..models import losses as model_losses

try:
    import tensorflow as tf
    from tensorflow import keras
    tf_available = True
except Exception:
    tf = None
    keras = None
    tf_available = False

LOGGER = logging.getLogger(__name__)


class MultimodalTrainer:
    def __init__(self, config, horizon: Optional[int] = None):
        # Expect a ConfigLoader instance
        self.config_loader = config
        self.training_cfg = self.config_loader.get('training', {})
        self.dl_cfg = self.training_cfg.get('deep_learning', {})
        self.mm_cfg = self.dl_cfg.get('multimodal', {})
        self.config = self.config_loader.as_dict()
        self.horizon = horizon
        self.model = None
        self.history = None
        self._qat_applied = False

    def load_and_prepare_data(self, data_dir: str):
        labeled = dataset_utils.load_labeled_dataset(data_dir)
        train_w, val_w, test_w = dataset_utils.temporal_train_test_split(
            labeled,
            test_size=self.training_cfg.get('test_size', 0.2),
            val_size=self.training_cfg.get('val_size', 0.1)
        )
        # camera features: assume adapter produced 'camera_series' in windows
        seq_cfg = self.training_cfg.get('sequence_preparation', {})
        seq_length = int(seq_cfg.get('seq_length', 10))
        norm = seq_cfg.get('normalization_method', 'zscore')
        # For multimodal alignment we require exactly one sequence per window:
        # use_sliding=False ensures each window yields a single (seq_length, C)
        # subsequence by padding/truncating the full time-series.
        # produce raw sequences and compute normalization stats on training split
        Xc_train_raw, yc_train, meta_c_train = sequence_utils.prepare_sequences_from_windows(
            train_w,
            seq_length=seq_length,
            feature_keys=self.mm_cfg.get('camera_feature_keys'),
            normalization='none',
            horizon=self.horizon,
            timeseries_key='camera_series',
            use_sliding=False
        )
        Xs_train_raw, ys_train, meta_s_train = sequence_utils.prepare_sequences_from_windows(
            train_w,
            seq_length=seq_length,
            feature_keys=self.mm_cfg.get('sensor_feature_keys'),
            normalization='none',
            horizon=self.horizon,
            timeseries_key='sensor_timeseries',
            use_sliding=False
        )
        # tabular features: extract via FeatureExtractor on train_w (placeholder)
        from ..feature_engineering.extractor import FeatureExtractor
        fe = FeatureExtractor(self.config_loader)
        df_train = fe.extract_to_dataframe(train_w)
        df_val = fe.extract_to_dataframe(val_w)
        df_test = fe.extract_to_dataframe(test_w)
        # determine tabular feature columns, explicitly excluding label/metadata columns
        requested_tabular = self.mm_cfg.get('tabular_feature_keys')
        if requested_tabular:
            tab_cols = list(requested_tabular)
        else:
            # derive from dataframe columns while excluding labels/ids
            try:
                all_cols = list(df_train.columns)
            except Exception:
                all_cols = []
            def is_label_col(c: str) -> bool:
                if not isinstance(c, str):
                    return False
                if c == 'is_accident':
                    return True
                if c.startswith('label_'):
                    return True
                # horizon labels like is_accident_h10
                if c.startswith('is_accident_h'):
                    return True
                # common metadata fields
                if c in ('window_id', 'trip_id', 'window_timestamp'):
                    return True
                return False
            tab_cols = [c for c in all_cols if not is_label_col(c)]
        # remember tabular feature keys for downstream export metadata
        self.tabular_feature_keys = list(tab_cols)
        # Align tabular rows to the same window objects used to build sensor sequences
        # This ensures per-window alignment even if FeatureExtractor changes its row ordering
        try:
            windows_for_seq_train = [m['window'] for m in meta_s_train]
        except Exception:
            windows_for_seq_train = list(train_w)

        # Prepare training tabular inputs aligned to sensor windows. Val/test
        # tabular alignment must be performed after meta_s_val/meta_s_test are
        # created below (they don't exist yet at this point).
        Xt_train, _meta_tab_train = sequence_utils.prepare_tabular_sequences(windows_for_seq_train, feature_keys=tab_cols)
        # labels: use sensor stream labels consistently
        y_train = ys_train
        # compute normalization stats per-branch on training split and apply to val/test
        # Align camera sequences to sensor windows: build a map from camera meta window id -> index
        try:
            cam_map_train = {id(m['window']): idx for idx, m in enumerate(meta_c_train)} if meta_c_train is not None else {}
        except Exception:
            cam_map_train = {}
        cam_chan = Xc_train_raw.shape[-1] if getattr(Xc_train_raw, 'size', 0) else 0
        Xc_train = np.zeros((len(Xs_train_raw), seq_length, cam_chan), dtype=float)
        for j, ms in enumerate(meta_s_train):
            cam_idx = cam_map_train.get(id(ms['window']))
            if cam_idx is not None and cam_chan > 0 and len(Xc_train_raw) > cam_idx:
                Xc_train[j] = Xc_train_raw[cam_idx]

        # compute normalization stats per-branch on training split and apply to aligned arrays
        cam_stats = sequence_utils.compute_normalization_stats(Xc_train, method=norm)
        sens_stats = sequence_utils.compute_normalization_stats(Xs_train_raw, method=norm)
        self.normalization_stats = {'camera': cam_stats, 'sensor': sens_stats}
        Xc_train = sequence_utils.apply_normalization(Xc_train, cam_stats, method=norm)
        Xs_train = sequence_utils.apply_normalization(Xs_train_raw, sens_stats, method=norm)

        Xc_val_raw, yc_val, meta_c_val = sequence_utils.prepare_sequences_from_windows(
            val_w,
            seq_length=seq_length,
            feature_keys=self.mm_cfg.get('camera_feature_keys'),
            normalization='none',
            horizon=self.horizon,
            timeseries_key='camera_series',
            use_sliding=False
        )
        Xs_val_raw, ys_val, meta_s_val = sequence_utils.prepare_sequences_from_windows(
            val_w,
            seq_length=seq_length,
            feature_keys=self.mm_cfg.get('sensor_feature_keys'),
            normalization='none',
            horizon=self.horizon,
            timeseries_key='sensor_timeseries',
            use_sliding=False
        )
        Xc_test_raw, yc_test, meta_c_test = sequence_utils.prepare_sequences_from_windows(
            test_w,
            seq_length=seq_length,
            feature_keys=self.mm_cfg.get('camera_feature_keys'),
            normalization='none',
            horizon=self.horizon,
            timeseries_key='camera_series',
            use_sliding=False
        )
        Xs_test_raw, ys_test, meta_s_test = sequence_utils.prepare_sequences_from_windows(
            test_w,
            seq_length=seq_length,
            feature_keys=self.mm_cfg.get('sensor_feature_keys'),
            normalization='none',
            horizon=self.horizon,
            timeseries_key='sensor_timeseries',
            use_sliding=False
        )

        # Optionally extract landmark sequences (pose/hand) when enabled in multimodal config
        use_landmarks = bool(self.mm_cfg.get('use_landmark_sequences', False))
        if use_landmarks:
            Xp_train_raw, _yp_train, _meta_p_train = sequence_utils.prepare_sequences_from_windows(
                train_w,
                seq_length=seq_length,
                feature_keys=self.mm_cfg.get('pose_landmark_keys'),
                normalization='none',
                horizon=self.horizon,
                timeseries_key='pose_landmarks',
                use_sliding=False
            )
            Xh_train_raw, _yh_train, _meta_h_train = sequence_utils.prepare_sequences_from_windows(
                train_w,
                seq_length=seq_length,
                feature_keys=self.mm_cfg.get('hand_landmark_keys'),
                normalization='none',
                horizon=self.horizon,
                timeseries_key='hand_landmarks',
                use_sliding=False
            )
            # align shapes similar to camera alignment above
            # For simplicity, align by meta_s_train ordering
            Xp_train = np.zeros((len(Xs_train_raw), seq_length, Xp_train_raw.shape[-1] if getattr(Xp_train_raw, 'size', 0) else 0), dtype=float)
            Xh_train = np.zeros((len(Xs_train_raw), seq_length, Xh_train_raw.shape[-1] if getattr(Xh_train_raw, 'size', 0) else 0), dtype=float)
            try:
                for j, ms in enumerate(meta_s_train):
                    # find corresponding index in meta_p_train/meta_h_train by id
                    # fallback: use same index mapping as camera
                    cam_idx = cam_map_train.get(id(ms['window']))
                    if cam_idx is not None and Xp_train.shape[-1] > 0 and len(Xp_train_raw) > cam_idx:
                        Xp_train[j] = Xp_train_raw[cam_idx]
                    if cam_idx is not None and Xh_train.shape[-1] > 0 and len(Xh_train_raw) > cam_idx:
                        Xh_train[j] = Xh_train_raw[cam_idx]
            except Exception:
                pass
            # normalize landmark sequences
            if Xp_train.shape[-1] > 0:
                pose_stats = sequence_utils.compute_normalization_stats(Xp_train, method=norm)
                Xp_train = sequence_utils.apply_normalization(Xp_train, pose_stats, method=norm)
            if Xh_train.shape[-1] > 0:
                hand_stats = sequence_utils.compute_normalization_stats(Xh_train, method=norm)
                Xh_train = sequence_utils.apply_normalization(Xh_train, hand_stats, method=norm)
        else:
            Xp_train = Xh_train = None
        # Now that meta_s_val/meta_s_test exist, align val/test tabular rows
        try:
            windows_for_seq_val = [m['window'] for m in meta_s_val]
        except Exception:
            windows_for_seq_val = list(val_w)
        try:
            windows_for_seq_test = [m['window'] for m in meta_s_test]
        except Exception:
            windows_for_seq_test = list(test_w)

        Xt_val, _meta_tab_val = sequence_utils.prepare_tabular_sequences(windows_for_seq_val, feature_keys=tab_cols)
        Xt_test, _meta_tab_test = sequence_utils.prepare_tabular_sequences(windows_for_seq_test, feature_keys=tab_cols)
        # Align validation camera to sensor windows
        try:
            cam_map_val = {id(m['window']): idx for idx, m in enumerate(meta_c_val)} if meta_c_val is not None else {}
        except Exception:
            cam_map_val = {}
        cam_chan_val = Xc_val_raw.shape[-1] if getattr(Xc_val_raw, 'size', 0) else 0
        Xc_val = np.zeros((len(Xs_val_raw), seq_length, cam_chan_val), dtype=float)
        for j, ms in enumerate(meta_s_val):
            cam_idx = cam_map_val.get(id(ms['window']))
            if cam_idx is not None and cam_chan_val > 0 and len(Xc_val_raw) > cam_idx:
                Xc_val[j] = Xc_val_raw[cam_idx]

        # Align test camera to sensor windows
        try:
            cam_map_test = {id(m['window']): idx for idx, m in enumerate(meta_c_test)} if meta_c_test is not None else {}
        except Exception:
            cam_map_test = {}
        cam_chan_test = Xc_test_raw.shape[-1] if getattr(Xc_test_raw, 'size', 0) else 0
        Xc_test = np.zeros((len(Xs_test_raw), seq_length, cam_chan_test), dtype=float)
        for j, ms in enumerate(meta_s_test):
            cam_idx = cam_map_test.get(id(ms['window']))
            if cam_idx is not None and cam_chan_test > 0 and len(Xc_test_raw) > cam_idx:
                Xc_test[j] = Xc_test_raw[cam_idx]

        # apply normalization to aligned camera/sensor arrays
        Xs_val = sequence_utils.apply_normalization(Xs_val_raw, sens_stats, method=norm)
        Xs_test = sequence_utils.apply_normalization(Xs_test_raw, sens_stats, method=norm)
        Xc_val = sequence_utils.apply_normalization(Xc_val, cam_stats, method=norm)
        Xc_test = sequence_utils.apply_normalization(Xc_test, cam_stats, method=norm)

        # validate channel dimensions: fail fast if any time-series branch has zero channels
        if Xc_train.shape[-1] == 0 or Xs_train.shape[-1] == 0:
            raise RuntimeError('Camera or sensor sequences have zero channels after preparation; aborting training')

        # ensure alignment after camera->sensor mapping: one sample per originating window -> tabular rows must match
        if not (Xc_train.shape[0] == Xs_train.shape[0] == Xt_train.shape[0] == y_train.shape[0]):
            raise AssertionError('Training branches are misaligned after enforcing one-sequence-per-window and camera alignment: shapes %s, %s, %s, %s' % (
                Xc_train.shape[0], Xs_train.shape[0], Xt_train.shape[0], y_train.shape[0]
            ))
        if not (Xc_val.shape[0] == Xs_val.shape[0] == Xt_val.shape[0] == ys_val.shape[0]):
            raise AssertionError('Validation branches are misaligned after camera alignment')
        if not (Xc_test.shape[0] == Xs_test.shape[0] == Xt_test.shape[0] == ys_test.shape[0]):
            raise AssertionError('Test branches are misaligned after camera alignment')
        # labels: use sensor stream labels consistently
        y_val = ys_val
        y_test = ys_test
        # sanity: if camera-derived labels exist, assert they match sensor labels for non-empty sequences
        try:
            import numpy as _np
            if len(yc_train) == len(ys_train) and not _np.all(_np.asarray(yc_train) == _np.asarray(ys_train)):
                LOGGER.warning('Camera-derived labels differ from sensor-derived labels on training split')
        except Exception:
            pass
        # include landmark branches in returned tuples when enabled
        if use_landmarks:
            return (Xc_train, Xs_train, Xp_train, Xh_train, Xt_train, y_train), (Xc_val, Xs_val, Xp_train, Xh_train, Xt_val, y_val), (Xc_test, Xs_test, Xp_train, Xh_train, Xt_test, y_test)
        return (Xc_train, Xs_train, Xt_train, y_train), (Xc_val, Xs_val, Xt_val, y_val), (Xc_test, Xs_test, Xt_test, y_test)

    def build_model(self, camera_shape: Tuple[int, int], sensor_shape: Tuple[int, int], tabular_shape: Tuple[int], pose_shape: Optional[Tuple[int, int]] = None, hand_shape: Optional[Tuple[int, int]] = None):
        if not tf_available:
            raise RuntimeError('TensorFlow is required to build multimodal models')
        cfg = self.mm_cfg
        fusion_type = cfg.get('fusion_type', 'concat')
        # camera branch
        cam_in = keras.Input(shape=camera_shape, name='camera_seq')
        x_cam = cam_in
        # Optionally use a pretrained camera encoder if configured
        use_pretrained = bool(cfg.get('use_pretrained_camera', False))
        pretrained_path = cfg.get('pretrained_camera_path', '')
        if use_pretrained and pretrained_path:
            try:
                LOGGER.info('Loading pretrained camera encoder from %s', pretrained_path)
                enc_model = keras.models.load_model(pretrained_path, compile=False)
                # attempt to strip a final classification head by using the penultimate layer as output
                try:
                    enc_output = enc_model.layers[-2].output
                    encoder = keras.Model(inputs=enc_model.inputs, outputs=enc_output, name='pretrained_camera_encoder')
                except Exception:
                    # fallback: use the loaded model as-is
                    encoder = enc_model
                # apply encoder to the camera input
                x_cam = encoder(cam_in)
                # Optionally freeze encoder layers
                if bool(cfg.get('freeze_pretrained_camera', True)):
                    try:
                        for layer in encoder.layers:
                            layer.trainable = False
                    except Exception:
                        pass
            except Exception as exc:
                LOGGER.warning('Failed to load pretrained camera encoder (%s); falling back to Conv1D branch: %s', pretrained_path, exc)
                # fallback to learned Conv1D stack below
                x_cam = cam_in
                for layer in cfg.get('camera_branch', {}).get('conv_layers', [{'filters':32,'kernel_size':3}]):
                    x_cam = keras.layers.Conv1D(filters=layer.get('filters',32), kernel_size=layer.get('kernel_size',3), activation='relu', padding='same')(x_cam)
                x_cam = keras.layers.GlobalAveragePooling1D()(x_cam)
                x_cam = keras.layers.Dense(64, activation='relu')(x_cam)
        else:
            for layer in cfg.get('camera_branch', {}).get('conv_layers', [{'filters':32,'kernel_size':3}]):
                x_cam = keras.layers.Conv1D(filters=layer.get('filters',32), kernel_size=layer.get('kernel_size',3), activation='relu', padding='same')(x_cam)
            x_cam = keras.layers.GlobalAveragePooling1D()(x_cam)
            x_cam = keras.layers.Dense(64, activation='relu')(x_cam)

        # sensor branch
        sens_in = keras.Input(shape=sensor_shape, name='sensor_seq')
        x_sens = sens_in
        for layer in cfg.get('sensor_branch', {}).get('lstm_layers', [{'units':64}]):
            x_sens = keras.layers.LSTM(layer.get('units', 64), return_sequences=False)(x_sens)
        x_sens = keras.layers.Dense(64, activation='relu')(x_sens)

        # optional landmark branches
        pose_in = None
        hand_in = None
        x_pose = None
        x_hand = None
        use_landmarks = bool(cfg.get('use_landmark_sequences', False))
        if use_landmarks and pose_shape is not None:
            pose_in = keras.Input(shape=pose_shape, name='pose_landmarks')
            x_pose = pose_in
            for layer in cfg.get('pose_branch', {}).get('lstm_layers', [{'units':64}]):
                x_pose = keras.layers.LSTM(layer.get('units', 64), return_sequences=False)(x_pose)
            x_pose = keras.layers.Dense(32, activation='relu')(x_pose)
        if use_landmarks and hand_shape is not None:
            hand_in = keras.Input(shape=hand_shape, name='hand_landmarks')
            x_hand = hand_in
            for layer in cfg.get('hand_branch', {}).get('lstm_layers', [{'units':64}]):
                x_hand = keras.layers.LSTM(layer.get('units', 64), return_sequences=False)(x_hand)
            x_hand = keras.layers.Dense(32, activation='relu')(x_hand)

        # tabular branch
        tab_in = keras.Input(shape=tabular_shape, name='tabular')
        x_tab = tab_in
        for d in cfg.get('tabular_branch', {}).get('dense_layers', [128, 64]):
            x_tab = keras.layers.Dense(d, activation='relu')(x_tab)
        # fusion
        merged = None
        # simple concat fusion
        if fusion_type == 'concat':
            to_concat = [x_cam, x_sens, x_tab]
            if x_pose is not None:
                to_concat.append(x_pose)
            if x_hand is not None:
                to_concat.append(x_hand)
            merged = keras.layers.concatenate(to_concat)
        # attention-based fusion: learn per-modality attention weights and weighted-sum
        elif fusion_type == 'attention':
            # scalar logits per modality
            att_cam = keras.layers.Dense(1, name='att_cam')(x_cam)
            att_sens = keras.layers.Dense(1, name='att_sens')(x_sens)
            att_tab = keras.layers.Dense(1, name='att_tab')(x_tab)
            att_concat = keras.layers.concatenate([att_cam, att_sens, att_tab], axis=-1, name='att_concat')
            att_weights = keras.layers.Softmax(axis=1, name='att_weights')(att_concat)
            # split weights and apply
            w_cam = keras.layers.Lambda(lambda z: z[:, 0:1], name='w_cam')(att_weights)
            w_sens = keras.layers.Lambda(lambda z: z[:, 1:2], name='w_sens')(att_weights)
            w_tab = keras.layers.Lambda(lambda z: z[:, 2:3], name='w_tab')(att_weights)
            cam_scaled = keras.layers.Multiply()([x_cam, w_cam])
            sens_scaled = keras.layers.Multiply()([x_sens, w_sens])
            tab_scaled = keras.layers.Multiply()([x_tab, w_tab])
            merged = keras.layers.Add(name='att_fused')([cam_scaled, sens_scaled, tab_scaled])
            # store a reference so we can extract attention weights later
            self._fusion_attention_tensor_name = 'att_weights'
        # weighted-sum fusion using learned dense transforms
        elif fusion_type == 'weighted':
            # produce per-modality projection then sum
            pc = keras.layers.Dense(64, activation='relu')(x_cam)
            ps = keras.layers.Dense(64, activation='relu')(x_sens)
            pt = keras.layers.Dense(64, activation='relu')(x_tab)
            merged = keras.layers.Add()([pc, ps, pt])
        else:
            # fallback to concat
            merged = keras.layers.concatenate([x_cam, x_sens, x_tab])

        for d in cfg.get('fusion_dense_layers', [128, 64]):
            merged = keras.layers.Dense(d, activation='relu')(merged)
        out = keras.layers.Dense(1, activation='sigmoid', name='out')(merged)
        inputs = [cam_in, sens_in, tab_in]
        if pose_in is not None:
            inputs.append(pose_in)
        if hand_in is not None:
            inputs.append(hand_in)
        model = keras.Model(inputs=inputs, outputs=out, name=f'multimodal_h{self.horizon or "default"}')
        opt_name = cfg.get('optimizer', 'adam')
        lr = cfg.get('learning_rate', 1e-3)
        optimizer = model_compression.get_optimizer(opt_name, lr)
        # select loss function according to multimodal config (supports focal loss)
        if self.mm_cfg.get('use_focal_loss', False):
            fl_alpha = self.mm_cfg.get('focal_loss_alpha', 0.25)
            fl_gamma = self.mm_cfg.get('focal_loss_gamma', 2.0)
            loss_fn = model_losses.focal_loss_keras(alpha=fl_alpha, gamma=fl_gamma)
        else:
            loss_fn = self.mm_cfg.get('loss', 'binary_crossentropy')
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'])
        self.model = model
        LOGGER.info('Built multimodal model: %s', model.summary())
        return model

    def visualize_fusion_weights(self, sample_tuple, output_dir: Optional[str] = None):
        """Run the model on a small sample and return/optionally save fusion attention weights.

        This only applies when `fusion_type=='attention'` during build. Returns a dict with
        arrays for the attention weights per sample.
        """
        if self.model is None:
            raise RuntimeError('Model not built or loaded')
        fusion_type = self.mm_cfg.get('fusion_type', 'concat')
        if fusion_type != 'attention':
            raise RuntimeError('Fusion weights only available for attention fusion type')
        # build a sub-model that outputs the attention weights tensor
        try:
            att_layer = self.model.get_layer(self._fusion_attention_tensor_name)
        except Exception:
            raise RuntimeError('Attention weights layer not found in model')
        inspect_model = keras.Model(inputs=self.model.inputs, outputs=att_layer.output)
        weights = inspect_model.predict(sample_tuple)
        # weights shape: (batch, 3)
        result = {'attention_weights': weights}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            outp = os.path.join(output_dir, 'fusion_attention.npy')
            try:
                import numpy as _np
                _np.save(outp, weights)
            except Exception:
                LOGGER.warning('Failed to save attention weights')
        return result

    def ablation_study(self, test_tuple, modalities=('camera','sensor','tabular')):
        """Perform a simple ablation study by zeroing out each modality in turn.

        Returns a mapping modality -> evaluation report produced by the evaluation helper.
        """
        Xc, Xs, Xt, y = test_tuple
        reports = {}
        # baseline
        baseline_probs = self.predict_proba([Xc, Xs, Xt])
        reports['baseline'] = evaluation.generate_evaluation_report(y, baseline_probs, output_dir=None, model_name='multimodal_baseline')
        # zero-out each modality and re-evaluate
        for mod in modalities:
            if mod == 'camera':
                Xc_ab = Xc * 0
                probs = self.predict_proba([Xc_ab, Xs, Xt])
            elif mod == 'sensor':
                Xs_ab = Xs * 0
                probs = self.predict_proba([Xc, Xs_ab, Xt])
            elif mod == 'tabular':
                Xt_ab = Xt * 0
                probs = self.predict_proba([Xc, Xs, Xt_ab])
            else:
                continue
            reports[f'no_{mod}'] = evaluation.generate_evaluation_report(y, probs, output_dir=None, model_name=f'multimodal_no_{mod}')
        return reports

    def train(self, train_tuple, val_tuple=None, output_dir: Optional[str] = None,
              apply_pruning: Optional[bool] = None, apply_qat: Optional[bool] = None, export_tflite: Optional[bool] = None):
        (Xc_train, Xs_train, Xt_train, y_train) = train_tuple
        if self.model is None:
            self.build_model(camera_shape=Xc_train.shape[1:], sensor_shape=Xs_train.shape[1:], tabular_shape=(Xt_train.shape[1],))
        cfg = self.mm_cfg
        if apply_pruning is None:
            apply_pruning = self.training_cfg.get('model_compression', {}).get('pruning', {}).get('enable', False)
        if apply_qat is None:
            apply_qat = self.training_cfg.get('model_compression', {}).get('quantization', {}).get('method', '') == 'qat'
        if export_tflite is None:
            export_tflite = self.training_cfg.get('tflite_export', {}).get('enable', True)
        pruning_cfg = self.training_cfg.get('model_compression', {}).get('pruning', {})
        # allow focal loss and class weights
        use_focal = cfg.get('use_focal_loss', False)
        use_class_weight = cfg.get('use_class_weight', False)

        # Apply QAT wrapper before pruning so tfmot composes QAT+pruning correctly
        if apply_qat:
            try:
                # mixed_precision: prefer multimodal-specific setting, fall back to top-level training setting
                mp = self.mm_cfg.get('mixed_precision', self.training_cfg.get('mixed_precision', False))
                self.model = model_compression.quantization_aware_training_wrapper(self.model, mixed_precision=mp)
                # recompile with same optimizer/loss
                opt_name = cfg.get('optimizer', 'adam')
                lr = cfg.get('learning_rate', 1e-3)
                optimizer = model_compression.get_optimizer(opt_name, lr)
                if use_focal:
                    fl_alpha = cfg.get('focal_loss_alpha', 0.25)
                    fl_gamma = cfg.get('focal_loss_gamma', 2.0)
                    loss_fn = model_losses.focal_loss_keras(alpha=fl_alpha, gamma=fl_gamma)
                else:
                    loss_fn = cfg.get('loss', 'binary_crossentropy')
                self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['AUC'])
                self._qat_applied = True
                LOGGER.info('Applied QAT wrapper to multimodal model')
            except Exception as exc:
                LOGGER.warning('Failed to apply QAT wrapper: %s', exc)
        if apply_pruning and pruning_cfg.get('enable', False):
            self.model = model_compression.apply_magnitude_pruning(self.model,
                                                                   initial_sparsity=pruning_cfg.get('initial_sparsity', 0.0),
                                                                   final_sparsity=pruning_cfg.get('final_sparsity', 0.5),
                                                                   begin_step=pruning_cfg.get('begin_step', 0),
                                                                   end_step=pruning_cfg.get('end_step', 1000),
                                                                   frequency=pruning_cfg.get('frequency', 100))
            # recompile with same optimizer/loss
            opt_name = cfg.get('optimizer', 'adam')
            lr = cfg.get('learning_rate', 1e-3)
            optimizer = model_compression.get_optimizer(opt_name, lr)
            # preserve focal/class-weight selection
            if use_focal:
                fl_alpha = cfg.get('focal_loss_alpha', 0.25)
                fl_gamma = cfg.get('focal_loss_gamma', 2.0)
                loss_fn = model_losses.focal_loss_keras(alpha=fl_alpha, gamma=fl_gamma)
            else:
                loss_fn = cfg.get('loss', 'binary_crossentropy')
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
        val_data = None
        if val_tuple is not None:
            Xc_val, Xs_val, Xt_val, y_val = val_tuple
            val_data = ([Xc_val, Xs_val, Xt_val], y_val)
        # compute class weights if requested
        fit_kwargs = {}
        if use_class_weight:
            try:
                from sklearn.utils.class_weight import compute_class_weight
                import numpy as _np
                classes = _np.unique(y_train)
                cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
                fit_kwargs['class_weight'] = class_weight
            except Exception as exc:
                LOGGER.warning('Failed to compute class weights: %s', exc)

        # SMOTE-like oversampling for multimodal inputs (duplicate positive examples)
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
                    Xc_train = np.concatenate([Xc_train, Xc_train[rep_idx]], axis=0)
                    Xs_train = np.concatenate([Xs_train, Xs_train[rep_idx]], axis=0)
                    Xt_train = np.concatenate([Xt_train, Xt_train[rep_idx]], axis=0)
                    y_train = np.concatenate([y_train, y_train[rep_idx]], axis=0)
            except Exception:
                LOGGER.warning('SMOTE oversampling failed for multimodal; continuing without oversampling')

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
                fit_kwargs['sample_weight'] = sample_weight
            except Exception:
                LOGGER.warning('Failed to compute sample weights for multimodal; proceeding without them')

        history = self.model.fit([Xc_train, Xs_train, Xt_train], y_train,
                     validation_data=val_data,
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
                import numpy as _np
                if hasattr(self, 'normalization_stats') and self.normalization_stats:
                    # flatten nested stats dicts (e.g., {'camera': {'mean':..., 'std':...}, 'sensor': {...}})
                    flat = {}
                    for k, v in self.normalization_stats.items():
                        if isinstance(v, dict):
                            for subk, subv in v.items():
                                flat[f'{k}_{subk}'] = subv
                        else:
                            flat[k] = v
                    _np.savez(os.path.join(output_dir, 'normalization_stats.npz'), **flat)
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
        # optionally export to tflite
        if export_tflite:
            qcfg = self.training_cfg.get('model_compression', {}).get('quantization', {})
            quant_enabled = bool(qcfg.get('enable', False))
            outp = os.path.join(output_dir or '.', f'multimodal_h{self.horizon or "default"}.tflite')
            # Quantized export branch
            if quant_enabled:
                rep_size = qcfg.get('representative_dataset_size', 100)
                try:
                    # if QAT was applied, try direct conversion which should preserve fake-quant nodes
                    # Build metadata for export and delegate conversion to helper.
                    metadata = {
                        'horizon': self.horizon,
                        'input_shapes': {
                            'camera': tuple(Xc_train.shape[1:]) if getattr(Xc_train, 'shape', None) else None,
                            'sensor': tuple(Xs_train.shape[1:]) if getattr(Xs_train, 'shape', None) else None,
                            'tabular': (Xt_train.shape[1],) if getattr(Xt_train, 'shape', None) else None,
                        },
                        'tabular_features': getattr(self, 'tabular_feature_keys', [])
                    }
                    if getattr(self, '_qat_applied', False):
                        # QAT: request export that preserves fake-quant nodes (no post-training quant)
                        model_compression.export_tflite_with_metadata(self.model, outp, metadata=metadata, quantize=False, representative_dataset=None)
                    else:
                        # post-training quantization using representative dataset
                        rep_inputs = [Xc_train[:rep_size], Xs_train[:rep_size], Xt_train[:rep_size]]
                        rep_gen = model_compression.create_representative_dataset(rep_inputs)
                        model_compression.export_tflite_with_metadata(self.model, outp, metadata=metadata, quantize=True, representative_dataset=rep_gen)
                except Exception as exc:
                    LOGGER.warning('TFLite quantized export failed: %s', exc)
            else:
                # Float32 export branch: delegate to helper with metadata
                try:
                    metadata = {'horizon': self.horizon, 'input_shapes': {'camera': tuple(Xc_train.shape[1:]), 'sensor': tuple(Xs_train.shape[1:]), 'tabular': (Xt_train.shape[1],)}, 'tabular_features': getattr(self, 'tabular_feature_keys', [])}
                    model_compression.export_tflite_with_metadata(self.model, outp, metadata=metadata, quantize=False, representative_dataset=None)
                except Exception as exc:
                    LOGGER.warning('Float32 TFLite export failed: %s', exc)
        return self.history

    def predict_proba(self, inputs_tuple):
        if self.model is None:
            raise RuntimeError('Model not built or loaded')
        preds = self.model.predict(inputs_tuple)
        return preds.ravel()

    def evaluate(self, test_tuple, output_dir: Optional[str] = None):
        Xc_test, Xs_test, Xt_test, y_test = test_tuple
        probs = self.predict_proba([Xc_test, Xs_test, Xt_test])
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
        report = evaluation.generate_evaluation_report(y_test, probs, output_dir=output_dir or '.', model_name='multimodal', thresholds=thresholds)
        # Optionally run ablation study and persist results when enabled in config
        try:
            if self.mm_cfg.get('enable_ablation', False):
                reports = self.ablation_study((Xc_test, Xs_test, Xt_test, y_test))
                if output_dir:
                    ab_path = os.path.join(output_dir, 'ablation_summary.json')
                    try:
                        with open(ab_path, 'w') as fh:
                            json.dump(reports, fh)
                    except Exception:
                        LOGGER.warning('Failed to write ablation summary to %s', ab_path)
        except Exception as exc:
            LOGGER.warning('Ablation study failed: %s', exc)
        return report

    def export_tflite(self, output_path: str, quantize: bool = False, representative_data=None):
        if not tf_available:
            raise RuntimeError('TensorFlow required for TFLite export')
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_data is not None:
                converter.representative_dataset = representative_data
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        return output_path
