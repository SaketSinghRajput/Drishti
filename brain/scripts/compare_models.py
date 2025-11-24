"""CLI to compare trained models (XGBoost, Keras, TFLite).
"""
import argparse
import os
import json

from models.model_comparison import ModelComparator
from config.config_loader import ConfigLoader
from utils.sequence_utils import prepare_sequences_from_windows, apply_normalization, prepare_tabular_sequences
import logging
import numpy as _np

LOGGER = logging.getLogger(__name__)
from utils.dataset_utils import load_labeled_dataset


def load_model_predictor(path, model_type, config=None):
    if model_type == 'xgboost':
        from models.train_xgboost import XGBoostTrainer
        # construct trainer with a ConfigLoader when available
        tr = XGBoostTrainer(config, horizon=None)
        tr.load_model(path)
        return lambda X: tr.predict_proba(X)
    if model_type in ('lstm','cnn','multimodal'):
        # Decide loader based on file extension: prefer Keras formats (.h5, .keras, .hdf5)
        ext = os.path.splitext(path)[1].lower()
        # helper to get known custom objects
        try:
            from models import losses as model_losses
            custom_objs = getattr(model_losses, 'custom_objects_registry', lambda: {})()
        except Exception:
            custom_objs = {}

        # Keras-style model
        if ext in ('.h5', '.keras', '.hdf5'):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objs)
                return lambda X: model.predict(X).ravel()
            except Exception as exc:
                raise RuntimeError(f'Failed to load Keras model at {path}: {exc}')

        # TFLite-style model
        if ext == '.tflite' or ext == '.lite':
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            def pred_fn(X):
                if not isinstance(X, (list, tuple)):
                    X = [X]
                for i, arr in enumerate(X):
                    det = input_details[i]
                    tensor_dtype = det.get('dtype')
                    # handle quantized inputs
                    if tensor_dtype in (_np.int8, _np.uint8):
                        scale = det.get('quantization_parameters', {}).get('scales')
                        zp = det.get('quantization_parameters', {}).get('zero_points')
                        # Explicitly check for None or empty sequence to avoid
                        # ambiguous truth-value evaluation of NumPy arrays.
                        if scale is None or len(scale) == 0:
                            scale = det.get('quantization', (1.0, 0))[0:1]
                        if zp is None or len(zp) == 0:
                            zp = det.get('quantization', (1.0, 0))[1:2]
                        try:
                            s = float(scale[0])
                        except Exception:
                            s = float(scale) if scale is not None else 1.0
                        try:
                            z = int(zp[0])
                        except Exception:
                            z = int(zp) if zp is not None else 0
                        q = _np.round(arr / s + z).astype(det['dtype'])
                        interpreter.set_tensor(det['index'], q)
                    else:
                        interpreter.set_tensor(det['index'], arr.astype(det['dtype']))
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]['index'])
                return out.ravel()
            return pred_fn

        # Unknown extension: try Keras load first, then TFLite fallback
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objs)
            return lambda X: model.predict(X).ravel()
        except Exception:
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                def pred_fn(X):
                    if not isinstance(X, (list, tuple)):
                        X = [X]
                    for i, arr in enumerate(X):
                        det = input_details[i]
                        tensor_dtype = det.get('dtype')
                        if tensor_dtype in (_np.int8, _np.uint8):
                            scale = det.get('quantization_parameters', {}).get('scales')
                            zp = det.get('quantization_parameters', {}).get('zero_points')
                            # Explicitly check for None or empty sequence to avoid
                            # ambiguous truth-value evaluation of NumPy arrays.
                            if scale is None or len(scale) == 0:
                                scale = det.get('quantization', (1.0, 0))[0:1]
                            if zp is None or len(zp) == 0:
                                zp = det.get('quantization', (1.0, 0))[1:2]
                            try:
                                s = float(scale[0])
                            except Exception:
                                s = float(scale) if scale is not None else 1.0
                            try:
                                z = int(zp[0])
                            except Exception:
                                z = int(zp) if zp is not None else 0
                            q = _np.round(arr / s + z).astype(det['dtype'])
                            interpreter.set_tensor(det['index'], q)
                        else:
                            interpreter.set_tensor(det['index'], arr.astype(det['dtype']))
                    interpreter.invoke()
                    out = interpreter.get_tensor(output_details[0]['index'])
                    return out.ravel()
                return pred_fn
            except Exception as exc:
                raise RuntimeError(f'Failed to load model at {path} as Keras or TFLite: {exc}')
    raise ValueError('unknown model type')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--test-data-dir', required=True)
    parser.add_argument('--model-paths', nargs='+', required=True)
    parser.add_argument('--model-types', nargs='+', required=True)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ensemble', nargs='+', default=None, help='List of model names to ensemble (average)')
    parser.add_argument('--significance-pairs', nargs='+', default=None, help='List of model-name pairs to test for significance, format NAME_A:NAME_B')
    parser.add_argument('--feature-importance-pair', nargs='+', default=None, help='List of model-name pairs to compare feature importances, format NAME_A:NAME_B')
    parser.add_argument('--analyze-failures', action='store_true', help='Save failure case analysis into output_dir/failures')
    parser.add_argument('--benchmark-edge', action='store_true', help='Run edge benchmark (size/latency) for models with paths')
    args = parser.parse_args()
    cfg = ConfigLoader(args.config)
    labeled = load_labeled_dataset(args.test_data_dir)
    # Prepare inputs per model type
    from feature_engineering.extractor import FeatureExtractor
    fe = FeatureExtractor(cfg)
    df = fe.extract_to_dataframe(labeled)
    # choose label column according to requested horizon
    label_col = f'is_accident_h{args.horizon}' if getattr(args, 'horizon', None) is not None else 'is_accident'
    if label_col not in df.columns:
        # fallback to 'is_accident' if specific horizon column not present
        label_col = 'is_accident'
    def is_label_or_meta_col(c: str) -> bool:
        if not isinstance(c, str):
            return False
        if c == 'is_accident':
            return True
        if c.startswith('is_accident_h'):
            return True
        if c.startswith('label_'):
            return True
        if c in ('window_id', 'trip_id', 'window_timestamp'):
            return True
        return False

    feature_cols = [c for c in df.columns if not is_label_or_meta_col(c)]
    X_xgb = df[feature_cols].to_numpy()
    y_xgb = df[label_col].to_numpy()

    # sequence inputs for LSTM/CNN (sensor_timeseries)
    training_cfg = cfg.get('training', {})
    seq_cfg = training_cfg.get('sequence_preparation', {})
    dl_cfg = training_cfg.get('deep_learning', {})
    seq_len = int(seq_cfg.get('seq_length', 10))
    norm = seq_cfg.get('normalization_method', 'zscore')
    # Build sequence inputs: use sensor_timeseries as the authoritative source for labels
    # obtain raw sequences (no normalization) so we can apply training-time stats below
    X_seq, y_seq, meta_seq = prepare_sequences_from_windows(labeled, seq_length=seq_len, normalization='none', timeseries_key='sensor_timeseries', horizon=args.horizon)

    # multimodal inputs: camera, sensor, tabular
    # use FeatureExtractor for tabular features, and sequence_utils for camera/sensor
    # For multimodal alignment, produce exactly one sequence per window
    Xc, yc, meta_c = prepare_sequences_from_windows(labeled, seq_length=seq_len, normalization='none', timeseries_key='camera_series', horizon=args.horizon, use_sliding=False)
    Xs, ys, meta_s = prepare_sequences_from_windows(labeled, seq_length=seq_len, normalization='none', timeseries_key='sensor_timeseries', horizon=args.horizon, use_sliding=False)
    # Relax strict equality: sensor-derived labels are authoritative. If camera labels differ
    # or camera sequences have different counts, log a warning and align camera sequences
    # by window identity; missing camera data will be zero-padded so comparison can proceed.
    try:
        import numpy as _np
        # check sensor vs sequence labels
        if not (len(y_seq) == len(ys) and _np.array_equal(y_seq, ys)):
            LOGGER.warning('Sensor-derived labels differ from sequence labels; using sensor-derived labels')
        # align camera sequences to sensor windows when necessary
        if len(Xc) != len(Xs):
            LOGGER.warning('Camera sequences count (%d) differs from sensor sequences count (%d); aligning and padding camera inputs', len(Xc), len(Xs))
            # determine camera channel dim
            cam_chan = Xc.shape[-1] if Xc.size else 0
            Xc_aligned = _np.zeros((len(Xs), seq_len, cam_chan), dtype=float)
            # build map by window object identity from camera meta
            cam_map = {id(m['window']): idx for idx, m in enumerate(meta_c)}
            for j, ms in enumerate(meta_s):
                cam_idx = cam_map.get(id(ms['window']))
                if cam_idx is not None and cam_chan > 0 and len(Xc) > cam_idx:
                    Xc_aligned[j] = Xc[cam_idx]
            Xc = Xc_aligned
        # choose authoritative labels
        y = y_seq
    except Exception:
        LOGGER.warning('Label alignment/checks failed; proceeding using sequence-derived labels')
        y = y_seq
    multimodal_cfg = dl_cfg.get('multimodal', {})
    requested_tabular = multimodal_cfg.get('tabular_feature_keys')
    if requested_tabular:
        tab_cols = list(requested_tabular)
    else:
        try:
            all_cols = list(df.columns)
        except Exception:
            all_cols = []
        def is_label_col(c: str) -> bool:
            if not isinstance(c, str):
                return False
            if c == 'is_accident' or c.startswith('label_') or c.startswith('is_accident_h'):
                return True
            if c in ('window_id', 'trip_id', 'window_timestamp'):
                return True
            return False
        tab_cols = [c for c in all_cols if not is_label_col(c)]
    # Align tabular rows to the windows actually used for sensor sequences
    try:
        windows_for_seq = [m['window'] for m in meta_s]
        Xt, xt_meta = prepare_tabular_sequences(windows_for_seq, feature_keys=tab_cols)
    except Exception:
        # fallback to full dataframe when alignment fails
        Xt = df[tab_cols].to_numpy()
    # use sensor-derived labels as authoritative
    y = ys

    models = []
    for p, t in zip(args.model_paths, args.model_types):
        pred_fn = load_model_predictor(p, t, config=cfg)
        models.append({'name': os.path.basename(p), 'predict_fn': pred_fn, 'type': t, 'path': p})

    # Attempt to load normalization stats from model directories and apply to sequences
    def _load_stats_from_model(model_path: str):
        # Look for normalization_stats.npz beside the model file or in the same directory
        candidates = []
        if os.path.isdir(model_path):
            candidates.append(os.path.join(model_path, 'normalization_stats.npz'))
        else:
            candidates.append(os.path.join(os.path.dirname(model_path), 'normalization_stats.npz'))
            candidates.append(model_path + '.npz')
        for c in candidates:
            if os.path.exists(c):
                try:
                    return dict(_np.load(c, allow_pickle=True))
                except Exception:
                    continue
        return None

    # for sequence inputs, try applying the stats from the first deep model that provides them
    seq_stats = None
    cam_stats = None
    sens_stats = None
    for m in models:
        if m['type'] in ('lstm', 'cnn', 'multimodal'):
            s = _load_stats_from_model(m['path'])
            if s:
                # detect multimodal flattened keys
                if any(k.startswith('camera_') for k in s.keys()):
                    # build nested dicts for apply_normalization
                    cam_stats = {k.split('camera_')[1]: s[k] for k in s.keys() if k.startswith('camera_')}
                    sens_stats = {k.split('sensor_')[1]: s[k] for k in s.keys() if k.startswith('sensor_')}
                    break
                elif 'mean' in s or 'std' in s:
                    seq_stats = s
                    break

    # apply loaded normalization if available, otherwise fall back to 'as-is' raw sequences
    if seq_stats is not None:
        X_seq = apply_normalization(X_seq, seq_stats, method=seq_cfg.get('normalization_method', 'zscore'))
    if cam_stats is not None:
        Xc = apply_normalization(Xc, cam_stats, method=seq_cfg.get('normalization_method', 'zscore'))
    if sens_stats is not None:
        Xs = apply_normalization(Xs, sens_stats, method=seq_cfg.get('normalization_method', 'zscore'))

    inputs = {
        'xgboost': (X_xgb, y_xgb),
        'sequence': (X_seq, y_seq),
        'multimodal': ((Xc, Xs, Xt), y)
    }

    comp = ModelComparator(models, inputs)

    # Optional significance testing between model pairs
    if getattr(args, 'significance_pairs', None):
        sig_results = {}
        for pair in args.significance_pairs:
            try:
                if ':' in pair:
                    a, b = pair.split(':', 1)
                elif ',' in pair:
                    a, b = pair.split(',', 1)
                else:
                    raise ValueError('Invalid pair format')
                a = a.strip()
                b = b.strip()
                res = comp.compare_significance(a, b)
                sig_results[f'{a}_vs_{b}'] = res
            except Exception as exc:
                LOGGER.warning('Significance test failed for pair %s: %s', pair, exc)
                sig_results[f'{pair}'] = {'error': str(exc)}
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, 'significance_tests.json'), 'w') as fh:
                json.dump(sig_results, fh)
            print('Significance tests written to', os.path.join(args.output_dir, 'significance_tests.json'))
        except Exception as exc:
            LOGGER.warning('Failed to write significance test results: %s', exc)
    # Optional feature importance comparison between two models
    if getattr(args, 'feature_importance_pair', None):
        fi_errors = {}
        for pair in args.feature_importance_pair:
            try:
                if ':' in pair:
                    a, b = pair.split(':', 1)
                elif ',' in pair:
                    a, b = pair.split(',', 1)
                else:
                    raise ValueError('Invalid pair format')
                a = a.strip()
                b = b.strip()
                # Use previously computed tabular feature names when available
                feature_names = feature_cols if 'feature_cols' in locals() else None
                try:
                    fi_df = comp.feature_importance_comparison(a, b, feature_names=feature_names)
                    out_csv = os.path.join(args.output_dir, f'feature_importance_{a}_vs_{b}.csv')
                    fi_df.to_csv(out_csv, index=False)
                    print('Feature importance written to', out_csv)
                except Exception as exc:
                    LOGGER.warning('Feature importance comparison failed for %s vs %s: %s', a, b, exc)
                    fi_errors[f'{a}_vs_{b}'] = str(exc)
            except Exception as exc:
                LOGGER.warning('Feature importance pair parse failed for "%s": %s', pair, exc)
                fi_errors[pair] = str(exc)
        if fi_errors:
            try:
                with open(os.path.join(args.output_dir, 'feature_importance_errors.json'), 'w') as fh:
                    json.dump(fi_errors, fh)
            except Exception:
                LOGGER.warning('Failed to write feature importance errors file')
    # Optional edge benchmark
    if getattr(args, 'benchmark_edge', False):
        try:
            bench_df = comp.benchmark_edge_deployment()
            bench_csv = os.path.join(args.output_dir, 'edge_benchmark.csv')
            bench_df.to_csv(bench_csv, index=False)
            print('Edge benchmark saved to', bench_csv)
        except Exception as exc:
            LOGGER.warning('Edge benchmarking failed: %s', exc)

    # Optional ensemble evaluation
    if args.ensemble:
        try:
            # Ensemble predictions: let each model use its own appropriate input.
            ens_preds = comp.ensemble_predict(args.ensemble)
            # Use the first model in the ensemble as the reference for labels
            first_name = args.ensemble[0]
            m0 = next((m for m in models if m.get('name') == first_name), None)
            if m0 is None:
                raise RuntimeError(f'Model {first_name} not found for ensemble label reference')
            y_ref, _ = comp._get_preds_for_model(m0)
            if y_ref is None:
                raise RuntimeError('Reference labels for ensemble not available')
            from models.evaluation import compute_metrics
            metrics = compute_metrics(y_ref, ens_preds)
            with open(os.path.join(args.output_dir, 'ensemble_metrics.json'), 'w') as fh:
                json.dump(metrics, fh)
            print('Ensemble metrics written to', os.path.join(args.output_dir, 'ensemble_metrics.json'))
        except Exception as exc:
            LOGGER.warning('Ensemble evaluation failed: %s', exc)
        

    out = comp.export_report(args.output_dir)
    print('Comparison exported to', out)
    if getattr(args, 'analyze_failures', False):
        try:
            failures_dir = os.path.join(args.output_dir, 'failures')
            fp = comp.save_failure_cases(failures_dir)
            LOGGER.info('Failure analysis saved to %s', fp)
            print('Failure analysis saved to', fp)
        except Exception as exc:
            LOGGER.warning('Failure analysis failed: %s', exc)

if __name__ == '__main__':
    main()
