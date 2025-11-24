#!/usr/bin/env python
"""Project CLI: train deep-learning models with QAT/pruning and compare to XGBoost.

This CLI uses the project's `ConfigLoader` and trainer classes. It supports:
- model selection: lstm, cnn, multimodal
- flags: --apply-pruning, --apply-qat, --export-tflite, --compare-with-xgboost
- dry-run to validate data/model build without training

Run as a module to ensure package imports resolve, e.g.:
    python -m scripts.train_dl_model --config config/default_config.yaml --model-type multimodal --data-dir data/windows --output-dir out --apply-qat --apply-pruning
"""
import argparse
import os
import logging
import json
from types import SimpleNamespace

from config.config_loader import ConfigLoader
from models.model_comparison import ModelComparator
from models.train_xgboost import XGBoostTrainer
from feature_engineering.extractor import FeatureExtractor

LOGGER = logging.getLogger(__name__)


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace for trainers that use attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def apply_cli_overrides(cfg_loader: ConfigLoader, cfg_ns: SimpleNamespace, args):
    # enable pruning if requested
    if args.apply_pruning:
        try:
            cfg_loader.set('training.model_compression.pruning.enable', True)
        except Exception:
            pass
        try:
            cfg_ns.training.model_compression.pruning.enable = True
        except Exception:
            pass
    # enable QAT if requested
    if args.apply_qat:
        try:
            cfg_loader.set('training.model_compression.quantization.method', 'qat')
            cfg_loader.set('training.model_compression.quantization.enable', True)
        except Exception:
            pass
        try:
            cfg_ns.training.model_compression.quantization.method = 'qat'
            cfg_ns.training.model_compression.quantization.enable = True
        except Exception:
            pass

    # higher-level training flags
    if getattr(args, 'use_smote', False):
        try:
            cfg_loader.set('training.data_augmentation.use_smote', True)
        except Exception:
            pass
        try:
            cfg_ns.training.data_augmentation.use_smote = True
        except Exception:
            pass
    if getattr(args, 'use_weights', False):
        try:
            cfg_loader.set('training.use_sample_weights', True)
        except Exception:
            pass
        try:
            cfg_ns.training.use_sample_weights = True
        except Exception:
            pass
    if getattr(args, 'use_focal_loss', False):
        # apply focal loss flag into deep_learning blocks
        try:
            for k in ('lstm', 'cnn', 'multimodal'):
                cfg_loader.set(f'training.deep_learning.{k}.use_focal_loss', True)
        except Exception:
            pass
        try:
            for k in ('lstm', 'cnn', 'multimodal'):
                setattr(cfg_ns.training.deep_learning, k, getattr(cfg_ns.training.deep_learning, k, SimpleNamespace()))
                setattr(getattr(cfg_ns.training.deep_learning, k), 'use_focal_loss', True)
        except Exception:
            pass
    if getattr(args, 'cv_folds', None) is not None:
        try:
            cfg_loader.set('training.cv_folds', int(args.cv_folds))
        except Exception:
            pass
        try:
            cfg_ns.training.cv_folds = int(args.cv_folds)
        except Exception:
            pass
    if getattr(args, 'tensorboard_log_dir', None):
        try:
            cfg_loader.set('training.tensorboard_log_dir', args.tensorboard_log_dir)
        except Exception:
            pass
        try:
            cfg_ns.training.tensorboard_log_dir = args.tensorboard_log_dir
        except Exception:
            pass
    if getattr(args, 'resume_from_checkpoint', None):
        try:
            cfg_loader.set('training.resume_from_checkpoint', args.resume_from_checkpoint)
        except Exception:
            pass
        try:
            cfg_ns.training.resume_from_checkpoint = args.resume_from_checkpoint
        except Exception:
            pass
    if getattr(args, 'use_weights', False):
        # map to use_class_weight in deep learning blocks
        try:
            for k in ('lstm', 'cnn', 'multimodal'):
                cfg_loader.set(f'training.deep_learning.{k}.use_class_weight', True)
        except Exception:
            pass
        try:
            for k in ('lstm', 'cnn', 'multimodal'):
                setattr(cfg_ns.training.deep_learning, k, getattr(cfg_ns.training.deep_learning, k, SimpleNamespace()))
                setattr(getattr(cfg_ns.training.deep_learning, k), 'use_class_weight', True)
        except Exception:
            pass
    # Ensure the namespace view reflects any cfg_loader changes
    try:
        new_ns = dict_to_namespace(cfg_loader.as_dict())
        # update existing cfg_ns in-place so callers keep the same object reference
        cfg_ns.__dict__.clear()
        cfg_ns.__dict__.update(new_ns.__dict__)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='models/trained')
    parser.add_argument('--model-type', choices=['lstm', 'cnn', 'multimodal'], required=True)
    parser.add_argument('--horizon', type=int, default=None)
    # tri-state flags: None when not provided so trainers consult config
    parser.add_argument('--apply-pruning', action='store_true', default=None,
                        help='Override config: enable pruning for this run')
    parser.add_argument('--apply-qat', action='store_true', default=None,
                        help='Override config: enable QAT for this run')
    parser.add_argument('--export-tflite', choices=['auto','yes','no'], default='auto',
                        help='Emit a TFLite model: yes|no|auto(use config)')
    parser.add_argument('--compare-with-xgboost', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # additional planned flags
    parser.add_argument('--use-smote', action='store_true')
    parser.add_argument('--use-weights', dest='use_weights', action='store_true')
    parser.add_argument('--use-focal-loss', action='store_true')
    parser.add_argument('--cv-folds', type=int, default=None)
    parser.add_argument('--tensorboard-log-dir', type=str, default=None)
    parser.add_argument('--resume-from-checkpoint', type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load canonical config loader
    cfg_loader = ConfigLoader(args.config)
    # also produce a namespace view for trainers that expect attribute access
    cfg_ns = dict_to_namespace(cfg_loader.as_dict())
    # set seeds for reproducibility
    try:
        import random as _random
        import numpy as _np
        _random.seed(int(args.seed))
        _np.random.seed(int(args.seed))
        try:
            import tensorflow as _tf
            _tf.random.set_seed(int(args.seed))
        except Exception:
            pass
    except Exception:
        pass

    # Apply CLI overrides to both representations
    apply_cli_overrides(cfg_loader, cfg_ns, args)

    # Determine horizons to train for: single horizon or iterate over configured horizons
    labeling_cfg = cfg_loader.get('labeling', {})
    # Compute horizons list: prefer explicit --horizon, otherwise use labeling.prediction_horizons
    if args.horizon is not None:
        horizons = [int(args.horizon)]
    else:
        horizons = labeling_cfg.get('prediction_horizons') or cfg_loader.get('training', {}).get('prediction_horizons')
        if horizons is None:
            horizons = [None]
        elif not isinstance(horizons, (list, tuple)):
            horizons = [horizons]

    # map tri-state export flag to trainer argument (None -> defer to trainer/config)
    export_flag = None if args.export_tflite == 'auto' else (True if args.export_tflite == 'yes' else False)

    all_models_list = []
    inputs = {}
    final_report = None
    for h in horizons:
        LOGGER.info('Starting training for model type: %s horizon=%s', args.model_type, str(h))
        # instantiate trainer per-horizon to avoid state carryover
        if args.model_type == 'lstm':
            from models.train_lstm import LSTMTrainer
            trainer = LSTMTrainer(cfg_loader, horizon=h)
        elif args.model_type == 'cnn':
            from models.train_cnn import CNNTrainer
            trainer = CNNTrainer(cfg_loader, horizon=h)
        else:
            from models.train_multimodal import MultimodalTrainer
            trainer = MultimodalTrainer(cfg_loader, horizon=h)

        # create per-horizon output directory
        out_h = os.path.join(out_dir, args.model_type, f'h{h}' if h is not None else 'h_default')
        os.makedirs(out_h, exist_ok=True)

        # load data for this horizon
        LOGGER.info('Loading and preparing data for horizon %s from %s', str(h), args.data_dir)
        data = trainer.load_and_prepare_data(args.data_dir)

        # Dry-run: validate build only
        if args.dry_run:
            LOGGER.info('Dry-run: validating model build for horizon %s', str(h))
            if args.model_type == 'multimodal':
                train_t, val_t, test_t = data
                Xc, Xs, Xt, y = train_t
                trainer.build_model(camera_shape=Xc.shape[1:], sensor_shape=Xs.shape[1:], tabular_shape=(Xt.shape[1],))
            else:
                X_train, y_train, X_val, y_val, X_test, y_test = data
                trainer.build_model(input_shape=X_train.shape[1:])
            LOGGER.info('Dry-run successful for horizon %s', str(h))
            continue

        # Train/evaluate and save artifacts into per-horizon directory
        if args.model_type in ('lstm', 'cnn'):
            X_train, y_train, X_val, y_val, X_test, y_test = data
            trainer.build_model(input_shape=X_train.shape[1:])
            # write model summary to per-horizon directory
            try:
                if getattr(trainer, 'model', None) is not None:
                    smf = os.path.join(out_h, 'model_summary.txt')
                    with open(smf, 'w') as sf:
                        try:
                            trainer.model.summary(print_fn=lambda s: sf.write(s + "\n"))
                        except Exception:
                            sf.write('Model summary unavailable')
            except Exception:
                LOGGER.debug('Failed to write model summary')
            trainer.train(X_train, y_train, X_val, y_val, output_dir=out_h,
                          apply_pruning=args.apply_pruning, apply_qat=args.apply_qat, export_tflite=export_flag)
            report = trainer.evaluate(X_test, y_test, output_dir=out_h)
            # persist per-horizon metrics.json
            try:
                with open(os.path.join(out_h, 'metrics.json'), 'w') as mh:
                    json.dump(report or {}, mh)
            except Exception:
                LOGGER.warning('Failed to write per-horizon metrics.json for %s h=%s', args.model_type, h)
            # prepare per-horizon sequence inputs for comparator
            seq_key = f'sequence_h{h if h is not None else "default"}'
            inputs[seq_key] = (X_test, y_test)
            # record model entry; prefer trainer TFLite naming convention lstm_h{h}.tflite / cnn_h{h}.tflite
            tflite_name = f'{args.model_type}_h{h if h is not None else "default"}.tflite'
            tflite_path = os.path.join(out_h, tflite_name)
            model_entry = {'name': f'{args.model_type}_h{h}', 'type': args.model_type, 'predict_fn': trainer.predict_proba, 'horizon': h}
            if os.path.exists(tflite_path):
                model_entry['path'] = tflite_path
            # also attach canonical Keras artifact if present
            keras_path = os.path.join(out_h, 'model.h5')
            if os.path.exists(keras_path):
                model_entry['keras_path'] = keras_path
            all_models_list.append(model_entry)
        else:
            train_t, val_t, test_t = data
            trainer.build_model(camera_shape=train_t[0].shape[1:], sensor_shape=train_t[1].shape[1:], tabular_shape=(train_t[2].shape[1],))
            # write model summary to per-horizon directory
            try:
                if getattr(trainer, 'model', None) is not None:
                    smf = os.path.join(out_h, 'model_summary.txt')
                    with open(smf, 'w') as sf:
                        try:
                            trainer.model.summary(print_fn=lambda s: sf.write(s + "\n"))
                        except Exception:
                            sf.write('Model summary unavailable')
            except Exception:
                LOGGER.debug('Failed to write model summary')
            trainer.train(train_t, val_tuple=val_t, output_dir=out_h,
                          apply_pruning=args.apply_pruning, apply_qat=args.apply_qat, export_tflite=export_flag)
            report = trainer.evaluate(test_t, output_dir=out_h)
            # canonical per-horizon multimodal input key
            mm_key = f'multimodal_h{h if h is not None else "default"}'
            inputs[mm_key] = ((test_t[0], test_t[1], test_t[2]), test_t[3])
            tflite_name = f'{args.model_type}_h{h if h is not None else "default"}.tflite'
            tflite_path = os.path.join(out_h, tflite_name)
            model_entry = {'name': f'{args.model_type}_h{h}', 'type': args.model_type, 'predict_fn': trainer.predict_proba, 'horizon': h}
            if os.path.exists(tflite_path):
                model_entry['path'] = tflite_path
            keras_path = os.path.join(out_h, 'model.h5')
            if os.path.exists(keras_path):
                model_entry['keras_path'] = keras_path
            all_models_list.append(model_entry)

        final_report = report

    # After looping horizons, set models_list for comparator
    models_list = all_models_list

    # If only a single horizon was trained, populate generic keys for backwards compatibility
    try:
        if len(horizons) == 1:
            h0 = horizons[0]
            seq_key = f'sequence_h{h0 if h0 is not None else "default"}'
            mm_key = f'multimodal_h{h0 if h0 is not None else "default"}'
            if seq_key in inputs and 'sequence' not in inputs:
                inputs['sequence'] = inputs[seq_key]
            if mm_key in inputs and 'multimodal' not in inputs:
                inputs['multimodal'] = inputs[mm_key]
    except Exception:
        pass

    # Optionally include XGBoost baseline and compare. Align horizon with deep models.
    if args.compare_with_xgboost:
        LOGGER.info('Training or loading XGBoost baseline')
        # Determine which horizon to use for XGBoost: prefer explicit CLI horizon, otherwise use last trained horizon
        xgb_horizon = args.horizon if args.horizon is not None else (horizons[-1] if horizons else None)
        xgb_tr = XGBoostTrainer(cfg_loader, horizon=xgb_horizon)
        try:
            X_train_xgb, X_val_xgb, X_test_xgb, y_train_xgb, y_val_xgb, y_test_xgb, groups, sample_weights = xgb_tr.load_and_prepare_data(args.data_dir)
        except Exception as exc:
            LOGGER.exception('Failed to prepare XGBoost data: %s', exc)
            X_test_xgb, y_test_xgb = None, None
        else:
            model_path = os.path.join(out_dir, f'xgboost_h{xgb_horizon or "all"}.json')
            try:
                if os.path.exists(model_path):
                    xgb_tr.load_model(model_path)
                else:
                    xgb_tr.train(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, groups=groups, sample_weights=sample_weights)
                    xgb_tr.save_model(model_path)
                if X_test_xgb is None:
                    LOGGER.warning('No XGBoost test set available for comparison')
                else:
                    xgb_tr.evaluate(X_test_xgb, y_test_xgb, output_dir=out_dir)
                    # ensure the XGBoost inputs align to the chosen horizon
                    xgb_key = f'xgboost_h{xgb_horizon if xgb_horizon is not None else "all"}'
                    inputs[xgb_key] = (X_test_xgb, y_test_xgb)
                    # also provide a generic key for backwards compatibility so ModelComparator can find xgboost inputs
                    if 'xgboost' not in inputs:
                        inputs['xgboost'] = (X_test_xgb, y_test_xgb)
            except Exception:
                LOGGER.exception('XGBoost baseline training/loading failed')

    if len(all_models_list) > 0:
        comp_out = os.path.join(out_dir, 'comparison')
        # attach xgboost model entry if present; include horizon for correct input selection
        models_list = all_models_list[:]
        # Only attempt to construct an XGBoost entry when the baseline was requested
        # and xgboost inputs are available. This avoids referencing `xgb_tr`
        # when it was never initialized (e.g., --compare-with-xgboost not used).
        if args.compare_with_xgboost:
            # prefer any horizon-specific xgboost key, otherwise look for generic 'xgboost'
            xgb_keys = [k for k in inputs.keys() if k.startswith('xgboost_h')]
            if not xgb_keys and 'xgboost' in inputs:
                xgb_keys = ['xgboost']
            if xgb_keys:
                # use the first matching key
                k = xgb_keys[0]
                # parse horizon from key if possible
                hval = None
                if k.startswith('xgboost_h'):
                    try:
                        hstr = k.split('_h', 1)[1]
                        hval = None if hstr in ('all', 'default') else int(hstr)
                    except Exception:
                        hval = None
                # Only reference xgb_tr if it exists in locals (was successfully created above)
                if 'xgb_tr' in locals():
                    try:
                        xgb_entry = {'name': f'xgboost_h{hval}', 'type': 'xgboost', 'predict_fn': xgb_tr.predict_proba, 'path': os.path.join(out_dir, f'xgboost_h{hval or "all"}.json'), 'horizon': hval}
                        models_list.append(xgb_entry)
                    except Exception as exc:
                        # Narrow exception handling: log and continue without swallowing other errors
                        LOGGER.warning('Failed to attach XGBoost entry to comparator: %s', exc)
                else:
                    LOGGER.warning('XGBoost comparison was requested but XGBoost trainer was not initialized; skipping XGBoost entry')
        comp = ModelComparator(models=models_list, inputs=inputs)
        comp.export_report(comp_out)
        LOGGER.info('Comparison report saved to %s', comp_out)

    try:
        with open(os.path.join(out_dir, 'evaluation_report.json'), 'w') as fh:
            json.dump(final_report or {}, fh)
    except Exception:
        LOGGER.warning('Failed to write evaluation report')

    LOGGER.info('Training complete. Outputs in %s', out_dir)


if __name__ == '__main__':
    main()
