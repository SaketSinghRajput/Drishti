"""CLI for training XGBoost models.

Usage examples:
  python scripts/train_model.py --data-dir data/labeled --horizon 10 --use-smote
"""
import argparse
import os
import logging

from config.config_loader import ConfigLoader
from models import XGBoostTrainer

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/default_config.yaml')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', default='models/trained')
    p.add_argument('--horizon', type=int, choices=[1, 5, 10], help='Prediction horizon in seconds')
    p.add_argument('--use-smote', action='store_true', default=None, help='Override config to enable SMOTE')
    p.add_argument('--use-weights', action='store_true', default=None, help='Override config to enable sample weights')
    p.add_argument('--use-focal-loss', action='store_true', default=None, help='Override config to enable focal loss')
    p.add_argument('--cv-folds', type=int, default=5)
    p.add_argument('--export-onnx', action='store_true')
    p.add_argument('--explain', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = ConfigLoader(args.config)
    horizons = []
    if args.horizon is None:
        horizons = cfg.get('labeling', {}).get('prediction_horizons', []) or [cfg.get('model', {}).get('prediction_horizon_s', 10)]
    else:
        horizons = [args.horizon]

    summary = {}
    for h in horizons:
        trainer = XGBoostTrainer(cfg, horizon=h)
        X_train, X_val, X_test, y_train, y_val, y_test, groups, sample_weights = trainer.load_and_prepare_data(args.data_dir, use_smote=args.use_smote, use_weights=args.use_weights)
        res = trainer.train(X_train, y_train, X_val=X_val, y_val=y_val, groups=groups, sample_weights=sample_weights, use_focal_loss=args.use_focal_loss, cv_folds=args.cv_folds)
        model_dir = os.path.join(args.output_dir, f'h{h}')
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(os.path.join(model_dir, 'model.json'))
        metrics = trainer.evaluate(X_test, y_test, model_dir)
        if args.explain:
            shap_dir = os.path.join(model_dir, 'shap')
            trainer.explain(X_test, shap_dir, num_samples=cfg.get('training', {}).get('shap', {}).get('num_samples', 100))
            logger.info('SHAP outputs written to %s', shap_dir)
        if args.export_onnx:
            trainer.export_onnx(os.path.join(model_dir, 'model.onnx'), raise_on_error=True)
        summary[f'h{h}'] = metrics

    # save aggregated summary
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as fh:
            import json

            json.dump(summary, fh, indent=2)
    except Exception:
        logger.exception('Failed to write training summary')


if __name__ == '__main__':
    main()
