"""CLI to evaluate a trained model on labeled windows."""
import argparse
import os
import logging

from config.config_loader import ConfigLoader
from models import generate_evaluation_report
from models.train_xgboost import XGBoostTrainer
from feature_engineering.extractor import FeatureExtractor

try:
    import pandas as pd
except Exception:
    pd = None

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/default_config.yaml')
    p.add_argument('--model-path', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', default='evaluation_results')
    p.add_argument('--horizon', type=int, choices=[1, 5, 10], help='Prediction horizon')
    p.add_argument('--thresholds', type=str, help='Comma-separated thresholds, e.g. "0.1,0.3,0.6"')
    p.add_argument('--save-predictions', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = ConfigLoader(args.config)
    trainer = XGBoostTrainer(cfg, horizon=args.horizon)
    trainer.load_model(args.model_path)

    # Load data and extract features
    windows = []
    from utils.dataset_utils import load_labeled_dataset
    windows = load_labeled_dataset(args.data_dir)
    fe = FeatureExtractor(cfg)
    df = fe.extract_to_dataframe(windows)
    label = f'is_accident_h{args.horizon}' if args.horizon else 'is_accident'
    y = df[label]
    X = df.drop(columns=[label])

    os.makedirs(args.output_dir, exist_ok=True)
    y_pred = trainer.predict_proba(X)
    # Determine thresholds: CLI overrides config
    thresholds = None
    if args.thresholds:
        try:
            thresholds = [float(x) for x in args.thresholds.split(',')]
        except Exception:
            thresholds = None
    if thresholds is None:
        model_cfg = cfg.get('model', {})
        thresholds_cfg = model_cfg.get('risk_thresholds') if isinstance(model_cfg, dict) else None
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

    report = generate_evaluation_report(y.values, y_pred, args.output_dir, model_name=os.path.basename(args.model_path), thresholds=thresholds)
    if args.save_predictions:
        if pd is not None:
            out = df.copy()
            out['pred_proba'] = y_pred
            out.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)


if __name__ == '__main__':
    main()
