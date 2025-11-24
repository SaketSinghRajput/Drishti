"""XGBoost training pipeline.

Provides XGBoostTrainer: loads labeled windows, extracts features, performs
GroupKFold CV, handles imbalance strategies, evaluates, explains via SHAP,
and exports models to ONNX.

This is an integrated but defensive implementation: if optional dependencies
are missing the trainer will raise informative errors.
"""
from typing import Optional, Dict, Any, List, Tuple
import os
import json
import logging
import tempfile

import numpy as np

try:
    import xgboost as xgb
except Exception:
    xgb = None

from sklearn.model_selection import GroupKFold, train_test_split

try:
    import pandas as pd
except Exception:
    pd = None

from feature_engineering.extractor import FeatureExtractor
from utils.dataset_utils import load_labeled_dataset, temporal_train_test_split, compute_sample_weights, balance_dataset_smote
from config.config_loader import ConfigLoader
from models.losses import get_focal_loss_params, focal_loss_objective, focal_loss_eval
from models.evaluation import generate_evaluation_report

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    def __init__(self, config: ConfigLoader, horizon: Optional[int] = None):
        self.config = config
        self.horizon = horizon
        self.training_cfg = self.config.get('training', {})
        self.xgb_cfg = self.training_cfg.get('xgboost', {})
        self.model = None
        self.feature_names: List[str] = []
        self.booster: Optional[Any] = None

    def _label_name(self) -> str:
        if self.horizon is None:
            return 'is_accident'
        return f'is_accident_h{self.horizon}'

    def load_and_prepare_data(self, data_dir: str, use_smote: Optional[bool] = None, use_weights: Optional[bool] = None) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        """Load labeled windows, extract features into DataFrames, split into train/val/test.

        Returns: X_train, X_val, X_test, y_train, y_val, y_test, groups_train, sample_weights
        """
        if load_labeled_dataset is None:
            raise RuntimeError('Dataset utilities not available')

        logger.info('Loading labeled windows from %s', data_dir)
        labeled_windows = load_labeled_dataset(data_dir)

        test_size = float(self.training_cfg.get('test_size', 0.2))
        val_size = float(self.training_cfg.get('val_size', 0.1))

        # Split labeled windows by trips using the helper that expects window dicts
        train_windows, val_windows, test_windows = temporal_train_test_split(labeled_windows, test_size=test_size, val_size=val_size)

        # Optionally apply SMOTE at the window level (so trip ids remain consistent)
        strategy = self.training_cfg.get('imbalance_strategy', 'weights')
        # allow CLI args to override config by passing explicit booleans
        if use_smote is None:
            use_smote_effective = (strategy == 'smote' or strategy == 'smote_and_weights')
        else:
            use_smote_effective = use_smote

        if use_smote_effective:
            try:
                train_windows = balance_dataset_smote(train_windows, sampling_strategy=self.training_cfg.get('smote_sampling_strategy', 0.5), k_neighbors=self.training_cfg.get('smote_k_neighbors', 5))
            except Exception as exc:
                logger.warning('SMOTE at window-level failed: %s', exc)

        # Feature extraction per split
        fe = FeatureExtractor(self.config)
        logger.info('Extracting features for train/val/test splits')
        if pd is None:
            raise RuntimeError('pandas is required for training pipeline')

        df_train = fe.extract_to_dataframe(train_windows)
        df_val = fe.extract_to_dataframe(val_windows) if val_windows else pd.DataFrame()
        df_test = fe.extract_to_dataframe(test_windows) if test_windows else pd.DataFrame()

        label_col = self._label_name()
        if label_col not in df_train.columns:
            raise KeyError(f'label column {label_col} not found in training features')

        # derive X/y and groups for train/val/test
        y_train = df_train[label_col].astype(float)
        X_train = df_train.drop(columns=[label_col])
        y_val = df_val[label_col].astype(float) if label_col in df_val.columns else None
        X_val = df_val.drop(columns=[label_col]) if label_col in df_val.columns else None
        y_test = df_test[label_col].astype(float) if label_col in df_test.columns else None
        X_test = df_test.drop(columns=[label_col]) if label_col in df_test.columns else None

        # groups for GroupKFold from train set
        if 'trip_id' in df_train.columns:
            groups_train = df_train['trip_id']
        else:
            groups_train = pd.Series(range(len(X_train)))

        # Decide imbalance handling defaults from config if CLI did not override
        if use_weights is None:
            use_weights_effective = strategy in ('weights', 'smote_and_weights', 'weights_and_focal')
        else:
            use_weights_effective = bool(use_weights)

        # Optionally compute sample weights (do this after any resampling so they match X_train/y_train)
        sample_weights = None
        if use_weights_effective:
            try:
                labels_for_weights = [int(v) for v in y_train.tolist()]
                sample_weights = compute_sample_weights(labels_for_weights)
            except Exception:
                logger.warning('compute_sample_weights failed; continuing without weights')


        self.feature_names = list(X_train.columns)
        return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, sample_weights

    def train(self, X_train, y_train, X_val=None, y_val=None, groups=None, sample_weights=None, use_focal_loss: Optional[bool] = None, cv_folds: Optional[int] = None) -> Dict[str, Any]:
        """Train an XGBoost model. Performs GroupKFold CV if groups provided.

        Returns training metadata including metrics.
        """
        if xgb is None:
            raise RuntimeError('xgboost is required for training')

        params = dict(self.xgb_cfg)
        objective = params.pop('objective', 'binary:logistic')
        eval_metric = params.pop('eval_metric', 'aucpr')

        # Decide effective focal loss from config if caller didn't override
        strategy = self.training_cfg.get('imbalance_strategy', 'weights')
        if use_focal_loss is None:
            use_focal = strategy == 'focal_loss'
        else:
            use_focal = bool(use_focal_loss)

        # If scale_pos_weight not explicitly set, compute from y_train
        if 'scale_pos_weight' not in params or params.get('scale_pos_weight') is None:
            pos = float((y_train == 1).sum())
            neg = float((y_train == 0).sum())
            if pos > 0:
                params['scale_pos_weight'] = float(neg / pos)

        # Validate and convert sample weights to numpy array aligned with X_train
        if sample_weights is not None:
            if len(sample_weights) != len(X_train):
                raise ValueError('sample_weights length does not match X_train rows')
            weights_arr = np.asarray(sample_weights)
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_arr, feature_names=self.feature_names)
        else:
            weights_arr = None
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        evals_result = {}

        if groups is not None:
            if cv_folds is None:
                cv_folds = int(self.training_cfg.get('cv_folds', 5))
            gkf = GroupKFold(n_splits=cv_folds)
            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
                logger.info('Training fold %d/%d', fold + 1, cv_folds)
                X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
                # slice weights if provided
                if sample_weights is not None:
                    w_tr = weights_arr[train_idx]
                    w_va = weights_arr[val_idx]
                    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=self.feature_names)
                    dva = xgb.DMatrix(X_va, label=y_va, weight=w_va, feature_names=self.feature_names)
                else:
                    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=self.feature_names)
                    dva = xgb.DMatrix(X_va, label=y_va, feature_names=self.feature_names)

                # Prepare custom objective if requested
                obj = None
                feval = None
                if use_focal or self.xgb_cfg.get('use_focal_loss', False):
                    flp = get_focal_loss_params(self.config)
                    alpha = flp.get('alpha', 0.25)
                    gamma = flp.get('gamma', 2.0)

                    def obj(preds, dtrain):
                        return focal_loss_objective(preds, dtrain, alpha=alpha, gamma=gamma)

                    def feval(preds, dtrain):
                        return focal_loss_eval(preds, dtrain, alpha=alpha, gamma=gamma)

                bst = xgb.train(params, dtr, num_boost_round=int(self.xgb_cfg.get('n_estimators', 200)), evals=[(dva, 'val')], early_stopping_rounds=int(self.xgb_cfg.get('early_stopping_rounds', 20)), evals_result=evals_result, verbose_eval=False, obj=obj, feval=feval)
                # evaluate
                ypred = bst.predict(dva)
                try:
                    from models.evaluation import compute_metrics
                    metrics = compute_metrics(y_va.values, ypred)
                except Exception:
                    metrics = {}
                fold_metrics.append(metrics)

            # train final on full train set, with optional validation
        logger.info('Training final model on full training set')
        # Use validation set for early stopping if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            if sample_weights is not None:
                # We don't have per-sample weights for val by default; try to slice from sample_weights if same indexing
                # If sample_weights length matches X_train original length, we assume val weights are not available.
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            else:
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'val'))

        # Prepare objective/feval for final training
        obj = None
        feval = None
        if use_focal or self.xgb_cfg.get('use_focal_loss', False):
            flp = get_focal_loss_params(self.config)
            alpha = flp.get('alpha', 0.25)
            gamma = flp.get('gamma', 2.0)

            def obj(preds, dtrain):
                return focal_loss_objective(preds, dtrain, alpha=alpha, gamma=gamma)

            def feval(preds, dtrain):
                return focal_loss_eval(preds, dtrain, alpha=alpha, gamma=gamma)

        bst_full = xgb.train(params, dtrain, num_boost_round=int(self.xgb_cfg.get('n_estimators', 200)), evals=evals, early_stopping_rounds=int(self.xgb_cfg.get('early_stopping_rounds', 20)), evals_result=evals_result, verbose_eval=False, obj=obj, feval=feval)
        self.booster = bst_full
        self.model = bst_full
        return {'status': 'trained', 'feature_names': self.feature_names, 'evals_result': evals_result}

    def predict_proba(self, X) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError('Model is not trained')
        d = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.booster.predict(d)

    def evaluate(self, X_test, y_test, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        y_pred = self.predict_proba(X_test)
        # Fetch thresholds from config (model.risk_thresholds) if present
        model_cfg = self.config.get('model', {})
        thresholds_cfg = model_cfg.get('risk_thresholds') if isinstance(model_cfg, dict) else None
        thresholds = None
        if isinstance(thresholds_cfg, dict):
            # preserve logical order if keys are low/medium/high
            ordered = []
            for k in ('low', 'medium', 'high'):
                if k in thresholds_cfg:
                    ordered.append(float(thresholds_cfg[k]))
            if ordered:
                thresholds = ordered
            else:
                # fallback to values order
                try:
                    thresholds = [float(v) for v in thresholds_cfg.values()]
                except Exception:
                    thresholds = None
        elif isinstance(thresholds_cfg, (list, tuple)):
            thresholds = list(thresholds_cfg)

        report = generate_evaluation_report(y_test.values, y_pred, output_dir, model_name=f'h{self.horizon or "all"}', thresholds=thresholds)
        # Save basic metadata
        with open(os.path.join(output_dir, 'trainer_meta.json'), 'w') as fh:
            json.dump({'horizon': self.horizon, 'feature_count': len(self.feature_names)}, fh)
        return report

    def explain(self, X_sample, output_dir: str, num_samples: int = 100):
        # Check training.shap.enabled in config
        shap_cfg = self.training_cfg.get('shap', {}) if isinstance(self.training_cfg, dict) else {}
        if not shap_cfg.get('enabled', True):
            logger.info('SHAP disabled in config; skipping explainability')
            return

        try:
            import shap
            import matplotlib.pyplot as plt
        except Exception:
            logger.warning('shap or matplotlib not installed; skipping explainability')
            return

        # default num_samples from config if not provided
        if num_samples is None:
            num_samples = int(shap_cfg.get('num_samples', 100))
        os.makedirs(output_dir, exist_ok=True)
        explainer = shap.TreeExplainer(self.booster)
        sample = X_sample.sample(n=min(len(X_sample), num_samples))
        shap_values = explainer.shap_values(sample)
        try:
            shap.summary_plot(shap_values, sample, show=False)
            # persist the current matplotlib figure
            try:
                fig = plt.gcf()
                fig_path = os.path.join(output_dir, 'shap_summary.png')
                fig.savefig(fig_path, dpi=200)
                plt.close(fig)
                logger.info('Saved SHAP summary to %s', fig_path)
            except Exception:
                logger.exception('Failed to save SHAP summary figure')

            # compute mean absolute SHAP values per feature and save CSV
            try:
                import pandas as pd
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                fi = pd.DataFrame({'feature': sample.columns, 'mean_abs_shap': mean_abs})
                fi = fi.sort_values('mean_abs_shap', ascending=False)
                csv_path = os.path.join(output_dir, 'feature_importance.csv')
                fi.to_csv(csv_path, index=False)
                logger.info('Saved SHAP feature importances to %s', csv_path)
            except Exception:
                logger.exception('Failed to save SHAP feature importance CSV')
        except Exception:
            logger.exception('Error creating SHAP summary plot')

    def export_onnx(self, output_path: str, raise_on_error: bool = False):
        # Try native xgboost ONNX export if available
        try:
            if xgb is not None and hasattr(xgb, 'to_onnx'):
                try:
                    onx = xgb.to_onnx(self.booster)
                    with open(output_path, 'wb') as fh:
                        fh.write(onx)
                    logger.info('ONNX model exported via xgboost.to_onnx to %s', output_path)
                    return
                except Exception:
                    logger.info('xgboost native ONNX export not available or failed; falling back')
        except Exception:
            pass

        try:
            import onnxmltools
            from skl2onnx.common.data_types import FloatTensorType
        except Exception as exc:
            logger.warning('onnxmltools or skl2onnx not available; try installing requirements: %s', exc)
            if raise_on_error:
                raise
            return

        # Use onnxmltools to convert xgboost model
        try:
            # Build initial types for conversion based on feature count
            n_features = len(self.feature_names)
            initial_types = [('input', FloatTensorType([None, n_features]))]
            target_opset = int(self.training_cfg.get('onnx_opset_version', 14))
            onx = onnxmltools.convert_xgboost(self.booster, initial_types=initial_types, target_opset=target_opset)
            onnxmltools.utils.save_model(onx, output_path)
            logger.info('ONNX model exported to %s (opset=%s)', output_path, target_opset)
        except Exception as exc:
            logger.exception('ONNX export failed: %s', exc)
            if raise_on_error:
                raise

    def save_model(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.booster is None:
            raise RuntimeError('No model to save')
        self.booster.save_model(output_path)

    def load_model(self, model_path: str):
        if xgb is None:
            raise RuntimeError('xgboost not available')
        b = xgb.Booster()
        b.load_model(model_path)
        self.booster = b
        self.model = b
