"""Models package: training, evaluation, and custom losses.

Exports a concise API for training and evaluating XGBoost models.
"""
try:
    from .train_xgboost import XGBoostTrainer
except Exception:
    # Allow importing the package in minimal/test environments where
    # optional dependencies (scikit-learn / xgboost) may be missing.
    XGBoostTrainer = None

from .evaluation import compute_metrics, generate_evaluation_report
from .losses import focal_loss_objective, focal_loss_eval, get_focal_loss_params

__all__ = [
    'XGBoostTrainer',
    'compute_metrics',
    'generate_evaluation_report',
    'focal_loss_objective',
    'focal_loss_eval',
    'get_focal_loss_params',
]
"""Models package for training and evaluation pipelines.

This package provides the XGBoostTrainer, evaluation helpers, and custom
loss functions used by the training CLI and examples.
"""
