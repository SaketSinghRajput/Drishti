# XGBoost Training

This document describes the XGBoost-based training pipeline: multi-horizon
models, imbalance handling, evaluation metrics, SHAP explainability, and ONNX
export for edge deployment.

See `scripts/train_model.py` for a CLI wrapper and
`models/train_xgboost.py` for the trainer implementation.

Quick start:

```bash
python scripts/train_model.py --data-dir data/labeled --horizon 10 --use-smote
```

Configuration is located in `config/default_config.yaml` under the `training`
section (we added a `training` section with XGBoost hyperparameters, imbalance
strategy, SHAP and ONNX options). You can also review `config/training_defaults.yaml`.

Key concepts:
- Use `GroupKFold` split by `trip_id` to avoid leakage across trips.
- Prefer PR-AUC (average precision) for imbalanced datasets.
- Options for imbalance: sample weights, SMOTE, or focal loss.
- Export to ONNX for edge inference.

More details are available in the docstrings of `models.train_xgboost` and the
evaluation utilities in `models.evaluation`.
