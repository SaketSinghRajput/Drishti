# Retraining Pipeline

Overview

This document describes the retraining pipeline components, usage, configuration, and best practices.

Components

- DataValidator: validates incoming aggregated datasets and checks drift, label balance, and temporal coverage.
- ModelRegistry: stores metadata, metrics, and deployment records for trained models.
- ABTestingFramework: compares models on a hold-out test set and produces statistical summaries.
- RegressionDetector: compares metrics and flags regressions.
- ModelDeployer: packages models and deploys to edge devices.
- RetrainingScheduler: schedules and gates retraining runs with approval.
- RetrainingPipeline: orchestrates the end-to-end flow.

Configuration

See `config/default_config.yaml` for complete options under the `retraining` key. Important fields:

- `retraining.validation` - data validation thresholds and minima.
- `retraining.registry` - path to the SQLite registry and versioning options.
- `retraining.ab_testing` - AB testing toggles and thresholds.
- `retraining.deployment` - packaging and deployment options (auto_deploy, package_dir, edge_devices).
- `retraining.scheduler` - scheduling and approval gate settings.

Usage

- CLI: `python -m scripts.run_retraining_pipeline full --dataset-csv data/agg.csv --output-dir data/out`
- Dry-run example: `python examples/retraining_pipeline_example.py`

Testing

Basic tests are available in `tests/test_models/test_retrain_pipeline.py`.

Troubleshooting

- If SciPy is not available, statistical tests fall back to approximate checks. Install SciPy to enable full statistical tests.
- Ensure `training` configuration is present and XGBoost is installed for trainer usage.

Best Practices

- Keep a stable hold-out set for A/B tests and regression detection.
- Use the registry to track deployments and make rollback decisions based on versioned artifacts.
