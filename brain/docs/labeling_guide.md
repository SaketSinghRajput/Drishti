# Labeling Subsystem Guide

This document describes the `labeling` subsystem: detectors, temporal labeling,
CLI usage, configuration, and best practices.

## Overview

- Accident detection: `AccidentDetector` uses IMU deceleration spikes and camera
  overlap to mark collision events.
- Near-miss detection: `NearMissDetector` flags evasive maneuvers (braking/swerving)
  when contextual risk is high.
- Temporal labeling: `TemporalLabeler` marks prior windows within configured
  `prediction_horizons` as positive examples for prediction tasks.

## Configuration

Edit `config/default_config.yaml` under the `labeling:` section. Key fields:

- `decel_threshold_g`: deceleration threshold in g for accidents.
- `collision_distance_m`: camera distance threshold to consider collision.
- `brake_spike_threshold`: brake pressure threshold for near-miss detection.
- `near_miss_min_risk_score`: threshold to promote evasive maneuvers to near-miss.
- `prediction_horizons`: list of integer seconds (e.g. `[1,5,10]`).
- `confidence_threshold`: used by review mode to surface low-confidence labels.

Note: the CLI also supports `--prediction-horizons` to override the config at runtime.

## CLI Usage

Run the labeling CLI:

```
python scripts/label_windows.py --input-dir path/to/windows --output-dir out/labels \
  --config config/default_config.yaml --mode auto --balance --split
```

Options of interest:

- `--mode` : `auto` (default), `manual`, or `review`.
- `--prediction-horizons` : Override horizons, e.g. `--prediction-horizons 1,5,10`.
- `--filter-trip` : Process only windows for a given `trip_id`.
- `--dry-run` : Run detection/labeling without writing outputs.

Review mode:

- `--mode review` loads existing labels from `--output-dir` and presents those
  flagged with `requires_review=True` or with `confidence < confidence_threshold`.
  Use this to iterate on labels without rerunning detection.

## Testing & Examples

- Unit tests for dataset utilities are in `tests/test_utils/test_dataset_utils.py` (SMOTE, weights, export/load, temporal split).
- A small example demonstrating detection and labeling is at `examples/labeling_example.py` (synthetic windows, detection, temporal labeling, balancing, split).

## Labeling Strategies

- Accident criteria: a deceleration spike exceeding `decel_threshold_g` (g units) or a camera-detected object within `collision_distance_m` that is in the vehicle's path. System-reported crashes (`metadata.system_crash`) take precedence.
 - Accident criteria: by default the pipeline uses a stricter rule: BOTH a deceleration
   spike exceeding `decel_threshold_g` AND a camera-detected object within
   `collision_distance_m` in-path are required to mark an accident. A system-reported
   crash (`metadata.system_crash`) still takes precedence. For users who require
   the previous looser behavior (decel-only), set `labeling.allow_decel_only_accidents: true`.
- Near-miss criteria: an evasive maneuver (braking spike `brake_spike_threshold` or rapid steering) combined with contextual risk (close camera object or low visibility/wet/icy surface).
 - Near-miss criteria: an evasive maneuver (braking spike `brake_spike_threshold` or rapid steering) combined with a computed `risk_score` that aggregates weather/context and proximity. The pipeline computes a score from configured `risk_weights` (surface, visibility, precip) and adds a proximity bonus when a close object is detected. A near-miss is reported when `risk_score + proximity >= labeling.near_miss_min_risk_score`.
- Severity mapping: configured via `labeling.severity_thresholds` in `config/default_config.yaml` (minor/moderate/severe thresholds in g).

## Temporal Labeling

- `TemporalLabeler` marks windows that occur before a detected event within one of the configured `prediction_horizons` (e.g. `[1,5,10]` seconds). The labeler finds the nearest upcoming event and applies labels depending on whether the event is an `accident` (or any non-`near_miss` event type) or a `near_miss`.
- Multi-horizon support lets you train models for different lead times from the same labeled dataset.

## Label Schema

Key fields emitted in the labeling pipeline (see `config/schema/label_schema.json`):

- `is_accident`: boolean
- `accident_type`: enum (`collision`, `slip`, `off-road`, ...)
- `is_near_miss`: boolean
- `near_miss_type` / `evasive_action`: (`braking`, `swerving`, ...)
- `time_to_accident_s`: float seconds to the event
- `confidence`: float in [0,1]
- `requires_review`: boolean
- `annotator_id` / `annotation_timestamp`

Validation: run `utils.validation.validate_label` (if available) before exporting labels for training.

## Dataset Balancing

- SMOTE: `utils.dataset_utils.balance_dataset_smote` interprets `sampling_strategy` as the desired minority:majority ratio. Example: `0.5` aims for minority_count ~= 0.5 * majority_count. Synthetic samples are marked with `synthetic_sample=True`.
- Fallback: if `imbalanced-learn` or numeric features are not available, the function falls back to duplicating minority samples until the target is reached.
- Alternative: use `compute_sample_weights` to weight samples inversely by class frequency when training instead of oversampling.

## Best Practices

- Use `temporal_train_test_split` to split data by `trip_id` to avoid leakage.
- Surface low-confidence labels for manual review using `--mode review` and the `confidence_threshold`.
- When experimenting with horizons, use `--prediction-horizons` to test multiple lead times without editing config files.

## Troubleshooting

- No near-misses detected? Lower `brake_spike_threshold` or ensure `camera` entries include `distance_m` and `in_path`.
- SMOTE didn't change counts? Ensure there are numeric features available in the exported JSON for SMOTE, otherwise duplication fallback is used.

## Best Practices

- Use temporal train/val/test splits that group by `trip_id` to avoid leakage.
- Review low-confidence or flagged labels via `--mode review` instead of re-running
  the full pipeline.
- When SMOTE is used for balancing, the sampling strategy is treated as
  minority:majority ratio (e.g. `0.5` aims for minority count ~= 0.5 * majority).

## Troubleshooting

- If no near-misses are detected, check `brake_spike_threshold` and camera
  proximity in windows (camera `distance_m` and `in_path` flags).
- If the CLI appears to ignore prediction horizons, use `--prediction-horizons`
  to override at runtime.
# Labeling Guide

This file documents the automated and manual labeling system. It was added alongside the code in `data_collection/labeling.py` and dataset utilities in `utils/dataset_utils.py`.

(See README for quick start and `examples/labeling_example.py` for a runnable demo.)
