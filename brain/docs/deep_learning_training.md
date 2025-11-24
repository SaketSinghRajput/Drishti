# Deep Learning Training

This document describes the deep learning training pipeline used in this
repository: LSTM, 1D-CNN, and multimodal fusion models, together with
model compression (pruning, quantization) and TFLite export for edge
deployment.

Contents
- Overview
- Data preparation
- Model architectures
- Training and distillation
- Compression & TFLite export
- Evaluation & comparison

Overview
--------
The deep learning trainers follow a consistent pattern:
- load and prepare data (train/val/test splits)

Comparability caveat
--------------------
Be aware that different model types may be evaluated on different input
spaces and sample sets (tabular vs sequence vs multimodal). For example,
an XGBoost baseline may be computed on tabular rows while an LSTM is
evaluated on sequence-derived windows; multimodal models may have camera
inputs aligned or zero-padded differently. These differences mean that
metric values across model types are comparable only in a coarse sense —
small numerical differences should be interpreted cautiously.

If you require strictly paired comparisons, ensure all models are evaluated
on the same aligned sample set (e.g., by producing horizon-tagged inputs
and intersecting sample indices across modalities) prior to calling the
comparator.
- compute and persist normalization statistics from the training split
- build a Keras model
- optionally apply pruning or QAT wrappers
- train, validate, and persist model artifacts and normalization stats
- export TFLite artifacts for edge deployment

Data preparation
----------------
Sequence data is prepared using `utils.sequence_utils.prepare_sequences_from_windows()`.
Key behaviors:
- By default the helper produces overlapping sliding subsequences from each
	input window (multiple subsequences per window). This behavior is
	controlled by the `use_sliding` parameter (defaults to `True`) and the
	`seq_stride` parameter.
- To obtain exactly one sequence per input window (recommended for
	multimodal alignment), pass `use_sliding=False`; this pads/truncates the
	series to `seq_length` and returns a single subsequence per window.
- Normalization statistics are computed on the training split and saved as
	`normalization_stats.npz` next to trained models. Comparison tooling loads
	and applies these stats instead of normalizing on-the-fly.

Model architectures
-------------------
The repo contains example LSTM, 1D-CNN, and multimodal fusion architectures.
Fusion modes include `concat`, `attention`, and `weighted` projection fusion.

Training and distillation
-------------------------
Trainers accept configuration under `training.deep_learning.*`. Distillation
is implemented in `utils.model_compression.DistillationTrainer` and performs
combined supervised + KL soft-target training using teacher logits, a
temperature parameter `T`, and the standard scaling of the distillation
term by `1/T^2`.

Compression & TFLite export
---------------------------
- Pruning and QAT wrapping rely on TensorFlow Model Optimization (`tfmot`).
- Post-training quantization uses a representative dataset generator created
	from training/validation samples. When quantization is enabled the converters
	set int8 I/O types for TFLite when `dtype=='int8'`.
- Utility `utils.model_compression.export_tflite_with_metadata()` exports the
	tflite file and writes a small `.metadata.json` alongside it when metadata
	is provided.

Evaluation & comparison
-----------------------
`models.model_comparison.ModelComparator` centralizes model evaluation, metric
comparison, ROC/PR plotting, and edge benchmarking (size/latency). The
`scripts/compare_models.py` CLI loads saved normalization stats and applies
them to sequences before prediction to ensure fair comparisons.

Examples
--------
Train a multimodal model (example):

```powershell
python scripts/train_dl_model.py --config config/default_config.yaml --data-dir data/windows --output-dir out/ --export-tflite yes
```

Note: `--output-dir` controls the root output directory where per-model-type
and per-horizon subdirectories are created (for example: `output-dir/lstm/h10/`).

Compare models (example):

```powershell
python scripts/compare_models.py --config config/default_config.yaml --test-data-dir data/windows --model-paths out/multimodal.tflite out/xgb.model --model-types multimodal xgboost --output-dir out/compare
```

Notes
-----
Further details and architecture diagrams can be added. This doc is a living
reference and is intentionally concise; expand sections with code examples as
needed when developing new trainers or models.
