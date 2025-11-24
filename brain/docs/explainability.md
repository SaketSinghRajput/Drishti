## Explainability System

This document describes the runtime explainability features added to the Edge Inference engine.

See `inference/explainer.py`, `inference/alert_manager.py`, and `inference/explanation_logger.py`.

Key points:
- Uses SHAP TreeExplainer for XGBoost models to compute per-prediction feature contributions.
- Generates short human-readable messages with top contributing features.
- Two-stage alert manager issues visual/audio alerts then escalates to haptic when persistent.
- Explanation logging writes JSON Lines to `data/explanations/` with configurable retention.

Refer to `examples/explainability_example.py` for a quick demo.

LLM Integration:
- When enabled, the LLM integration can augment model explanations for complex cases. The decision output will include an `llm_reasoning` field containing the LLM's concise natural-language rationale. See `docs/llm_integration.md` for configuration and privacy details.
