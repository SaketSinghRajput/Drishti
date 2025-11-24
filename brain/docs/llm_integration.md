# LLM Integration (Gemini)

This document describes the optional LLM (Gemini) integration used to augment edge risk assessments for complex or low-confidence scenarios.

Overview
- The `LLMIntegration` component is an optional, gated enhancement that calls Gemini only when configured invocation policies detect edge cases (low confidence, sensor uncertainty, margin around thresholds, or close-proximity detections).
- Calls are cached (TTL), rate-limited (per-minute), and anonymized to avoid exposing identifiers.

Configuration
- Add or modify keys under `inference.llm_integration` in `config/default_config.yaml` including:
  - `enabled`: enable or disable the integration
  - `api_key_env`: environment variable name for Gemini API key
  - `model_name`: Gemini model identifier (e.g., `gemini-1.5-flash`)
  - `timeout_s`: request timeout in seconds
  - `multimodal_enabled`: whether to include image bytes in requests
  - `cache_ttl_s`, `cache_max_entries`: caching parameters
  - `rate_limit_per_minute`: per-minute call limit (refill rate = per_minute/60)
  - Invocation gating: `invoke_on_low_confidence`, `low_confidence_threshold`, `invoke_on_uncertainty`, `uncertainty_data_quality_threshold`, `invoke_on_margin`, `margin_around_risk_thresholds`, `nearest_object_threshold_m`

Example snippet to place under the `inference:` section of `config/default_config.yaml`:

```yaml
inference:
  llm_integration:
    enabled: false
    api_key_env: GEMINI_API_KEY
    model_name: gemini-1.5-flash
    timeout_s: 0.5
    multimodal_enabled: false
    cache_ttl_s: 300
    cache_max_entries: 1024
    rate_limit_per_minute: 30
    override_confidence: 0.9
    invoke_on_low_confidence: true
    low_confidence_threshold: 0.6
    invoke_on_uncertainty: true
    uncertainty_data_quality_threshold: 0.6
    invoke_on_margin: true
    margin_around_risk_thresholds: 0.05
    nearest_object_threshold_m: 5.0
```

Privacy and Safety
- Identifiers (vehicle IDs, timestamps, PII) are never included in prompts.
- Camera metadata is summarized into coarse scene descriptions (nearest object distance, lane offset, object types) — raw images are only sent when `multimodal_enabled` is true and at operator discretion.
- Calls are rate-limited and cached to control cost and latency.

Developer Notes
- The implementation lazy-loads `google.generativeai` and creates a `GenerativeModel` instance for the configured `model_name`.
- The prompt requests a structured JSON response including `risk_level`, `llm_score`, `confidence`, `reasoning`, `key_risk_factors`, and `recommended_actions`.
- The DecisionUnit blends or overrides model scores based on LLM confidence and surfaces `llm_reasoning` as a top-level explanation field.
