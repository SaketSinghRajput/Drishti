"""Rule-based alarm engine for battery health monitoring.

Deterministic checks for temperature, voltage imbalance, current spikes,
and SoH degradation. Rules are intentionally simple and configurable via
`config/default_config.yaml`.
"""






from .rule_engine import RuleEngine

__all__ = ["RuleEngine"]
