"""Machine learning inference for battery anomaly detection and RUL prediction.

Loads pre-trained models (joblib) to compute anomaly scores and RUL estimates.
Stubs are provided for Phase 1 to define module boundaries.
"""

from .inference_stub import BatteryMLInference
from .feature_extractor import BMSFeatureExtractor

__all__ = ["BatteryMLInference", "BMSFeatureExtractor"]
