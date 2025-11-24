"""Battery ML inference stub for anomaly detection and RUL prediction.

This module provides a small wrapper that attempts to load a model via
joblib and otherwise falls back to a deterministic dummy heuristic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import os
from battery_monitor.ml.feature_extractor import FEATURE_KEYS

logger = logging.getLogger(__name__)


class BatteryMLInference:
    def __init__(self, config):
        self.config = config
        self.model_path = self.config.get("battery_monitor.ml.model_path", "models/battery_anomaly.pkl")
        self.anomaly_threshold = float(self.config.get("battery_monitor.ml.anomaly_threshold", 0.7))
        self._model = None
        self._loaded = False
        try:
            self.load_model()
        except Exception:
            logger.exception("Model load failed; running in dummy mode")

    def load_model(self) -> None:
        try:
            import joblib

            if os.path.exists(self.model_path):
                self._model = joblib.load(self.model_path)
                # Validate model input dimensionality if available
                try:
                    expected = getattr(self._model, 'n_features_in_', None)
                    if expected is not None and expected != len(FEATURE_KEYS):
                        logger.error('Model at %s has incompatible input dimension (%s) != expected (%s). Rejecting model; please retrain.', self.model_path, expected, len(FEATURE_KEYS))
                        self._model = None
                        self._loaded = False
                        return
                except Exception:
                    pass
                self._loaded = True
                logger.info("Loaded battery model from %s", self.model_path)
            else:
                self._loaded = False
                logger.warning("Model path not found: %s; using dummy inference", self.model_path)
        except Exception:
            self._loaded = False
            logger.exception("Failed to load battery model")

    def is_model_loaded(self) -> bool:
        return self._loaded

    def predict_anomaly(self, features: Dict[str, float]) -> Dict[str, Any]:
        # If model loaded, try to use predict_proba or decision_function
        try:
            if self._loaded and hasattr(self._model, "predict_proba"):
                # Use deterministic FEATURE_KEYS ordering for model input
                vec = [float(features.get(k, 0.0)) for k in FEATURE_KEYS]
                proba = float(self._model.predict_proba([vec])[0][1])
                score = proba
            elif self._loaded and hasattr(self._model, "decision_function"):
                # map decision_function to 0-1 via sigmoid
                vec = [float(features.get(k, 0.0)) for k in FEATURE_KEYS]
                val = float(self._model.decision_function([vec])[0])
                import math

                score = 1.0 / (1.0 + math.exp(-val))
            else:
                # dummy heuristic: normalized voltage imbalance and temp
                iv = features.get("voltage_imbalance", 0.0)
                tg = features.get("temp_gradient", 0.0)
                score = min(1.0, (abs(iv) * 2.0 + abs(tg) / 10.0) / 2.0)

            is_anom = score > self.anomaly_threshold
            return {"anomaly_score": float(score), "is_anomaly": bool(is_anom)}
        except Exception:
            logger.exception("Anomaly prediction failed")
            return {"anomaly_score": 0.0, "is_anomaly": False}

    def predict_rul(self, features: Dict[str, float]) -> Optional[float]:
        # Dummy RUL estimator: if soh_mean < 80, estimate hours linear; otherwise None
        try:
            soh = features.get("soh_mean")
            if soh is None:
                return None
            soh = float(soh)
            if soh < 80.0:
                return max(0.0, soh * 10.0)
            return None
        except Exception:
            logger.exception("RUL prediction failed")
            return None
