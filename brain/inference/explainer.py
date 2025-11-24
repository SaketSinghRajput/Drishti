"""Runtime explainability using SHAP for XGBoost models.

Provides a lightweight Explainer that lazily initializes shap.TreeExplainer
for XGBoost boosters, computes single-sample SHAP values, extracts top
contributors and generates human-readable messages.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional

import numpy as np

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, config: ConfigLoader, model: Optional[object] = None):
        self.config = config
        self._model = model
        self._explainer = None
        self._shap = None
        self._top_n = int(self.config.get("inference.explainer_top_n_features", 5))
        # mapping for human readable names
        self._name_map = self.config.get("inference.explainer_feature_name_map", {}) or {}

    def set_model(self, model: object) -> None:
        """Set or replace underlying model (e.g., after load). Clears cached TreeExplainer."""
        self._model = model
        self._explainer = None

    def _ensure_shap(self) -> bool:
        if self._explainer is not None:
            return True
        if self._model is None:
            return False
        # Only support XGBoost boosters for now
        try:
            import shap
        except Exception as e:
            logger.warning("shap not available: %s", e)
            return False
        try:
            # access booster for XGBoostRiskModel patterns
            # Prefer `_booster`, but fall back to other common attributes to support different wrappers.
            booster = getattr(self._model, "_booster", None)
            if booster is None:
                booster = getattr(self._model, "booster", None)
            if booster is None and hasattr(self._model, "get_booster"):
                try:
                    booster = self._model.get_booster()
                except Exception:
                    booster = None
            if booster is None:
                logger.debug("No booster found on model for SHAP Explainer")
                return False
            # Create TreeExplainer
            self._shap = shap
            self._explainer = shap.TreeExplainer(booster)
            return True
        except Exception as e:
            logger.exception("Failed to create SHAP TreeExplainer: %s", e)
            self._explainer = None
            return False

    def explain(self, features: Dict[str, float], risk_score: float) -> Dict[str, Any]:
        """Compute SHAP explanation for a single sample and produce a message.

        Returns dict with keys: message, top_features, shap_values (optional), explainability_available
        """
        if not bool(self.config.get("inference.explainability_enabled", True)):
            return {}

        ok = self._ensure_shap()
        if not ok:
            return {"explainability_available": False, "message": "Explainability not available for this model type"}

        start = time.time()
        try:
            feature_names = []
            if hasattr(self._model, "get_feature_names"):
                feature_names = list(self._model.get_feature_names() or [])
            # Align features to model feature order
            if feature_names:
                arr = np.array([[float(features.get(n, np.nan)) for n in feature_names]], dtype=np.float32)
            else:
                # fallback: use insertion order
                keys = list(features.keys())
                arr = np.array([[float(features.get(k, np.nan)) for k in keys]], dtype=np.float32)

            # shap values: TreeExplainer returns array with shape (n_features, ) or similar
            shap_vals = self._explainer.shap_values(arr)
            # shap_vals may be list for multiclass; handle numpy array case
            if isinstance(shap_vals, list):
                shap_arr = np.asarray(shap_vals)[0]
            else:
                shap_arr = np.asarray(shap_vals).reshape(-1)

            # Build top features
            if feature_names:
                names = feature_names
            else:
                names = list(features.keys())

            abs_vals = np.abs(shap_arr)
            idxs = np.argsort(-abs_vals)[: self._top_n]
            top = []
            for i in idxs:
                try:
                    name = names[i]
                    sv = float(shap_arr[i])
                    fv = float(features.get(name, np.nan))
                except Exception:
                    continue
                top.append({"name": name, "shap_value": sv, "feature_value": fv, "direction": "↑" if sv > 0 else "↓"})

            # Humanize names
            for t in top:
                t["human_name"] = self._name_map.get(t["name"], t["name"])

            # Generate message prefix based on risk_score
            level = "High" if risk_score >= float(self.config.get("model.risk_thresholds.high", 0.6)) else (
                "Medium" if risk_score >= float(self.config.get("model.risk_thresholds.medium", 0.3)) else "Low"
            )
            parts = []
            for t in top:
                parts.append(f"{t['human_name']} {t['direction']} ({t['shap_value']:.2f})")

            message = f"{level} risk: " + ", ".join(parts) if parts else f"{level} risk"

            elapsed_ms = (time.time() - start) * 1000.0
            logger.debug("Explainer computed in %.2fms", elapsed_ms)

            return {
                "explainability_available": True,
                "message": message,
                "top_features": top,
                "shap_values": shap_arr.tolist(),
                "elapsed_ms": elapsed_ms,
            }
        except Exception as e:
            logger.exception("Explainer failed: %s", e)
            return {"explainability_available": False, "message": "Explainability computation failed"}
