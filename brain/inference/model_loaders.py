"""Unified model loaders for XGBoost, TFLite, and ONNX risk models.

Provides BaseRiskModel and concrete implementations with lazy imports and defensive handling.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseRiskModel:
    """Abstract base for risk models.

    Subclasses should implement load(), predict(), predict_batch(), get_feature_names(), get_model_metadata().
    """

    def __init__(self, path: str, model_type: Optional[str] = None):
        self.path = path
        self.model_type = model_type
        self._loaded = False

    def load(self):
        raise NotImplementedError()

    def predict(self, features: Dict[str, float]) -> float:
        raise NotImplementedError()

    def predict_batch(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        raise NotImplementedError()

    def get_feature_names(self) -> List[str]:
        raise NotImplementedError()

    def get_model_metadata(self) -> Dict:
        return {"path": self.path, "type": self.model_type}


class XGBoostRiskModel(BaseRiskModel):
    def __init__(self, path: str, feature_names: Optional[List[str]] = None):
        super().__init__(path, model_type="xgboost")
        self._booster = None
        self._feature_names = feature_names

    def load(self):
        try:
            import xgboost as xgb
        except Exception as e:
            logger.error("xgboost is not available: %s", e)
            raise

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"XGBoost model not found: {self.path}")

        t0 = time.time()
        try:
            # Try native Booster load
            self._booster = xgb.Booster()
            self._booster.load_model(self.path)
        except Exception:
            # Try loading as model file for Scikit API
            try:
                self._booster = xgb.Booster()
                self._booster.load_model(self.path)
            except Exception as e:
                logger.exception("Failed to load XGBoost model: %s", e)
                raise
        self._loaded = True
        logger.info("Loaded XGBoost model from %s in %.3fs", self.path, time.time() - t0)

    def _align_features(self, features: Dict[str, float]) -> np.ndarray:
        if self._feature_names is None:
            # Best-effort: sort keys for deterministic order
            keys = sorted(features.keys())
        else:
            keys = self._feature_names
        vals = []
        for k in keys:
            v = features.get(k)
            if v is None:
                # Use sentinel from config when available; default to NaN
                v = np.nan
            vals.append(float(v))
        return np.array(vals, dtype=np.float32).reshape(1, -1)

    def predict(self, features: Dict[str, float]) -> float:
        if not self._loaded:
            self.load()
        arr = self._align_features(features)
        try:
            dmat = None
            import xgboost as xgb

            dmat = xgb.DMatrix(arr, feature_names=self._feature_names)
            pred = self._booster.predict(dmat)
            # For binary:logistic returns prob for positive class
            if hasattr(pred, "__len__"):
                return float(pred[0])
            return float(pred)
        except Exception as e:
            logger.exception("XGBoost prediction failed: %s", e)
            raise

    def predict_batch(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        if not self._loaded:
            self.load()
        arrs = [self._align_features(f) for f in features_list]
        X = np.vstack(arrs)
        try:
            import xgboost as xgb

            dmat = xgb.DMatrix(X, feature_names=self._feature_names)
            pred = self._booster.predict(dmat)
            return np.asarray(pred, dtype=np.float32)
        except Exception as e:
            logger.exception("XGBoost batch prediction failed: %s", e)
            raise

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names) if self._feature_names is not None else []


class TFLiteRiskModel(BaseRiskModel):
    def __init__(
        self,
        path: str,
        input_shape: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        use_edgetpu: bool = False,
        edgetpu_lib: Optional[str] = None,
    ):
        super().__init__(path, model_type="tflite")
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._input_shape = input_shape
        self._feature_names = feature_names
        self.use_edgetpu = use_edgetpu
        self.edgetpu_lib = edgetpu_lib or "libedgetpu.so.1"

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"TFLite model not found: {self.path}")

        t0 = time.time()

        # Attempt to load EdgeTPU delegate if requested; prefer tflite_runtime but
        # fall back to TensorFlow's delegate loader if available.
        delegate = None
        use_tflite_rt = False
        if self.use_edgetpu:
            try:
                from tflite_runtime.interpreter import load_delegate  # type: ignore

                delegate = load_delegate(self.edgetpu_lib)
                use_tflite_rt = True
                logger.info("Loaded EdgeTPU delegate via tflite_runtime: %s", self.edgetpu_lib)
            except Exception:
                try:
                    import tensorflow as tf

                    # TF exposes experimental.load_delegate
                    delegate = tf.lite.experimental.load_delegate(self.edgetpu_lib)
                    use_tflite_rt = False
                    logger.info("Loaded EdgeTPU delegate via tensorflow: %s", self.edgetpu_lib)
                except Exception as ex:
                    logger.warning("EdgeTPU delegate not available, falling back to CPU: %s", ex)
                    delegate = None

        try:
            # Prefer tflite_runtime if available for smaller footprint
            try:
                import tflite_runtime.interpreter as tflite_rt  # type: ignore

                if delegate is not None:
                    self._interpreter = tflite_rt.Interpreter(model_path=self.path, experimental_delegates=[delegate])
                else:
                    self._interpreter = tflite_rt.Interpreter(model_path=self.path)
            except Exception:
                # Fall back to TensorFlow's lite interpreter
                import tensorflow as tf

                if delegate is not None:
                    self._interpreter = tf.lite.Interpreter(model_path=self.path, experimental_delegates=[delegate])
                else:
                    self._interpreter = tf.lite.Interpreter(model_path=self.path)

            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
        except Exception as e:
            logger.exception("Failed to load TFLite interpreter: %s", e)
            raise

        self._loaded = True
        logger.info("Loaded TFLite model from %s in %.3fs (edgetpu=%s)", self.path, time.time() - t0, bool(delegate))

    def _prepare_input(self, arr: np.ndarray) -> None:
        if self._input_details is None:
            raise RuntimeError("Interpreter not initialized")
        # Assume single input
        idx = self._input_details[0]["index"]
        # Handle quantized models
        if self._input_details[0].get("dtype") in (np.int8, np.uint8):
            scale, zp = self._input_details[0].get("quantization", (1.0, 0))
            if scale is None or (hasattr(scale, "__len__") and len(scale) == 0):
                scale = 1.0
            if zp is None or (hasattr(zp, "__len__") and len(zp) == 0):
                zp = 0
            # Convert float -> quantized
            q = np.round(arr / scale + zp).astype(self._input_details[0]["dtype"])
            self._interpreter.set_tensor(idx, q)
        else:
            self._interpreter.set_tensor(idx, arr.astype(self._input_details[0]["dtype"]))

    def _align_features(self, features: Dict[str, float]) -> np.ndarray:
        if self._feature_names is None:
            vals = list(features.values())
        else:
            vals = []
            for k in self._feature_names:
                v = features.get(k)
                if v is None:
                    v = np.nan
                vals.append(float(v))
        return np.array(vals, dtype=np.float32).reshape(1, -1)

    def predict(self, features: Dict[str, float]) -> float:
        if not self._loaded:
            self.load()
        arr = self._align_features(features)
        try:
            self._prepare_input(arr)
            self._interpreter.invoke()
            out_idx = self._output_details[0]["index"]
            out = self._interpreter.get_tensor(out_idx)
            val = float(out.reshape(-1)[0])
            return val
        except Exception as e:
            logger.exception("TFLite prediction failed: %s", e)
            raise

    def predict_batch(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        if not self._loaded:
            self.load()
        arr = np.vstack([self._align_features(f) for f in features_list])
        try:
            self._prepare_input(arr)
            self._interpreter.invoke()
            out_idx = self._output_details[0]["index"]
            out = self._interpreter.get_tensor(out_idx)
            return np.asarray(out).reshape(-1)
        except Exception as e:
            logger.exception("TFLite batch prediction failed: %s", e)
            raise

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names) if self._feature_names is not None else []


class ONNXRiskModel(BaseRiskModel):
    def __init__(self, path: str, providers: Optional[List[str]] = None, feature_names: Optional[List[str]] = None):
        super().__init__(path, model_type="onnx")
        self._session = None
        self._providers = providers
        self._feature_names = feature_names

    def load(self):
        try:
            import onnxruntime as ort
        except Exception as e:
            logger.error("onnxruntime not available: %s", e)
            raise

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"ONNX model not found: {self.path}")

        t0 = time.time()
        try:
            sess_opts = None
            if self._providers:
                self._session = ort.InferenceSession(self.path, providers=self._providers)
            else:
                self._session = ort.InferenceSession(self.path)
        except Exception as e:
            logger.exception("Failed to create ONNX InferenceSession: %s", e)
            raise
        self._loaded = True
        logger.info("Loaded ONNX model from %s in %.3fs", self.path, time.time() - t0)

    def predict(self, features: Dict[str, float]) -> float:
        if not self._loaded:
            self.load()
        # Prepare input by matching session input names
        try:
            inputs = self._session.get_inputs()
            if len(inputs) == 0:
                raise RuntimeError("ONNX session has no inputs")
            name = inputs[0].name
            # align features deterministically if feature names known
            if self._feature_names is None:
                arr = np.array([list(features.values())], dtype=np.float32)
            else:
                vals = []
                for k in self._feature_names:
                    v = features.get(k)
                    if v is None:
                        v = np.nan
                    vals.append(float(v))
                arr = np.array([vals], dtype=np.float32)
            res = self._session.run(None, {name: arr})
            out = np.asarray(res[0]).reshape(-1)[0]
            return float(out)
        except Exception as e:
            logger.exception("ONNX prediction failed: %s", e)
            raise

    def predict_batch(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        if not self._loaded:
            self.load()
        try:
            inputs = self._session.get_inputs()
            name = inputs[0].name
            if self._feature_names is None:
                arr = np.asarray([list(f.values()) for f in features_list], dtype=np.float32)
            else:
                rows = []
                for f in features_list:
                    r = []
                    for k in self._feature_names:
                        v = f.get(k)
                        if v is None:
                            v = np.nan
                        r.append(float(v))
                    rows.append(r)
                arr = np.asarray(rows, dtype=np.float32)
            res = self._session.run(None, {name: arr})
            return np.asarray(res[0]).reshape(-1)
        except Exception as e:
            logger.exception("ONNX batch prediction failed: %s", e)
            raise

    def get_feature_names(self) -> List[str]:
        return list(self._feature_names) if self._feature_names is not None else []


def load_risk_model(path: str, model_type: str = "auto", **kwargs) -> BaseRiskModel:
    """Factory to load appropriate risk model based on extension or explicit type."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if model_type == "auto":
        if ext in (".xgb", ".bin"):
            return XGBoostRiskModel(path, **kwargs)
        if ext in (".tflite",):
            return TFLiteRiskModel(path, **kwargs)
        if ext in (".onnx",):
            return ONNXRiskModel(path, **kwargs)
        raise ValueError(f"Unsupported model extension: {ext}")
    else:
        mt = model_type.lower()
        if mt in ("xgboost", "xgb"):
            return XGBoostRiskModel(path, **kwargs)
        if mt in ("tflite",):
            return TFLiteRiskModel(path, **kwargs)
        if mt in ("onnx",):
            return ONNXRiskModel(path, **kwargs)
        raise ValueError(f"Unsupported model_type: {model_type}")
