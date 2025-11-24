"""EdgeInferenceEngine: orchestrates real-time inference on edge devices."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional, Dict, Any, Deque, List

from config.config_loader import ConfigLoader
from feature_engineering.extractor import FeatureExtractor
from inference.model_loaders import load_risk_model, BaseRiskModel
from inference.sliding_window_buffer import SlidingWindowBuffer
from inference.decision_unit import DecisionUnit
from inference.explainer import Explainer
from inference.alert_manager import AlertManager
from inference.explanation_logger import ExplanationLogger
from utils.telemetry import TelemetryCollector
from inference.llm_integration import LLMIntegration

logger = logging.getLogger(__name__)


class EdgeInferenceEngine:
    def __init__(self, config: ConfigLoader, model_path: Optional[str] = None, model_type: Optional[str] = None, vehicle_hash: Optional[str] = None, trip_id: Optional[str] = None, _simulation_mode: bool = False):
        self.config = config
        self.model_path = model_path or self.config.get("inference.model_path")
        self.model_type = model_type or self.config.get("inference.model_type", "auto")
        self.vehicle_hash = vehicle_hash
        self.trip_id = trip_id
        self._simulation_mode = _simulation_mode

        # Lazy-loaded model
        self._model: Optional[BaseRiskModel] = None

        self.buffer = SlidingWindowBuffer(config, vehicle_hash=vehicle_hash, trip_id=trip_id)
        self.extractor = FeatureExtractor(config)
        self.decision_unit = DecisionUnit(config)
        # Explainability and alerting components (lazy-friendly)
        self.explainer = Explainer(config, model=None)
        self.alert_manager = AlertManager(config)
        self.explanation_logger = ExplanationLogger(config) if bool(config.get("inference.explanation_log_enabled", True)) else None
        self.telemetry = TelemetryCollector(config) if bool(config.get("telemetry.enabled", True)) else None

        # Optional LLM integration (Gemini)
        try:
            if bool(self.config.get("inference.llm_integration.enabled", False)):
                self.llm_integration = LLMIntegration(config)
            else:
                self.llm_integration = None
        except Exception:
            logger.exception("Failed to initialize LLMIntegration")
            self.llm_integration = None

        self._stop_event = threading.Event()
        self._inference_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None

        self._state_lock = threading.Lock()
        self._latest_risk: Optional[float] = None
        self._latest_decision: Optional[Dict[str, Any]] = None
        self._last_inference_ts: Optional[float] = None

        self.latency_window: Deque[float] = deque(maxlen=100)
        self.total_inferences = 0
        # uptime tracking
        self._start_ts: Optional[float] = None
        # separate latency windows for extraction and model inference
        self.extract_latency_window: Deque[float] = deque(maxlen=100)
        self.model_latency_window: Deque[float] = deque(maxlen=100)

        self.inference_interval = float(self.config.get("inference.inference_interval_s", 1.0))
        self.latency_budget_ms = float(self.config.get("inference.latency_budget_ms", 200))
        self.latency_warn_th = float(self.config.get("inference.latency_warning_threshold_ms", 180))
        self.enable_monitoring = bool(self.config.get("inference.enable_monitoring", True))

    def _load_model(self):
        if self._model is not None:
            return
        if not self.model_path:
            raise RuntimeError("No model path provided for EdgeInferenceEngine")
        # Pass EdgeTPU flags from config so TFLiteRiskModel can enable delegate
        use_edgetpu = bool(self.config.get("inference.use_edgetpu", False))
        edgetpu_lib = self.config.get("inference.edgetpu_lib", None)
        self._model = load_risk_model(self.model_path, model_type=self.model_type, use_edgetpu=use_edgetpu, edgetpu_lib=edgetpu_lib)
        self._model.load()
        # attach to explainer if present
        try:
            if hasattr(self, "explainer") and self.explainer:
                self.explainer.set_model(self._model)
        except Exception:
            logger.exception("Failed to attach model to explainer")

    def start(self):
        # Load model lazily unless in simulation mode
        if not self._simulation_mode:
            self._load_model()
        self._stop_event.clear()
        self._inference_thread = threading.Thread(target=self._inference_loop, name="inference-loop", daemon=True)
        self._inference_thread.start()
        if self.enable_monitoring:
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, name="inference-monitor", daemon=True)
            self._monitor_thread.start()
        self._start_ts = time.time()
        logger.info("EdgeInferenceEngine started")

    def stop(self):
        self._stop_event.set()
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5.0)
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        # close logger
        try:
            if self.explanation_logger:
                self.explanation_logger.close()
        except Exception:
            logger.exception("Failed to close explanation logger")
        try:
            if self.telemetry:
                self.telemetry.close()
        except Exception:
            logger.exception("Failed to close telemetry collector")
        logger.info("EdgeInferenceEngine stopped")

    def add_sensor_reading(self, sensor_name: str, data: Dict[str, Any]) -> None:
        self.buffer.add_reading(sensor_name, data)

    def _inference_loop(self):
        next_log = 0
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                if not self.buffer.is_ready():
                    time.sleep(0.05)
                    continue

                window = self.buffer.get_current_window()
                sensor_health = window.get("sensor_health_snapshot", {})

                # Feature extraction
                t_ex0 = time.time()
                try:
                    features = self.extractor.extract(window)
                except Exception as e:
                    logger.exception("Feature extraction failed: %s", e)
                    time.sleep(self.inference_interval)
                    continue
                t_ex1 = time.time()

                # Model inference
                try:
                    t_inf0 = time.time()
                    score = float(self._model.predict(features)) if self._model is not None else 0.0
                    t_inf1 = time.time()
                    # record component latencies
                    self.extract_latency_window.append((t_ex1 - t_ex0) * 1000.0)
                    self.model_latency_window.append((t_inf1 - t_inf0) * 1000.0)
                except Exception as e:
                    logger.exception("Model inference failed: %s", e)
                    time.sleep(self.inference_interval)
                    continue

                # Decision
                # derive data quality from window metadata if present
                dq = window.get("metadata", {}).get("data_quality_score", 1.0)

                # Explainability (if enabled)
                explanation = {}
                try:
                    if bool(self.config.get("inference.explainability_enabled", True)) and self._model is not None:
                        # Do NOT reset explainer model every iteration; explainer.set_model
                        # should only be called when the model actually changes (e.g., in _load_model()).
                        explanation = self.explainer.explain(features, float(score)) or {}
                    else:
                        explanation = {}
                except Exception:
                    logger.exception("Explainability failed")
                    explanation = {}

                # Optional LLM enhancement (gated, best-effort)
                llm_output = None
                try:
                    if getattr(self, "llm_integration", None):
                        # Prepare camera metadata and window data for LLM
                        camera_metadata = window.get("camera_metadata")
                        # Pass whole window for complex heuristics
                        window_data = window
                        # Check invocation policy first
                        try:
                            should_call = self.llm_integration._should_invoke(float(score), sensor_health or {}, data_quality=dq, explanation=explanation, window_data=window_data, features=features)
                        except Exception:
                            should_call = False
                        if should_call:
                            llm_output = self.llm_integration.enhance_risk_assessment(features, explanation, sensor_health, float(score), camera_metadata=camera_metadata, window_data=window_data)
                except Exception:
                    logger.exception("LLM enhancement failed")

                decision = self.decision_unit.update(score, sensor_health, data_quality=dq, explanation=explanation, llm_output=llm_output)

                # Alert management
                try:
                    alert_action = self.alert_manager.process_decision(decision)
                except Exception:
                    logger.exception("AlertManager processing failed")
                    alert_action = {}

                # Update shared state and include explanation and alert_action
                with self._state_lock:
                    decision_out = dict(decision)
                    # ensure metadata contains trip/vehicle identifiers for correlation
                    meta = dict(decision_out.get("metadata") or {})
                    # preserve existing keys and add engine-level identifiers
                    if self.vehicle_hash is not None:
                        meta.setdefault("vehicle_hash", self.vehicle_hash)
                    if self.trip_id is not None:
                        meta.setdefault("trip_id", self.trip_id)
                    decision_out["metadata"] = meta
                    decision_out["explanation_message"] = (explanation or {}).get("message")
                    decision_out["explanation"] = explanation or {}
                    # include LLM augmentation if available
                    decision_out["llm_output"] = llm_output
                    # surface LLM natural language reasoning as primary field for downstream consumers
                    llm_reasoning = None
                    if llm_output and isinstance(llm_output, dict):
                        llm_reasoning = llm_output.get("reasoning") or llm_output.get("explanation")
                    decision_out["llm_reasoning"] = llm_reasoning
                    decision_out["alert_action"] = alert_action
                    self._latest_risk = score
                    self._latest_decision = decision_out
                    self._last_inference_ts = time.time()
                    self.total_inferences += 1

                # Latency tracking (ms)
                total_ms = (time.time() - t0) * 1000.0
                self.latency_window.append(total_ms)
                if total_ms > self.latency_budget_ms:
                    logger.warning("Inference exceeded latency budget: %.1fms", total_ms)
                elif total_ms > self.latency_warn_th:
                    logger.info("Inference approaching latency budget: %.1fms", total_ms)

                if self.total_inferences % 10 == 0 and time.time() >= next_log:
                    logger.info("Inference summary: total=%d, last_latency_ms=%.1f", self.total_inferences, total_ms)
                    next_log = time.time() + 10

                # Explanation logging (best-effort)
                try:
                    if self.explanation_logger:
                        self.explanation_logger.log_prediction(decision_out, explanation or {}, features, sensor_health)
                except Exception:
                    logger.exception("Explanation logging failed")
                # Telemetry logging (best-effort)
                try:
                    if self.telemetry:
                        # telemetry expects risk_score, alert_level, features, sensor_health, metadata
                        # include llm_reasoning in metadata for downstream telemetry systems
                        meta_with_llm = dict(meta)
                        if decision_out.get("llm_reasoning"):
                            meta_with_llm["llm_reasoning"] = decision_out.get("llm_reasoning")
                        self.telemetry.log_prediction(risk_score=score, alert_level=str(decision.get("alert_level", "unknown")), features=features, sensor_health=sensor_health, metadata=meta_with_llm)
                except Exception:
                    logger.exception("Telemetry logging failed")

            except Exception as e:
                logger.exception("Unhandled error in inference loop: %s", e)
            finally:
                # Sleep to maintain interval
                elapsed = time.time() - t0
                to_sleep = max(0.0, self.inference_interval - elapsed)
                time.sleep(to_sleep)

    def _monitoring_loop(self):
        while not self._stop_event.is_set():
            time.sleep(10.0)
            with self._state_lock:
                latencies = list(self.latency_window)
                extract_lat = list(self.extract_latency_window)
                model_lat = list(self.model_latency_window)
            if not latencies:
                continue
            import statistics

            mean = statistics.mean(latencies)
            try:
                p95 = sorted(latencies)[int(0.95 * len(latencies))]
            except Exception:
                p95 = max(latencies)
            try:
                p99 = sorted(latencies)[int(0.99 * len(latencies))]
            except Exception:
                p99 = max(latencies)
            # compute component stats if available
            try:
                ex_mean = statistics.mean(extract_lat) if extract_lat else 0.0
            except Exception:
                ex_mean = 0.0
            try:
                model_mean = statistics.mean(model_lat) if model_lat else 0.0
            except Exception:
                model_mean = 0.0
            logger.info("Latency stats ms: mean=%.1f p95=%.1f p99=%.1f | extract_mean=%.1f model_mean=%.1f", mean, p95, p99, ex_mean, model_mean)

    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            if self._latest_decision is None:
                return None
            return dict(self._latest_decision)

    def get_status(self) -> Dict[str, Any]:
        with self._state_lock:
            latencies = list(self.latency_window)
            extract_lat = list(self.extract_latency_window)
            model_lat = list(self.model_latency_window)
        uptime = (time.time() - self._start_ts) if self._start_ts else 0.0
        status = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "buffer_sizes": self.buffer.get_buffer_size(),
            "sensor_health": self.buffer.get_sensor_health_snapshot(),
            "total_inferences": self.total_inferences,
            "uptime_s": uptime,
            "latency_stats": {
                "mean_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
                "p95_ms": (sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0),
                "p99_ms": (sorted(latencies)[int(0.99 * len(latencies))] if latencies else 0.0),
                "extract_mean_ms": (sum(extract_lat) / len(extract_lat)) if extract_lat else 0.0,
                "model_mean_ms": (sum(model_lat) / len(model_lat)) if model_lat else 0.0,
            },
        }
        # include LLM integration stats if present
        try:
            if getattr(self, "llm_integration", None):
                status["llm_integration"] = self.llm_integration.get_stats()
        except Exception:
            logger.exception("Failed to gather LLMIntegration stats")
        return status

    def reset(self):
        self.buffer.clear()
        self.decision_unit.reset()
        try:
            if hasattr(self, 'alert_manager') and self.alert_manager:
                self.alert_manager.reset()
        except Exception:
            logger.exception("Failed to reset alert manager")
        with self._state_lock:
            self._latest_risk = None
            self._latest_decision = None
            self.total_inferences = 0
            self.latency_window.clear()
