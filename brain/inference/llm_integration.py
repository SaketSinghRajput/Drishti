"""LLMIntegration: optional Gemini (google-generativeai) wrapper for edge risk augmentation.

Design goals:
- Lazy-load the Gemini client only when needed.
- TTL LRU cache to avoid repeat requests for identical inputs.
- Token-bucket rate limiter to keep API usage bounded.
- Anonymize identifiers before sending prompts.
- Non-blocking behaviour via a short timeout and graceful fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMIntegration:
    def __init__(self, config):
        self.config = config
        self.enabled = bool(self.config.get("inference.llm_integration.enabled", False))
        if not self.enabled:
            return

        # Basic settings
        self.api_key_env = self.config.get("inference.llm_integration.api_key_env", "GEMINI_API_KEY")
        # support new config key `model_name` (Gemini model identifiers)
        self.model_name = self.config.get("inference.llm_integration.model_name", self.config.get("inference.llm_integration.model", "gemini-1.5-flash"))
        self.timeout_s = float(self.config.get("inference.llm_integration.timeout_s", 0.5))
        self.multimodal = bool(self.config.get("inference.llm_integration.multimodal_enabled", False))
        self.anonymize = bool(self.config.get("inference.llm_integration.anonymize_vehicle_hash", True))

        # Cache settings (simple TTL LRU)
        self.cache_ttl = int(self.config.get("inference.llm_integration.cache_ttl_s", 300))
        self.cache_max = int(self.config.get("inference.llm_integration.cache_max_entries", 1024))
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_lock = threading.Lock()

        # Rate limiting: token bucket semantics
        per_min = float(self.config.get("inference.llm_integration.rate_limit_per_minute", 30))
        # capacity = max burst tokens (per_min allows burst up to per_min)
        self._rate_capacity = max(1.0, per_min)
        # refill rate per second
        self._refill_per_sec = float(per_min) / 60.0
        # start full
        self._tokens = float(self._rate_capacity)
        self._last_refill = time.time()
        self._rate_lock = threading.Lock()

        # Override behavior threshold
        self.override_confidence = float(self.config.get("inference.llm_integration.override_confidence", 0.9))

        # Stats
        self._stats = {"calls": 0, "cache_hits": 0, "cache_misses": 0, "rate_limited": 0, "errors": 0}
        self._stats_lock = threading.Lock()

        # Thread pool for non-blocking calls
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Lazy client placeholder
        self._client = None
        self._client_lock = threading.Lock()

    def _refill_tokens(self):
        now = time.time()
        with self._rate_lock:
            elapsed = now - self._last_refill
            refill = elapsed * self._refill_per_sec
            if refill > 0:
                self._tokens = min(self._rate_capacity, self._tokens + refill)
                self._last_refill = now

    def _consume_token(self) -> bool:
        self._refill_tokens()
        with self._rate_lock:
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def _should_invoke(self, current_score: float, sensor_health: Dict[str, Any], data_quality: Optional[float] = 1.0, explanation: Optional[Dict[str, Any]] = None, window_data: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> bool:
        """Policy deciding whether to call the LLM for this window.

        Rules (configurable):
        - invoke_on_low_confidence: call when model confidence (or complementary heuristics) low
        - invoke_on_uncertainty: call when sensor health indicates missing/stale critical sensors or low data quality
        - invoke_on_margin: call when current_score is within margin of configured risk thresholds
        """
        if not self.enabled:
            return False
        try:
            # basic thresholds
            invoke_low_conf = bool(self.config.get("inference.llm_integration.invoke_on_low_confidence", True))
            low_conf_th = float(self.config.get("inference.llm_integration.low_confidence_threshold", 0.6))
            invoke_uncertainty = bool(self.config.get("inference.llm_integration.invoke_on_uncertainty", True))
            data_quality_th = float(self.config.get("inference.llm_integration.uncertainty_data_quality_threshold", 0.6))
            invoke_margin = bool(self.config.get("inference.llm_integration.invoke_on_margin", True))
            margin = float(self.config.get("inference.llm_integration.margin_around_risk_thresholds", 0.05))

            # low confidence trigger (we interpret model score near medium/high boundaries as uncertain)
            if invoke_low_conf and current_score >= low_conf_th and current_score < 1.0:
                return True

            # data quality / sensor uncertainty
            if invoke_uncertainty and (data_quality is not None and data_quality < data_quality_th):
                return True
            if invoke_uncertainty and sensor_health:
                for s, info in sensor_health.items():
                    if info is None or info.get("stale", False):
                        return True

            # margin-based: if score is within margin of any threshold
            if invoke_margin:
                thresholds = self.config.get("model.risk_thresholds", {"low": 0.1, "medium": 0.3, "high": 0.6})
                for k, v in thresholds.items():
                    try:
                        v_f = float(v)
                        if abs(current_score - v_f) <= margin:
                            return True
                    except Exception:
                        continue

            # complex scenario heuristic: presence of window_data with camera metadata indicating close objects
            if window_data:
                cam = window_data.get("camera_metadata")
                if cam:
                    nearest = cam.get("nearest_object_distance_m")
                    if nearest is not None and float(nearest) <= float(self.config.get("inference.llm_integration.nearest_object_threshold_m", 5.0)):
                        return True

            return False
        except Exception:
            # on errors, be conservative and avoid calling
            return False

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._cache_lock:
            rec = self._cache.get(key)
            if not rec:
                return None
            # check TTL
            if time.time() - rec["ts"] > self.cache_ttl:
                try:
                    del self._cache[key]
                except Exception:
                    pass
                return None
            # move to end (most recent)
            self._cache.move_to_end(key)
            return rec["val"]

    def _cache_set(self, key: str, val: Dict[str, Any]) -> None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = {"ts": time.time(), "val": val}
            while len(self._cache) > self.cache_max:
                try:
                    self._cache.popitem(last=False)
                except Exception:
                    break

    def _make_cache_key(self, features: Dict[str, Any], explanation: Dict[str, Any], current_score: float, camera_metadata: Optional[Dict[str, Any]] = None, window_data: Optional[Dict[str, Any]] = None) -> str:
        # Keep the key compact: include a sorted subset of features, explanation message and score
        try:
            small = {k: features.get(k) for k in sorted(list(features.keys()))}
        except Exception:
            small = features
        payload = {"f": small, "e": explanation.get("message") if explanation else None, "s": float(current_score)}
        # include a compact camera signature if present
        if camera_metadata:
            try:
                payload["cam"] = {"nearest": camera_metadata.get("nearest_object_distance_m"), "lane_offset": camera_metadata.get("lane_offset_m")}
            except Exception:
                pass
        if window_data:
            try:
                payload["w_hash"] = hash(str(window_data.get("window_id") or window_data.get("start_ts") or ""))
            except Exception:
                pass
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _ensure_client(self):
        # Lazy import and client setup. Do not throw if unavailable; we'll fallback.
        with self._client_lock:
            if self._client is not None:
                return
            try:
                import os
                try:
                    import google.generativeai as genai
                except Exception:
                    # client not available in environment
                    self._client = None
                    logger.info("google.generativeai not available; LLMIntegration offline")
                    return
                api_key = os.environ.get(self.api_key_env)
                if not api_key:
                    logger.info("No API key in env %s; LLMIntegration offline", self.api_key_env)
                    self._client = None
                    return
                # configure global client
                genai.configure(api_key=api_key)
                # create a GenerativeModel instance for the configured model name
                try:
                    model = genai.GenerativeModel(self.model_name)
                except Exception:
                    # older/newer versions may vary; fall back to module-level access
                    model = getattr(genai, "GenerativeModel", None)
                    if callable(model):
                        model = model(self.model_name)
                self._client = model
            except Exception as e:
                logger.exception("Failed to initialize Gemini client: %s", e)
                self._client = None

    def _build_prompt(self, features: Dict[str, Any], explanation: Dict[str, Any], sensor_health: Dict[str, Any], current_score: float, camera_metadata: Optional[Dict[str, Any]] = None, window_data: Optional[Dict[str, Any]] = None) -> str:
        # Privacy-first, structured prompt tailored for accident risk assessment over the configured prediction horizon.
        parts = ["You are an expert safety assistant that assesses accident risk over the upcoming prediction horizon."]
        parts.append(f"Current model risk score: {current_score:.3f}")

        # Summarize camera metadata into human-readable scene description
        if camera_metadata:
            try:
                scene = []
                if camera_metadata.get("nearest_object_distance_m") is not None:
                    scene.append(f"Nearest object at {camera_metadata.get('nearest_object_distance_m'):.1f} m")
                if camera_metadata.get("lane_offset_m") is not None:
                    scene.append(f"lane offset {camera_metadata.get('lane_offset_m'):.2f} m")
                obj_types = camera_metadata.get("object_types", [])
                if obj_types:
                    scene.append(f"objects: {', '.join(map(str, obj_types))}")
                if scene:
                    parts.append("Scene summary:")
                    parts.extend([f"- {s}" for s in scene])
            except Exception:
                parts.append("Scene summary: (omitted)")

        # Structured sensor context
        parts.append("Sensor context (driver state, environment, road, nearby vehicles):")
        try:
            # driver-related features
            driver_keys = [k for k in (features or {}).keys() if 'driver' in k or 'drowsiness' in k or 'perclos' in k or 'blink' in k]
            if driver_keys:
                parts.append("Driver features:")
                for k in sorted(driver_keys)[:10]:
                    parts.append(f"- {k}: {features.get(k)}")
            # environment/road
            env_keys = [k for k in (features or {}).keys() if 'visibility' in k or 'road' in k or 'surface' in k]
            if env_keys:
                parts.append("Environment/road features:")
                for k in sorted(env_keys)[:10]:
                    parts.append(f"- {k}: {features.get(k)}")
            # proximity features
            prox_keys = [k for k in (features or {}).keys() if 'distance' in k or 'nearest' in k or 'proximity' in k]
            if prox_keys:
                parts.append("Proximity features:")
                for k in sorted(prox_keys)[:10]:
                    parts.append(f"- {k}: {features.get(k)}")
        except Exception:
            parts.append("- (feature summary omitted)")

        # sensor health summary (available/stale)
        parts.append("Sensor health summary:")
        try:
            for s, info in (sensor_health or {}).items():
                parts.append(f"- {s}: {'stale' if info.get('stale') else 'ok'}")
        except Exception:
            parts.append("- (sensor health omitted)")

        if explanation and explanation.get("message"):
            parts.append("Model explanation:")
            parts.append(explanation.get("message"))

        # If window_data contains notable events, summarize briefly
        if window_data:
            try:
                ev = window_data.get("events") or {}
                if ev:
                    parts.append("Recent notable events:")
                    for k, v in list(ev.items())[:10]:
                        parts.append(f"- {k}: {v}")
            except Exception:
                pass

        # Instructions for structured JSON response
        parts.append(
            "Respond with a JSON object containing at least these fields: `risk_level` (one of LOW/MEDIUM/HIGH/CRITICAL), `llm_score` (0.0-1.0), `confidence` (0.0-1.0), `reasoning` (concise natural-language explanation), `key_risk_factors` (list of strings), and `recommended_actions` (list of short action strings). Also include any numeric indicators helpful for downstream use. Do NOT include raw identifiers (vehicle ids, timestamps, or PII). Keep the response compact."
        )
        return "\n".join(parts)

    def _call_gemini(self, prompt: str, multimodal_payload: Optional[bytes] = None) -> Optional[Dict[str, Any]]:
        # Attempt to call the Gemini API via google.generativeai GenerativeModel.generate_content
        self._ensure_client()
        if not self._client:
            return None
        try:
            parts = []
            # Gemini parts: text part
            parts.append({"type": "input_text", "text": prompt})
            # multimodal: attach image bytes if provided and enabled
            if self.multimodal and multimodal_payload:
                parts.append({"type": "input_image_bytes", "bytes": multimodal_payload})

            # Try calling generate_content on the model instance
            try:
                response = self._client.generate_content(input=parts)
            except TypeError:
                # some versions use `generate` or `generate_content(parts=...)`
                try:
                    response = self._client.generate_content(parts)
                except Exception:
                    response = getattr(self._client, "generate", lambda *a, **k: None)(parts)

            text = None
            # Response parsing: support common shapes
            #  - response.text
            #  - response.candidates[0].content.parts
            #  - response.candidates[0].content[0].text
            try:
                if hasattr(response, "text") and response.text:
                    text = response.text
                elif hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    # candidates content may be list of parts
                    cont = getattr(cand, "content", None)
                    if isinstance(cont, (list, tuple)) and len(cont) > 0:
                        # concatenate text pieces
                        parts_out = []
                        for p in cont:
                            if isinstance(p, dict) and p.get("type") == "output_text":
                                parts_out.append(p.get("text") or p.get("text_raw") or str(p))
                            else:
                                # object may be a simple wrapper
                                try:
                                    parts_out.append(str(p.get("text")))
                                except Exception:
                                    parts_out.append(str(p))
                        text = "\n".join([t for t in parts_out if t])
                    else:
                        # try direct attribute
                        text = getattr(cand, "text", None) or str(cand)
            except Exception:
                text = str(response)

            if not text:
                text = str(response)

            # try to extract JSON blob from text
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                raw = text[start:end+1]
                parsed = json.loads(raw)
                return parsed
            # fallback: try to parse the whole text
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            logger.exception("Gemini call failed: %s", e)
            with self._stats_lock:
                self._stats["errors"] += 1
            return None

    def enhance_risk_assessment(self, features: Dict[str, Any], explanation: Dict[str, Any], sensor_health: Dict[str, Any], current_score: float, camera_metadata: Optional[Dict[str, Any]] = None, window_data: Optional[Dict[str, Any]] = None, multimodal_payload: Optional[bytes] = None) -> Optional[Dict[str, Any]]:
        """Main entrypoint. Returns llm_output dict or None.

        Behavior:
        - Check cache first.
        - Enforce rate limit.
        - Submit to Gemini on background thread and wait up to `timeout_s`.
        - Store response in cache.
        - Return structured dict: {llm_score, confidence, reasoning, metadata}
        """
        if not self.enabled:
            return None

        # Invocation gating: check whether we should call the LLM for this window
        try:
            if not self._should_invoke(float(current_score), sensor_health or {}, data_quality=window_data.get("metadata", {}).get("data_quality_score") if window_data else None, explanation=explanation, window_data=window_data, features=features):
                return None
        except Exception:
            # on gating errors, avoid calling
            return None

        key = self._make_cache_key(features or {}, explanation or {}, current_score, camera_metadata=camera_metadata, window_data=window_data)
        cached = self._cache_get(key)
        if cached is not None:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            return dict(cached)
        with self._stats_lock:
            self._stats["cache_misses"] += 1

        # Rate limit check
        if not self._consume_token():
            with self._stats_lock:
                self._stats["rate_limited"] += 1
            return None

        prompt = self._build_prompt(features or {}, explanation or {}, sensor_health or {}, float(current_score), camera_metadata=camera_metadata, window_data=window_data)

        def _work():
            with self._stats_lock:
                self._stats["calls"] += 1
            try:
                parsed = self._call_gemini(prompt, multimodal_payload=multimodal_payload)
                if not parsed:
                    return None
                # Parse structured fields with backward compatibility
                # Preferred fields: risk_level, llm_score, confidence, reasoning, key_risk_factors, recommended_actions
                try:
                    llm_score = float(parsed.get("llm_score") or parsed.get("score") or parsed.get("numeric_score") or 0.0)
                except Exception:
                    llm_score = 0.0
                try:
                    confidence = float(parsed.get("confidence") or parsed.get("conf") or 0.0)
                except Exception:
                    confidence = 0.0
                reasoning = parsed.get("reasoning") or parsed.get("explanation") or parsed.get("reason") or None
                risk_level = parsed.get("risk_level") or parsed.get("level")
                key_factors = parsed.get("key_risk_factors") or parsed.get("risk_factors") or []
                recommended = parsed.get("recommended_actions") or parsed.get("actions") or []

                out = {
                    "llm_score": max(0.0, min(1.0, float(llm_score))),
                    "confidence": max(0.0, min(1.0, float(confidence))),
                    "reasoning": (reasoning if reasoning is not None else str(parsed)),
                    "risk_level": (risk_level if risk_level is not None else None),
                    "key_risk_factors": key_factors,
                    "recommended_actions": recommended,
                    "metadata": {},
                }
                return out
            except Exception:
                with self._stats_lock:
                    self._stats["errors"] += 1
                return None

        future = self._executor.submit(_work)
        try:
            result = future.result(timeout=self.timeout_s)
        except TimeoutError:
            with self._stats_lock:
                self._stats["errors"] += 1
            return None
        except Exception:
            with self._stats_lock:
                self._stats["errors"] += 1
            return None

        if result is not None:
            # cache and return
            self._cache_set(key, result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            s = dict(self._stats)
        # include cache size
        with self._cache_lock:
            s["cache_size"] = len(self._cache)
        return s
