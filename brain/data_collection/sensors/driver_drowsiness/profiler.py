"""Latency profiling utilities for the drowsiness pipeline.

Provides a simple context manager API to measure durations of stages and
maintain rolling statistics for alerting and telemetry.
"""
from __future__ import annotations

import time
import threading
from collections import deque
from typing import Deque, Dict
import statistics
import logging

logger = logging.getLogger(__name__)


class LatencyProfiler:
    def __init__(self, budget_ms: int = 200, window_size: int = 100):
        self.budget_ms = int(budget_ms)
        self.window_size = int(window_size)
        self._lock = threading.Lock()
        self._per_stage: Dict[str, Deque[float]] = {}
        self._total_times: Deque[float] = deque(maxlen=self.window_size)
        self._budget_exceeded = 0

    class _ProfileCtx:
        def __init__(self, parent: "LatencyProfiler", stage: str):
            self.parent = parent
            self.stage = stage

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = (time.perf_counter() - self._t0) * 1000.0
            with self.parent._lock:
                q = self.parent._per_stage.setdefault(self.stage, deque(maxlen=self.parent.window_size))
                q.append(dt)

    def profile(self, stage_name: str):
        return LatencyProfiler._ProfileCtx(self, stage_name)

    def record_total(self, total_ms: float):
        with self._lock:
            self._total_times.append(total_ms)
            if total_ms > self.budget_ms:
                self._budget_exceeded += 1

    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            stats = {}
            for k, q in self._per_stage.items():
                if len(q) == 0:
                    continue
                arr = list(q)
                stats[f"{k}_mean_ms"] = float(statistics.mean(arr))
                stats[f"{k}_max_ms"] = float(max(arr))
                try:
                    stats[f"{k}_p95_ms"] = float(sorted(arr)[int(0.95 * len(arr))])
                except Exception:
                    stats[f"{k}_p95_ms"] = float(max(arr))
            total_arr = list(self._total_times)
            stats["total_mean_ms"] = float(statistics.mean(total_arr)) if total_arr else 0.0
            stats["total_max_ms"] = float(max(total_arr)) if total_arr else 0.0
            stats["budget_exceeded_count"] = int(self._budget_exceeded)
            stats["budget_exceeded_percent"] = (self._budget_exceeded / len(total_arr) * 100.0) if total_arr else 0.0
            return stats

    def log_warning_if_exceeded(self):
        stats = self.get_stats()
        if stats.get("total_mean_ms", 0.0) > self.budget_ms:
            logger.warning("LatencyProfiler: mean latency %.1fms exceeds budget %dms", stats.get("total_mean_ms"), self.budget_ms)

    def reset(self):
        with self._lock:
            self._per_stage.clear()
            self._total_times.clear()
            self._budget_exceeded = 0

    def to_dict(self) -> Dict[str, float]:
        return self.get_stats()
