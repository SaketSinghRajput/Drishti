import os
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class AnalyticsEngine:
    """Lightweight analytics engine parsing JSONL telemetry and explanation logs.

    Can optionally delegate to injected `telemetry_collector` and
    `explanation_logger` objects when provided; falls back to filesystem parsing.
    Results are cached in-memory for a TTL to reduce repeated computation.
    """

    def __init__(self, telemetry_dir: str, explanation_dir: str, cache_ttl_s: int = 300, telemetry_collector=None, explanation_logger=None):
        self.telemetry_dir = telemetry_dir
        self.explanation_dir = explanation_dir
        self.cache_ttl_s = cache_ttl_s
        self.telemetry_collector = telemetry_collector
        self.explanation_logger = explanation_logger
        self._cache = {}

    def _cache_get(self, key: str):
        entry = self._cache.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self.cache_ttl_s:
            del self._cache[key]
            return None
        return value

    def _cache_set(self, key: str, value):
        self._cache[key] = (time.time(), value)

    def _iter_jsonl(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def compute_false_alarm_rate(self, start_date: Optional[datetime], end_date: Optional[datetime], annotations_db=None) -> Dict:
        key = f"false_alarm:{start_date}:{end_date}"
        cached = self._cache_get(key)
        if cached:
            return cached
        # naive implementation: scan telemetry_dir JSONL files and count predictions with alert_level >= medium
        tp = 0
        fp = 0
        # If annotations_db is provided, use it to count TP/FP by joining on incident/prediction ids
        # For now, try to aggregate by reading annotations if available
        if annotations_db:
            ann = annotations_db.get_annotations()
            by_incident = {a["incident_id"]: a for a in ann}
        else:
            by_incident = {}

        # Prefer telemetry_collector if provided
        if self.telemetry_collector is not None and hasattr(self.telemetry_collector, 'iter_predictions'):
            iterator = self.telemetry_collector.iter_predictions(start_date=start_date, end_date=end_date)
        else:
            iterator = self._iter_directory_predictions(self.telemetry_dir)

        for rec in iterator:
            incident_id = rec.get("incident_id") or rec.get("prediction_id")
            if not incident_id:
                continue
            ann = by_incident.get(incident_id)
            if ann and ann.get("label") == "FP":
                fp += 1
            elif ann and ann.get("label") == "TP":
                tp += 1

        rate = None
        if tp + fp > 0:
            rate = fp / (tp + fp)

        result = {"rate": rate, "tp": tp, "fp": fp}
        self._cache_set(key, result)
        return result

    def compute_detection_latency_stats(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> Dict:
        key = f"latency:{start_date}:{end_date}"
        cached = self._cache_get(key)
        if cached:
            return cached
        latencies = []
        if self.explanation_logger is not None and hasattr(self.explanation_logger, 'iter_explanations'):
            iterator = self.explanation_logger.iter_explanations(start_date=start_date, end_date=end_date)
        else:
            iterator = self._iter_directory_explanations(self.explanation_dir)

        for rec in iterator:
            if "latency_ms" in rec:
                latencies.append(rec.get("latency_ms"))
        if not latencies:
            result = {"count": 0, "mean": None}
            self._cache_set(key, result)
            return result
        latencies.sort()
        n = len(latencies)
        mean = sum(latencies) / n
        def pct(p):
            idx = min(n - 1, int(p * n))
            return latencies[idx]

        result = {"count": n, "mean": mean, "p95": pct(0.95), "p99": pct(0.99), "max": latencies[-1]}
        self._cache_set(key, result)
        return result

    def compute_feature_importance_trends(self, start_date: Optional[datetime], end_date: Optional[datetime], top_n: int = 20) -> Dict:
        key = f"feat_trends:{start_date}:{end_date}:{top_n}"
        cached = self._cache_get(key)
        if cached:
            return cached
        agg = {}
        counts = {}
        if self.explanation_logger is not None and hasattr(self.explanation_logger, 'iter_explanations'):
            iterator = self.explanation_logger.iter_explanations(start_date=start_date, end_date=end_date)
        else:
            iterator = self._iter_directory_explanations(self.explanation_dir)

        for rec in iterator:
            shap = rec.get("shap_values") or {}
            for k, v in shap.items():
                agg[k] = agg.get(k, 0.0) + abs(v)
                counts[k] = counts.get(k, 0) + 1
        
        # basic trend: compare recent 7 days vs previous 30 days if timestamps available
        features = []
        for k, total_abs in agg.items():
            features.append({"feature": k, "mean_abs_shap": total_abs / max(1, counts.get(k, 1))})
        features.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        result = {"top_features": features[:top_n]}
        self._cache_set(key, result)
        return result

    def _iter_directory_predictions(self, telemetry_dir):
        # iterate files in directory and yield records
        if not os.path.isdir(telemetry_dir):
            return []
        for fname in os.listdir(telemetry_dir):
            p = os.path.join(telemetry_dir, fname)
            for rec in self._iter_jsonl(p):
                yield rec

    def _iter_directory_explanations(self, explanation_dir):
        if not os.path.isdir(explanation_dir):
            return []
        for fname in os.listdir(explanation_dir):
            p = os.path.join(explanation_dir, fname)
            for rec in self._iter_jsonl(p):
                yield rec

    def compute_alert_distribution(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> Dict:
        key = f"alert_dist:{start_date}:{end_date}"
        cached = self._cache_get(key)
        if cached:
            return cached
        counts = {}
        iterator = self.telemetry_collector.iter_predictions(start_date=start_date, end_date=end_date) if self.telemetry_collector and hasattr(self.telemetry_collector, 'iter_predictions') else self._iter_directory_predictions(self.telemetry_dir)
        total = 0
        for rec in iterator:
            lvl = rec.get('alert_level', 'low')
            counts[lvl] = counts.get(lvl, 0) + 1
            total += 1
        dist = {k: {'count': v, 'pct': v / total if total else 0.0} for k, v in counts.items()}
        result = {'total': total, 'distribution': dist}
        self._cache_set(key, result)
        return result

    def compute_sensor_reliability(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> Dict:
        key = f"sensor_rel:{start_date}:{end_date}"
        cached = self._cache_get(key)
        if cached:
            return cached
        counts = {}
        ok_counts = {}
        iterator = self.telemetry_collector.iter_sensor_health(start_date=start_date, end_date=end_date) if self.telemetry_collector and hasattr(self.telemetry_collector, 'iter_sensor_health') else self._iter_directory_predictions(self.telemetry_dir)
        total = 0
        for rec in iterator:
            total += 1
            sh = rec.get('sensor_health') or {}
            for s, v in sh.items():
                counts[s] = counts.get(s, 0) + 1
                if v.get('status') == 'ok' or v == 'ok':
                    ok_counts[s] = ok_counts.get(s, 0) + 1
        result = {}
        for s, c in counts.items():
            result[s] = {'availability_pct': ok_counts.get(s, 0) / c if c else None, 'total_samples': c}
        self._cache_set(key, result)
        return result

    def compute_model_performance_summary(self, annotations_db, start_date: Optional[datetime], end_date: Optional[datetime]) -> Dict:
        key = f"model_perf:{start_date}:{end_date}"
        cached = self._cache_get(key)
        if cached:
            return cached
        anns = annotations_db.get_annotations() if annotations_db else []
        tp = fp = tn = fn = 0
        for a in anns:
            label = a.get('label')
            if label == 'TP':
                tp += 1
            elif label == 'FP':
                fp += 1
            elif label == 'TN':
                tn += 1
            elif label == 'FN':
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None
        result = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}
        self._cache_set(key, result)
        return result
