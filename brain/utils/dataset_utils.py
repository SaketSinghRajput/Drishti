"""Dataset utilities: balancing, weighting, temporal splits, and export helpers.

This module provides lightweight implementations that prefer minimal external
dependencies. Where `imbalanced-learn` or `pandas` are available they are used;
otherwise the code falls back to simpler strategies suitable for tests.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from utils.time_utils import parse_timestamp

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

logger = logging.getLogger(__name__)


def _extract_label_array(labels: List[Dict[str, Any]]) -> List[int]:
    return [1 if (l.get("labels") and l["labels"].get("is_accident")) else 0 for l in labels]


def balance_dataset_smote(labeled_windows: List[Dict[str, Any]], sampling_strategy: float = 0.5, k_neighbors: int = 5) -> List[Dict[str, Any]]:
    """Return a new list with additional synthetic samples when possible.

    If SMOTE is not available or too few minority samples, fall back to random
    oversampling by duplicating minority samples until the target ratio.
    Synthetic samples are marked with `synthetic_sample: True`.
    """
    labels = _extract_label_array(labeled_windows)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        logger.warning("No positive samples; skipping SMOTE")
        return labeled_windows[:]

    # Interpret sampling_strategy as desired minority:majority ratio r
    r = float(sampling_strategy)
    # target_pos is desired number of minority samples after resampling
    target_pos = int(max(n_pos, round(r * float(n_neg))))

    # If SMOTE available and there are enough positives
    if SMOTE is not None and pd is not None and n_pos >= (k_neighbors + 1):
        df = pd.DataFrame(labeled_windows)
        numeric_cols = [c for c in df.columns if df[c].dtype in ("int64", "float64")]
        if not numeric_cols:
            logger.info("No numeric features for SMOTE, falling back to duplication")
        else:
            X = df[numeric_cols].fillna(0).values
            y = np.array(labels)
            try:
                # SMOTE expects sampling_strategy as minority:majority ratio
                sm = SMOTE(sampling_strategy=float(r), k_neighbors=k_neighbors)
                X_res, y_res = sm.fit_resample(X, y)
                added = []
                if len(X_res) > len(X):
                    # build index lists by class to pick a source window when creating synthetic samples
                    class_idx = {0: [], 1: []}
                    for orig_i, labv in enumerate(y.tolist() if hasattr(y, 'tolist') else list(y)):
                        class_idx[int(labv)].append(orig_i)

                    for i in range(len(X), len(X_res)):
                        cls = int(y_res[i])
                        # pick a source original index of the same class (round-robin)
                        candidates = class_idx.get(cls) or [j for j, vv in enumerate(y) if int(vv) == cls]
                        src_idx = candidates[i % len(candidates)] if candidates else 0
                        src_win = labeled_windows[src_idx].copy()
                        # update numeric cols from resampled features
                        for j, col in enumerate(numeric_cols):
                            try:
                                val = X_res[i, j]
                                # numpy scalars may need conversion
                                try:
                                    v = float(val)
                                except Exception:
                                    v = val
                                src_win[col] = v
                            except Exception:
                                continue
                        src_win["synthetic_sample"] = True
                        # ensure labels key exists and matches y_res
                        src_win.setdefault("labels", {})
                        src_win["labels"]["is_accident"] = bool(int(y_res[i]))
                        added.append(src_win)
                    return labeled_windows + added
            except Exception:
                logger.exception("SMOTE failed, falling back to duplication")

    # fallback duplication logic aligned with target_pos
    if n_pos < target_pos:
        needed = target_pos - n_pos
        pos_samples = [w for i, w in enumerate(labeled_windows) if labels[i] == 1]
        if not pos_samples:
            return labeled_windows[:]
        out = labeled_windows[:]
        idx = 0
        while needed > 0:
            s = dict(pos_samples[idx % len(pos_samples)])
            s["synthetic_sample"] = True
            out.append(s)
            idx += 1
            needed -= 1
        return out
    return labeled_windows[:]


def compute_sample_weights(labels: List[int]) -> List[float]:
    """Compute inverse-frequency weights normalized to length of labels."""
    n = len(labels)
    if n == 0:
        return []
    from collections import Counter

    c = Counter(labels)
    weights = {}
    for k, v in c.items():
        if v <= 0:
            weights[k] = 1.0
        else:
            weights[k] = float(n) / float(v)
    out = [weights.get(l, 1.0) for l in labels]
    # normalize to sum==n
    s = sum(out)
    if s <= 0:
        return out
    factor = float(n) / float(s)
    return [float(x * factor) for x in out]


def temporal_train_test_split(labeled_windows: List[Dict[str, Any]], test_size: float = 0.2, val_size: float = 0.1, holdout_trips: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split by trip id to avoid leakage. Returns train, val, test lists."""
    if not labeled_windows:
        return [], [], []
    by_trip = {}
    for w in labeled_windows:
        trip = w.get("trip_id", "__no_trip__")
        by_trip.setdefault(trip, []).append(w)
    trips = sorted(by_trip.keys())
    if holdout_trips:
        test_trips = [t for t in trips if t in set(holdout_trips)]
        other_trips = [t for t in trips if t not in set(holdout_trips)]
    else:
        # simple temporal split by ordering trips (best-effort)
        other_trips = trips
        test_trips = []
        n = len(trips)
        ntest = max(1, int(n * test_size))
        nval = max(0, int(n * val_size))
        test_trips = trips[-ntest:]
        val_trips = trips[-(ntest + nval) : -ntest] if nval > 0 else []
    train, val, test = [], [], []
    for t in trips:
        items = by_trip.get(t, [])
        if t in test_trips:
            test.extend(items)
        else:
            train.extend(items)
    # if val was computed above, split it out
    if holdout_trips is None:
        # we computed val_trips above
        try:
            val_trips
        except NameError:
            val = []
        else:
            for t in val_trips:
                val.extend(by_trip.get(t, []))
            # remove val items from train
            train = [w for w in train if w.get("trip_id") not in set(val_trips)]
    return train, val, test


def export_labeled_dataset(labeled_windows: List[Dict[str, Any]], output_dir: str, fmt: str = "json") -> None:
    os.makedirs(output_dir, exist_ok=True)
    if fmt == "json":
        # validate dataset if validation helper is available
        try:
            from utils.validation import validate_labeled_dataset

            vs = validate_labeled_dataset(labeled_windows)
            logger.info('Validation summary for export: %s', vs)
        except Exception:
            logger.exception('Validation unavailable or failed; continuing export')

        min_ts = None
        max_ts = None
        n_acc = 0
        n_near = 0
        for w in labeled_windows:
            wid = w.get("window_id") or str(hash(str(w)))
            p = os.path.join(output_dir, f"{wid}.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(w, fh)
            lab = w.get("labels", {}) or {}
            if lab.get("is_accident"):
                n_acc += 1
            if lab.get("is_near_miss"):
                n_near += 1
            try:
                ts = parse_timestamp(w.get("timestamp"))
                if min_ts is None or ts < min_ts:
                    min_ts = ts
                if max_ts is None or ts > max_ts:
                    max_ts = ts
            except Exception:
                continue

        # write metadata
        meta = {
            "count": len(labeled_windows),
            "n_accidents": n_acc,
            "n_near_misses": n_near,
            "accident_ratio": float(n_acc) / max(1, len(labeled_windows)),
            "near_miss_ratio": float(n_near) / max(1, len(labeled_windows)),
            "min_timestamp": min_ts.isoformat() if min_ts is not None else None,
            "max_timestamp": max_ts.isoformat() if max_ts is not None else None,
        }
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    else:
        raise NotImplementedError("Only json export supported in lightweight utils")


def load_labeled_dataset(input_dir: str, fmt: str = "json", trip_id: Optional[str] = None, start_ts: Optional[str] = None, end_ts: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load labeled dataset JSON files from `input_dir`.

    Optional filters:
      - `trip_id`: return only windows with matching `trip_id`.
      - `start_ts` / `end_ts`: ISO timestamps to filter by `timestamp`.
    """
    out = []
    if fmt == "json":
        for fn in os.listdir(input_dir):
            if not fn.endswith(".json"):
                continue
            if fn == "metadata.json":
                continue
            p = os.path.join(input_dir, fn)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                # filters
                if trip_id is not None and obj.get("trip_id") != trip_id:
                    continue
                if start_ts or end_ts:
                    try:
                        ts = parse_timestamp(obj.get("timestamp"))
                        if start_ts:
                            st = parse_timestamp(start_ts)
                            if ts < st:
                                continue
                        if end_ts:
                            et = parse_timestamp(end_ts)
                            if ts > et:
                                continue
                    except Exception:
                        # if parse fails, skip the filter and include object
                        pass
                out.append(obj)
            except Exception:
                logger.exception("Failed to load %s", p)
    else:
        raise NotImplementedError("Only json import supported in lightweight utils")
    return out


__all__ = [
    "balance_dataset_smote",
    "compute_sample_weights",
    "temporal_train_test_split",
    "export_labeled_dataset",
    "load_labeled_dataset",
]
