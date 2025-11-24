"""BLEMeshAdapter: compute aggregate features from BLE mesh neighbor samples.

This follows the feature-adapter pattern used by DriverFatigueAdapter.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from config.config_loader import ConfigLoader
from utils.stats_utils import safe_mean, safe_std, safe_min, safe_max
from utils.ble_utils import filter_stale_neighbors, compute_closing_speed

logger = logging.getLogger(__name__)


class BLEMeshAdapter:
    """Adapter to compute mesh-level features from BLE samples.

    Config (sensors.ble_mesh):
      - stale_threshold_ms
      - closing_speed_threshold_mps
      - enable_aggregate_features
    """

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        sect = "sensors.ble_mesh"
        self.stale_threshold_ms = int(self.config.get(f"{sect}.stale_threshold_ms", 5000))
        self.closing_speed_threshold_mps = float(self.config.get(f"{sect}.closing_speed_threshold_mps", 2.0))
        self.enable_aggregate = bool(self.config.get(f"{sect}.enable_aggregate_features", True))

    def analyze(self, samples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not samples or not self.enable_aggregate:
            return None

        try:
            # collect per-sample neighbor counts and all neighbor distances/rssi
            neighbor_counts = []
            all_distances = []
            all_rssi = []
            unique_ids: Set[str] = set()
            approaching = 0
            receding = 0
            max_closing = 0.0

            prev_ids = None
            churns = 0
            samples_with_neighbors = 0

            for s in samples:
                neigh = s.get("neighbors") or []
                # filter stale
                neigh = filter_stale_neighbors(neigh, max_age_ms=self.stale_threshold_ms)
                if not neigh:
                    neighbor_counts.append(0)
                    prev_ids = set()
                    continue

                samples_with_neighbors += 1
                counts = len(neigh)
                neighbor_counts.append(counts)

                ids = set()
                for n in neigh:
                    ids.add(n.get("id_hash"))
                    d = n.get("dist_m")
                    r = n.get("rssi")
                    try:
                        if d is not None:
                            all_distances.append(float(d))
                        if r is not None:
                            all_rssi.append(float(r))
                    except Exception:
                        pass

                    # approaching/receding based on closing speed helper
                    try:
                        rel = compute_closing_speed(n)
                        if rel < -self.closing_speed_threshold_mps:
                            approaching += 1
                            if rel < max_closing:
                                max_closing = rel
                        elif rel > self.closing_speed_threshold_mps:
                            receding += 1
                    except Exception:
                        pass

                unique_ids.update(ids)

                if prev_ids is not None:
                    # churn = symmetric difference
                    churns += len(prev_ids.symmetric_difference(ids))
                prev_ids = ids

            if samples_with_neighbors == 0:
                return None

            features: Dict[str, Any] = {}
            features["neighbor_count_mean"] = safe_mean(neighbor_counts)
            features["neighbor_count_std"] = safe_std(neighbor_counts)
            features["neighbor_count_min"] = safe_min(neighbor_counts)
            features["neighbor_count_max"] = safe_max(neighbor_counts)
            features["nearest_distance_m"] = safe_min(all_distances)
            features["farthest_distance_m"] = safe_max(all_distances)
            features["avg_distance_m"] = safe_mean(all_distances)
            features["mesh_density"] = int(features["neighbor_count_max"] or 0)
            features["approaching_neighbors_count"] = int(approaching)
            features["receding_neighbors_count"] = int(receding)
            features["max_closing_speed_mps"] = float(max_closing)
            features["unique_neighbors_count"] = int(len(unique_ids))
            # churn rate normalized by total possible changes (approx)
            features["neighbor_churn_rate"] = churns / max(1, len(samples))
            features["avg_rssi"] = safe_mean(all_rssi)
            features["rssi_std"] = safe_std(all_rssi)
            features["sample_count"] = len(samples)

            return features
        except Exception as e:
            logger.exception("BLEMeshAdapter analysis failed: %s", e)
            return None
