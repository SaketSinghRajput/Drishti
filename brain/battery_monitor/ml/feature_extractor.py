"""BMS feature extractor for ML models.

Computes a compact set of statistical features over a time window of BMS
telemetry samples. Designed to be simple and robust for a first-pass model.
"""
from __future__ import annotations

from typing import Any, Dict, List
from math import sqrt
from statistics import mean, pstdev

from config.config_loader import ConfigLoader
from utils import stats_utils

# Deterministic feature ordering to ensure model input vectors are stable
FEATURE_KEYS = [
    "cell_voltage_mean",
    "cell_voltage_std",
    "cell_voltage_min",
    "cell_voltage_max",
    "voltage_imbalance",
    "cell_temp_mean",
    "cell_temp_std",
    "cell_temp_max",
    "temp_gradient",
    "current_rms",
    "soc_mean",
    "soc_std",
    "soh_mean",
    "soh_std",
    "fault_count_total",
    # pack-level features
    "pack_power_mean_w",
    "pack_power_std_w",
    "pack_power_max_w",
    "voltage_drop_flag",
    "current_spike_flag",
    "pack_temp_mean_c",
    "pack_temp_max_c",
]


class BMSFeatureExtractor:
    def __init__(self, config: ConfigLoader | dict = None):
        # Ensure config-like accessor even when None is provided
        self.config = config
        if config is None:
            class _CfgStub:
                def get(self, key, default=None):
                    return default

            self.config = _CfgStub()
        self.window_s = 60
        if config:
            try:
                self.window_s = int(config.get("battery_monitor.ml.feature_window_s", self.window_s))
            except Exception:
                pass
    def _flatten_cell_lists(self, samples: List[Dict[str, Any]], key: str) -> List[float]:
        vals = []
        for s in samples:
            arr = s.get(key)
            if arr:
                try:
                    vals.extend([float(x) for x in arr])
                except Exception:
                    continue
        return vals

    def extract_features(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        if not samples:
            raise ValueError("samples empty")

        # cell voltage stats across all cells in window
        cell_volts = self._flatten_cell_lists(samples, "cell_voltages")
        if cell_volts:
            cell_voltage_mean = stats_utils.safe_mean(cell_volts)
            cell_voltage_std = stats_utils.safe_std(cell_volts)
            cell_voltage_min = min(cell_volts)
            cell_voltage_max = max(cell_volts)
        else:
            cell_voltage_mean = -999.0
            cell_voltage_std = 0.0
            cell_voltage_min = -999.0
            cell_voltage_max = -999.0

        # voltage imbalance per-sample mean
        imbalance_list = []
        for s in samples:
            v = s.get("cell_voltages")
            if v:
                try:
                    v = [float(x) for x in v]
                    imbalance_list.append(max(v) - min(v))
                except Exception:
                    continue
        voltage_imbalance = stats_utils.safe_mean(imbalance_list) if imbalance_list else 0.0

        # cell temp stats
        cell_temps = self._flatten_cell_lists(samples, "cell_temps")
        if cell_temps:
            cell_temp_mean = stats_utils.safe_mean(cell_temps)
            cell_temp_std = stats_utils.safe_std(cell_temps)
            cell_temp_max = max(cell_temps)
        else:
            cell_temp_mean = -999.0
            cell_temp_std = 0.0
            cell_temp_max = -999.0

        # temp gradient per-sample mean
        temp_grad_list = []
        for s in samples:
            t = s.get("cell_temps")
            if t:
                try:
                    t = [float(x) for x in t]
                    temp_grad_list.append(max(t) - min(t))
                except Exception:
                    continue
        temp_gradient = stats_utils.safe_mean(temp_grad_list) if temp_grad_list else 0.0

        # current RMS
        currents = [float(s.get("pack_current", 0.0)) for s in samples if s.get("pack_current") is not None]
        if currents:
            rms = sqrt(sum(c * c for c in currents) / len(currents))
        else:
            rms = 0.0

        # pack power stats (from power_w) and pack voltage/current samples
        powers = [float(s.get('power_w')) for s in samples if s.get('power_w') is not None]
        pack_power_mean = stats_utils.safe_mean(powers) if powers else 0.0
        pack_power_std = stats_utils.safe_std(powers) if powers and len(powers) > 1 else 0.0
        pack_power_max = max(powers) if powers else 0.0

        pack_voltages = [float(s.get('pack_voltage')) if s.get('pack_voltage') is not None else float(s.get('pack_voltage_v')) for s in samples if s.get('pack_voltage') is not None or s.get('pack_voltage_v') is not None]
        voltage_drop_flag = 0
        try:
            if len(pack_voltages) >= 2 and (max(pack_voltages) - min(pack_voltages)) > 0.5:
                voltage_drop_flag = 1
        except Exception:
            voltage_drop_flag = 0

        current_spike_flag = 0
        try:
            for s in samples:
                pv = s.get('pack_current') if s.get('pack_current') is not None else s.get('pack_current_a')
                if pv is not None and abs(float(pv)) > float(self.config.get('battery_monitor.rules.thresholds.pack_current_spike_a', self.config.get('battery_monitor.rules.thresholds.current_spike_threshold_a', 100.0))):
                    current_spike_flag = 1
                    break
        except Exception:
            current_spike_flag = 0

        pack_temp_vals = [float(s.get('pack_temp_c')) if s.get('pack_temp_c') is not None else float(s.get('max_temp_c')) for s in samples if s.get('pack_temp_c') is not None or s.get('max_temp_c') is not None]
        if pack_temp_vals:
            pack_temp_mean = stats_utils.safe_mean(pack_temp_vals)
            pack_temp_max = max(pack_temp_vals)
        else:
            pack_temp_mean = -999.0
            pack_temp_max = -999.0

        # SoC/SoH stats
        socs = [float(s.get("soc", 0.0)) for s in samples if s.get("soc") is not None]
        sohs = [float(s.get("soh", 0.0)) for s in samples if s.get("soh") is not None]
        soc_mean = stats_utils.safe_mean(socs) if socs else -999.0
        soc_std = stats_utils.safe_std(socs) if socs and len(socs) > 1 else 0.0
        soh_mean = stats_utils.safe_mean(sohs) if sohs else -999.0
        soh_std = stats_utils.safe_std(sohs) if sohs and len(sohs) > 1 else 0.0

        # fault count
        fault_count = 0
        for s in samples:
            f = s.get("fault_codes")
            if f:
                try:
                    fault_count += len(f)
                except Exception:
                    continue

        features = {
            "cell_voltage_mean": float(cell_voltage_mean),
            "cell_voltage_std": float(cell_voltage_std),
            "cell_voltage_min": float(cell_voltage_min),
            "cell_voltage_max": float(cell_voltage_max),
            "voltage_imbalance": float(voltage_imbalance),
            "cell_temp_mean": float(cell_temp_mean),
            "cell_temp_std": float(cell_temp_std),
            "cell_temp_max": float(cell_temp_max),
            "temp_gradient": float(temp_gradient),
            "current_rms": float(rms),
            "soc_mean": float(soc_mean),
            "soc_std": float(soc_std),
            "soh_mean": float(soh_mean),
            "soh_std": float(soh_std),
            "fault_count_total": int(fault_count),
            # pack-level features
            "pack_power_mean_w": float(pack_power_mean),
            "pack_power_std_w": float(pack_power_std),
            "pack_power_max_w": float(pack_power_max),
            "voltage_drop_flag": int(voltage_drop_flag),
            "current_spike_flag": int(current_spike_flag),
            "pack_temp_mean_c": float(pack_temp_mean),
            "pack_temp_max_c": float(pack_temp_max),
        }
        return features
