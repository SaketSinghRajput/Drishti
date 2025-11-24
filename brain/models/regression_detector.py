from typing import Dict, Any
import math


class RegressionDetector:
    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = cfg.get('enabled', True)
        self.min_pr_auc = cfg.get('min_pr_auc', 0.7)
        self.max_pr_auc_drop = cfg.get('max_pr_auc_drop', 0.05)
        self.min_f1 = cfg.get('min_f1', 0.5)
        self.max_brier_increase = cfg.get('max_brier_increase', 0.1)

    def detect_regression(self, new_metrics: Dict[str, float], baseline_metrics: Dict[str, float], thresholds: Dict = None) -> Dict[str, Any]:
        if not self.enabled:
            return {'has_regression': False, 'failed_checks': []}
        failed = []
        # PR-AUC
        baseline_pr = baseline_metrics.get('pr_auc')
        new_pr = new_metrics.get('pr_auc')
        if baseline_pr is not None and new_pr is not None:
            if baseline_pr - new_pr > self.max_pr_auc_drop:
                failed.append('pr_auc_drop')
        # F1
        baseline_f1 = baseline_metrics.get('f1')
        new_f1 = new_metrics.get('f1')
        if baseline_f1 is not None and new_f1 is not None:
            if baseline_f1 - new_f1 > (baseline_f1 - self.min_f1 if baseline_f1>self.min_f1 else 0):
                failed.append('f1_drop')
        # Brier
        baseline_brier = baseline_metrics.get('brier')
        new_brier = new_metrics.get('brier')
        if baseline_brier is not None and new_brier is not None:
            if new_brier - baseline_brier > self.max_brier_increase:
                failed.append('brier_increase')

        severity = 'none'
        if len(failed) == 1:
            severity = 'minor'
        elif len(failed) == 2:
            severity = 'major'
        elif len(failed) >= 3:
            severity = 'critical'

        return {'has_regression': len(failed) > 0, 'failed_checks': failed, 'severity': severity}
