import json
import numpy as np
from typing import Dict, Any
from sklearn import metrics

# SciPy is optional: fall back to lighter-weight approximations if missing.
try:
    from scipy import stats  # type: ignore
    HAVE_SCIPY = True
except Exception:
    stats = None
    HAVE_SCIPY = False


class ABTestingFramework:
    def __init__(self, config=None):
        cfg = config or {}
        self.significance = cfg.get('significance_level', 0.05)
        self.bootstrap_iters = cfg.get('bootstrap_iterations', 1000)
        self.min_sample_size = cfg.get('min_sample_size', 100)

    def mcnemar_test(self, y_true, y_pred_a, y_pred_b, threshold=0.5) -> Dict[str, Any]:
        a = (y_pred_a >= threshold).astype(int)
        b = (y_pred_b >= threshold).astype(int)
        n01 = int(((a == 0) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        # Prefer SciPy chi2 if available; otherwise use a simple normal approximation
        if HAVE_SCIPY and hasattr(stats, 'chi2'):
            try:
                stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10 + 1e-9)
                p = 1 - stats.chi2.cdf(stat, df=1)
            except Exception:
                stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10 + 1e-9)
                # fallback: use an approximate p-value from normal approximation
                z = (n01 - n10) / ((n01 + n10 + 1e-9) ** 0.5)
                from math import erfc

                p = 0.5 * erfc(abs(z) / (2 ** 0.5))
        else:
            # SciPy not available: return a descriptive skipped/warning entry
            stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10 + 1e-9)
            p = None
        significant = (p is not None and p < self.significance)
        return {'n01': n01, 'n10': n10, 'stat': stat, 'p_value': p, 'significant': significant, 'scipy_available': HAVE_SCIPY}

    def bootstrap_confidence_interval(self, y_true, y_pred_a, y_pred_b, metric='average_precision', n_iters=None):
        n_iters = n_iters or self.bootstrap_iters
        rng = np.random.default_rng(0)
        deltas = []
        n = len(y_true)
        for _ in range(n_iters):
            idx = rng.integers(0, n, n)
            a = y_pred_a[idx]
            b = y_pred_b[idx]
            y = y_true[idx]
            if metric == 'average_precision':
                ma = metrics.average_precision_score(y, a)
                mb = metrics.average_precision_score(y, b)
            else:
                ma = metrics.roc_auc_score(y, a)
                mb = metrics.roc_auc_score(y, b)
            deltas.append(ma - mb)
        arr = np.array(deltas)
        return {'mean_delta': float(arr.mean()), 'ci_lower': float(np.percentile(arr, 2.5)), 'ci_upper': float(np.percentile(arr, 97.5))}

    def compare_models(self, model_a_predict, model_b_predict, X_test, y_test) -> Dict[str, Any]:
        # model_predict functions should accept X and return score array
        scores_a = model_a_predict(X_test)
        scores_b = model_b_predict(X_test)
        report = {}
        report['pr_auc_a'] = metrics.average_precision_score(y_test, scores_a)
        report['pr_auc_b'] = metrics.average_precision_score(y_test, scores_b)
        report['roc_auc_a'] = metrics.roc_auc_score(y_test, scores_a)
        report['roc_auc_b'] = metrics.roc_auc_score(y_test, scores_b)
        report['mcnemar'] = self.mcnemar_test(y_test, scores_a, scores_b)
        report['bootstrap'] = self.bootstrap_confidence_interval(y_test, scores_a, scores_b)
        report['recommendation'] = 'deploy' if report['bootstrap']['mean_delta'] > 0 and report['mcnemar']['significant'] else 'hold'
        return report
