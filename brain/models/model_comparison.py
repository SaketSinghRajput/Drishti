"""Utilities to compare models (XGBoost baseline vs deep models).

Produces tables and plots comparing metrics, latency, and model size.

Note on comparability of metrics
--------------------------------
This comparator accepts models trained on different input spaces (for
example: tabular/XGBoost, sequence-based LSTM/CNN, or multimodal models).
As a result, metrics computed for different model types may be evaluated on
different sample sets and input representations. For example, a tabular
baseline may be evaluated on rows extracted from windows while a sequence
model is evaluated on sequence-aligned windows; multimodal models may
have missing camera data zero-padded or aligned differently.

Consequences:
- Metrics are comparable in a coarse, high-level sense (e.g., overall AUC
    trends), but they may *not* represent strictly paired evaluations on the
    identical set of samples. Care should be taken when interpreting small
    differences between models of different input modalities.
- Where strict paired comparisons are required, callers should ensure that
    inputs supplied in `inputs` represent the same sample set (same ordering
    and labels) for every model involved, or filter/align inputs to a common
    subset prior to constructing the `ModelComparator`.

Future option:
----------------
An explicit mode to automatically intersect and restrict comparisons to a
common subset of samples could be added. This would require the comparator
to know how to map samples across modalities (e.g., via window IDs present
in per-sample metadata) and to reindex input arrays accordingly. For now,
the comparator prefers horizon-specific keys (e.g., `sequence_h{h}`) when
available and otherwise falls back to generic keys.
"""
from typing import List, Dict, Any
import os
import json
import pandas as pd
import numpy as np
import logging

from . import evaluation
from ..utils import model_compression
from sklearn import metrics as skmetrics
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


class ModelComparator:
    def __init__(self, models: List[Dict[str, Any]], inputs: Dict[str, Any]):
        """models: list of dicts {'name': str, 'predict_fn': callable, 'type': str, 'path': str (optional)}

        inputs: dict with keys 'xgboost'->(X,y), 'sequence'->(X_seq,y), 'multimodal'->((Xc,Xs,Xt), y)
        """
        self.models = models
        self.inputs = inputs

    def _select_input_for_model(self, m: Dict[str, Any]):
        t = m.get('type')
        h = m.get('horizon', None)
        if t == 'xgboost':
            # prefer the generic 'xgboost' key when present (backwards compatibility)
            if 'xgboost' in self.inputs:
                return self.inputs.get('xgboost')
            # if the model specifies a horizon, prefer the horizon-specific key
            if h is not None:
                key = f'xgboost_h{h if h is not None else "all"}'
                if key in self.inputs:
                    return self.inputs.get(key)
            # final fallback: return any available xgboost_h* key
            for k in self.inputs.keys():
                if k.startswith('xgboost_h'):
                    return self.inputs.get(k)
            return None
        if t in ('lstm', 'cnn'):
            # prefer horizon-specific sequence if model indicates a horizon
            if h is not None:
                key = f'sequence_h{h if h is not None else "default"}'
                if key in self.inputs:
                    return self.inputs.get(key)
            # fallback: if only one sequence key exists, use it
            if 'sequence' in self.inputs:
                return self.inputs.get('sequence')
            # try to find any sequence_* key
            for k in self.inputs.keys():
                if k.startswith('sequence_'):
                    return self.inputs.get(k)
            return None
        if t == 'multimodal':
            if h is not None:
                key = f'multimodal_h{h if h is not None else "default"}'
                if key in self.inputs:
                    return self.inputs.get(key)
            if 'multimodal' in self.inputs:
                return self.inputs.get('multimodal')
            for k in self.inputs.keys():
                if k.startswith('multimodal_'):
                    return self.inputs.get(k)
            return None
        # fallback
        return self.inputs.get('xgboost')

    def compare_metrics(self):
        rows = []
        roc_data = {}
        pr_data = {}
        for m in self.models:
            inp = self._select_input_for_model(m)
            if inp is None:
                LOGGER.warning('No input found for model %s of type %s', m.get('name'), m.get('type'))
                continue
            # Inputs are stored as (X, y) where X may itself be a tuple for multimodal models
            X_test, y_test = inp
            # if multimodal inputs, X_test may be tuple
            try:
                preds = m['predict_fn'](X_test)
            except Exception as exc:
                LOGGER.exception('Prediction failed for model %s: %s', m.get('name'), exc)
                preds = np.zeros_like(y_test, dtype=float)
            metrics = evaluation.compute_metrics(y_test, preds)
            rows.append({'model': m['name'], 'type': m.get('type'), **metrics})
            # Compute ROC and PR curves when possible. If the label distribution
            # is degenerate (all positives or all negatives) sklearn may raise
            # a ValueError; in that case we skip storing curve data but keep
            # scalar metrics so the rest of the report remains useful.
            try:
                fpr, tpr, _ = skmetrics.roc_curve(y_test, preds)
                prec, rec, _ = skmetrics.precision_recall_curve(y_test, preds)
                roc_data[m['name']] = (fpr, tpr)
                pr_data[m['name']] = (rec, prec)
            except Exception as exc:
                LOGGER.warning('Skipping ROC/PR curves for model %s due to error computing curves: %s', m.get('name'), exc)
        df = pd.DataFrame(rows)
        return df, roc_data, pr_data

    def _measure_size_and_latency(self, m: Dict[str, Any], sample_input=None):
        info = {}
        path = m.get('path')
        if path and os.path.exists(path):
            try:
                info['size_mb'] = model_compression.measure_model_size(path)
            except Exception:
                info['size_mb'] = None
            # latency only for TFLite
            if path.endswith('.tflite'):
                try:
                    mean_ms, std_ms = model_compression.benchmark_inference_latency(path, sample_input)
                    # translate NaN results from benchmarking into None for JSON-friendly output
                    try:
                        if np.isnan(mean_ms) or np.isnan(std_ms):
                            info['latency_ms_mean'] = None
                            info['latency_ms_std'] = None
                        else:
                            info['latency_ms_mean'] = float(mean_ms)
                            info['latency_ms_std'] = float(std_ms)
                    except Exception:
                        info['latency_ms_mean'] = None
                        info['latency_ms_std'] = None
                except Exception:
                    info['latency_ms_mean'] = None
                    info['latency_ms_std'] = None
        return info

    def benchmark_edge_deployment(self, model_names: List[str] = None, sample_inputs: Dict[str, Any] = None):
        """Benchmark size and latency for selected models and return a DataFrame.

        Args:
            model_names: list of model names to benchmark; if None, benchmark all models with paths.
            sample_inputs: optional mapping model_name -> sample_input to use for latency.
        Returns:
            pandas.DataFrame with columns ['model','size_mb','latency_ms_mean','latency_ms_std']
        """
        rows = []
        to_run = self.models if model_names is None else [m for m in self.models if m.get('name') in model_names]
        for m in to_run:
            si = None
            if sample_inputs and m.get('name') in sample_inputs:
                si = sample_inputs[m.get('name')]
            else:
                inp = self._select_input_for_model(m)
                if inp is not None:
                    X_test = inp[0]
                    si = [x[:1] for x in X_test] if isinstance(X_test, tuple) else X_test[:1]
            info = self._measure_size_and_latency(m, si)
            info['model'] = m.get('name')
            rows.append(info)
        return pd.DataFrame(rows)

    def _get_preds_for_model(self, m: Dict[str, Any]):
        """Return (y_true, preds) for a given model by selecting appropriate input."""
        inp = self._select_input_for_model(m)
        if inp is None:
            return None, None
        X_test, y_test = inp
        try:
            preds = m['predict_fn'](X_test)
        except Exception:
            preds = np.zeros(len(y_test), dtype=float)
        return y_test, np.asarray(preds, dtype=float)

    def _any_input_of_type(self, t: str):
        """Return any available input tuple for a given type (e.g., 'multimodal' or 'sequence').

        Prefers the generic key (e.g., 'multimodal') if present, otherwise returns the
        first key matching the pattern '<type>_h*'. Returns None if no input available.
        """
        if t in self.inputs:
            return self.inputs[t]
        for k in self.inputs.keys():
            if k.startswith(f'{t}_h'):
                return self.inputs[k]
        return None

    def compare_significance(self, name_a: str, name_b: str, n_boot: int = 1000, metric: str = 'pr_auc') -> Dict[str, float]:
        """Perform a simple paired bootstrap to test difference in `metric` between two models.

        Returns dict with observed difference and p-value.
        """
        # find models
        ma = next((m for m in self.models if m.get('name') == name_a), None)
        mb = next((m for m in self.models if m.get('name') == name_b), None)
        if ma is None or mb is None:
            raise ValueError('Model names not found')
        ya, pa = self._get_preds_for_model(ma)
        yb, pb = self._get_preds_for_model(mb)
        if ya is None or pa is None or yb is None or pb is None:
            raise RuntimeError('Missing predictions for models')
        # ensure same labels
        y = ya
        rng = np.random.default_rng(42)
        def score(preds):
            from sklearn.metrics import average_precision_score
            if metric == 'pr_auc':
                return float(average_precision_score(y, preds))
            if metric == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                return float(roc_auc_score(y, preds))
            return 0.0
        obs_diff = score(pa) - score(pb)
        diffs = []
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            s_a = score(pa[idx])
            s_b = score(pb[idx])
            diffs.append(s_a - s_b)
        diffs = np.asarray(diffs)
        pval = float(np.mean(diffs >= obs_diff))
        return {'observed_diff': float(obs_diff), 'p_value': pval}

    def ensemble_predict(self, model_names: List[str]):
        """Average predictions from multiple models using each model's own test input.

        For each model name in `model_names`, this method fetches that model's
        test inputs via `_get_preds_for_model()` and computes predictions. All
        returned label arrays must be present and identical (same length and
        values); otherwise a RuntimeError is raised. The final ensemble is the
        element-wise mean of model prediction arrays.
        """
        preds = []
        y_ref = None
        for name in model_names:
            m = next((mm for mm in self.models if mm.get('name') == name), None)
            if m is None:
                raise ValueError(f'Model {name} not found')
            y, p = self._get_preds_for_model(m)
            if y is None or p is None:
                raise RuntimeError(f'Missing predictions/labels for model {name}')
            if y_ref is None:
                y_ref = np.asarray(y)
            else:
                # ensure labels match exactly
                y_arr = np.asarray(y)
                if y_arr.shape != y_ref.shape or not np.array_equal(y_arr, y_ref):
                    raise RuntimeError('Models to be ensembled do not share the same label set; ensure they were evaluated on the same test samples')
            preds.append(np.asarray(p, dtype=float))
        if not preds:
            return np.zeros((0,), dtype=float)
        # stack and average
        stacked = np.vstack(preds)
        return np.mean(stacked, axis=0)

    def feature_importance_comparison(self, model_a: str, model_b: str, feature_names: List[str] = None):
        """Compare feature importances between two models that expose `feature_importances_`.

        Returns a DataFrame with columns ['feature','importance_a','importance_b','delta']
        """
        ma = next((m for m in self.models if m.get('name') == model_a), None)
        mb = next((m for m in self.models if m.get('name') == model_b), None)
        if ma is None or mb is None:
            raise ValueError('Model names not found')
        # try to load sklearn-style feature_importances_
        def _get_importances(m):
            path = m.get('path')
            try:
                # attempt to load model object if trainer has it
                if hasattr(m.get('predict_fn'), '__self__'):
                    obj = getattr(m['predict_fn'], '__self__')
                    if hasattr(obj, 'feature_importances_'):
                        return np.asarray(obj.feature_importances_, dtype=float)
                # fallback: try to load from disk using numpy
                # some XGB models store feature scores in a .json next to path
                return None
            except Exception:
                return None
        ia = _get_importances(ma)
        ib = _get_importances(mb)
        if ia is None or ib is None:
            raise RuntimeError('Feature importances not available for one or both models')
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(len(ia))]
        import pandas as _pd
        df = _pd.DataFrame({'feature': feature_names, 'importance_a': ia, 'importance_b': ib})
        df['delta'] = df['importance_a'] - df['importance_b']
        return df

    def save_failure_cases(self, out_dir: str, threshold: float = 0.5):
        """Save indices and inputs where models disagree or all fail.

        Produces a JSON file with disagreement indices and per-model predictions.
        """
        os.makedirs(out_dir, exist_ok=True)
        disagreement = {'indices': [], 'predictions': {}}
        # collect predictions for each model on its input
        for m in self.models:
            y, p = self._get_preds_for_model(m)
            if y is None:
                continue
            disagreement['predictions'][m['name']] = p.tolist()
            if 'ref_y' not in disagreement:
                disagreement['ref_y'] = y.tolist()
        # simple disagreement: indices where not all predictions agree on binary label
        if 'ref_y' in disagreement:
            preds_arr = np.vstack([np.asarray(v) for v in disagreement['predictions'].values()])
            bin_preds = (preds_arr >= threshold).astype(int)
            disagree_idx = np.where(np.any(bin_preds != bin_preds[0], axis=0))[0].tolist()
            disagreement['indices'] = disagree_idx
        # enrich with original features where possible
        try:
            # attempt to pull tabular features from any multimodal input available
            mm = self._any_input_of_type('multimodal')
            if mm is not None:
                Xc, Xs, Xt = mm[0]
                # if Xt is 2D, include feature rows for disagreement indices
                if isinstance(Xt, (list, tuple)):
                    Xt_arr = Xt[0]
                else:
                    Xt_arr = Xt
                import numpy as _np
                if 'indices' in disagreement and disagreement['indices']:
                    rows = _np.asarray(Xt_arr)[disagreement['indices']].tolist()
                    disagreement['failure_tabular_rows'] = rows
        except Exception:
            pass
        outp = os.path.join(out_dir, 'failure_cases.json')
        with open(outp, 'w') as fh:
            json.dump(disagreement, fh)
        return outp

    def export_report(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        df, roc_data, pr_data = self.compare_metrics()
        csv_path = os.path.join(out_dir, 'metrics_comparison.csv')
        df.to_csv(csv_path, index=False)

        # plots: ROC and PR
        plt.figure()
        for name, (fpr, tpr) in roc_data.items():
            plt.plot(fpr, tpr, label=name)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curves')
        plt.legend()
        roc_png = os.path.join(out_dir, 'roc_curves.png')
        plt.savefig(roc_png)
        plt.close()

        plt.figure()
        for name, (rec, prec) in pr_data.items():
            plt.plot(rec, prec, label=name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        pr_png = os.path.join(out_dir, 'pr_curves.png')
        plt.savefig(pr_png)
        plt.close()

        # measure size/latency for models with paths
        size_rows = []
        for m in self.models:
            inp = self._select_input_for_model(m)
            sample_input = None
            if inp is not None:
                X_test = inp[0]
                # take first sample, adapt for tflite interpreter (list for multimodal)
                if isinstance(X_test, tuple):
                    sample_input = [x[:1] for x in X_test]
                else:
                    sample_input = X_test[:1]
            info = self._measure_size_and_latency(m, sample_input)
            info['model'] = m.get('name')
            size_rows.append(info)
        size_df = pd.DataFrame(size_rows)
        size_csv = os.path.join(out_dir, 'size_latency.csv')
        size_df.to_csv(size_csv, index=False)

        # combined JSON report
        report = {
            'metrics_csv': os.path.basename(csv_path),
            'roc_png': os.path.basename(roc_png),
            'pr_png': os.path.basename(pr_png),
            'size_csv': os.path.basename(size_csv),
        }
        with open(os.path.join(out_dir, 'comparison_report.json'), 'w') as fh:
            json.dump(report, fh)

        # export markdown summary
        md = ['# Model Comparison Report', '']
        md.append('## Metrics')
        md.append(df.to_markdown(index=False))
        md.append('')
        md.append('## Size / Latency')
        md.append(size_df.to_markdown(index=False))
        with open(os.path.join(out_dir, 'comparison_report.md'), 'w') as fh:
            fh.write('\n'.join(md))

        return out_dir
