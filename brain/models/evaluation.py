"""Evaluation utilities for binary classification models.

Provides metric computation and plotting helpers used by the training pipeline.
"""
from typing import Dict, Any, List, Optional
import os
import json

import numpy as np

try:
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        roc_auc_score,
        roc_curve,
        brier_score_loss,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
except Exception:
    # Lightweight fallbacks for environments without scikit-learn (tests)
    def precision_recall_curve(y_true, probas_pred):
        # simple implementation: sort by probas and accumulate
        import numpy as _np

        desc = _np.argsort(-probas_pred)
        y_sorted = _np.asarray(y_true)[desc]
        tp = _np.cumsum(y_sorted == 1)
        fp = _np.cumsum(y_sorted == 0)
        recall = tp / max(1, tp[-1]) if tp[-1] > 0 else _np.zeros_like(tp, dtype=float)
        precision = tp / (tp + fp + 1e-8)
        return precision, recall, _np.arange(len(probas_pred))

    def average_precision_score(y_true, probas_pred):
        import numpy as _np

        p, r, _ = precision_recall_curve(y_true, probas_pred)
        # approximate AP via trapezoid on recall vs precision
        return float(_np.trapz(p[::-1], r[::-1])) if len(p) > 1 else float(p[0] if len(p) else 0.0)

    def roc_auc_score(y_true, probas_pred):
        import numpy as _np

        # simple AUC implementation via trapezoid
        desc = _np.argsort(probas_pred)
        y_sorted = _np.asarray(y_true)[desc]
        fpr = _np.cumsum(y_sorted == 0) / max(1, (_np.sum(y_sorted == 0)))
        tpr = _np.cumsum(y_sorted == 1) / max(1, (_np.sum(y_sorted == 1)))
        return float(_np.trapz(tpr, fpr))

    def roc_curve(y_true, probas_pred):
        import numpy as _np

        # approximate by sorting
        desc = _np.argsort(-probas_pred)
        y_sorted = _np.asarray(y_true)[desc]
        tp = _np.cumsum(y_sorted == 1)
        fp = _np.cumsum(y_sorted == 0)
        tpr = tp / max(1, tp[-1]) if tp[-1] > 0 else _np.zeros_like(tp, dtype=float)
        fpr = fp / max(1, fp[-1]) if fp[-1] > 0 else _np.zeros_like(fp, dtype=float)
        return fpr, tpr, _np.arange(len(probas_pred))

    def brier_score_loss(y_true, probas_pred):
        import numpy as _np

        y = _np.asarray(y_true, dtype=float)
        p = _np.asarray(probas_pred, dtype=float)
        return float(_np.mean((y - p) ** 2))

    def precision_score(y_true, y_pred, zero_division=0):
        import numpy as _np

        tp = int(_np.sum((_np.asarray(y_true) == 1) & (_np.asarray(y_pred) == 1)))
        fp = int(_np.sum((_np.asarray(y_true) == 0) & (_np.asarray(y_pred) == 1)))
        denom = tp + fp
        if denom == 0:
            return float(zero_division)
        return float(tp / denom)

    def recall_score(y_true, y_pred, zero_division=0):
        import numpy as _np

        tp = int(_np.sum((_np.asarray(y_true) == 1) & (_np.asarray(y_pred) == 1)))
        fn = int(_np.sum((_np.asarray(y_true) == 1) & (_np.asarray(y_pred) == 0)))
        denom = tp + fn
        if denom == 0:
            return float(zero_division)
        return float(tp / denom)

    def f1_score(y_true, y_pred, zero_division=0):
        p = precision_score(y_true, y_pred, zero_division=zero_division)
        r = recall_score(y_true, y_pred, zero_division=zero_division)
        if p + r == 0:
            return float(zero_division)
        return 2 * (p * r) / (p + r)

    def confusion_matrix(y_true, y_pred):
        import numpy as _np

        y_true = _np.asarray(y_true)
        y_pred = _np.asarray(y_pred)
        tn = int(_np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(_np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(_np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(_np.sum((y_true == 1) & (y_pred == 1)))
        return _np.array([[tn, fp], [fn, tp]])

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Basic checks
    if len(y_true) == 0:
        return {}

    out['pr_auc'] = float(average_precision_score(y_true, y_pred_proba))
    try:
        out['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        out['roc_auc'] = None

    out['brier_score'] = float(brier_score_loss(y_true, y_pred_proba))

    if thresholds is None:
        thresholds = [0.1, 0.3, 0.6]

    thr_metrics = {}
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        balanced_acc = float((tpr + specificity) / 2.0)
        thr_metrics[str(t)] = {
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'specificity': specificity,
            'npv': npv,
            'balanced_accuracy': balanced_acc,
        }

    out['thresholds'] = thr_metrics
    return out


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str):
    if plt is None:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve (AP={ap:.3f})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str):
    if plt is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str, n_bins: int = 10):
    try:
        from sklearn.calibration import calibration_curve
    except Exception:
        return
    if plt is None:
        return
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, labels: Optional[List[str]] = None):
    if plt is None:
        return
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        if sns is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels or ['neg', 'pos'], yticklabels=labels or ['neg', 'pos'])
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            for (i, j), val in np.ndenumerate(cm):
                plt.text(j, i, int(val), ha='center', va='center')
            plt.xticks([0, 1], labels or ['neg', 'pos'])
            plt.yticks([0, 1], labels or ['neg', 'pos'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
    except Exception:
        return


def save_metrics_report(metrics: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)


def generate_evaluation_report(y_true: np.ndarray, y_pred_proba: np.ndarray, output_dir: str, model_name: str = 'model', thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred_proba, thresholds)
    save_metrics_report(metrics, os.path.join(output_dir, 'metrics.json'))
    # plots
    plot_precision_recall_curve(y_true, y_pred_proba, os.path.join(output_dir, 'pr_curve.png'))
    plot_roc_curve(y_true, y_pred_proba, os.path.join(output_dir, 'roc_curve.png'))
    plot_calibration_curve(y_true, y_pred_proba, os.path.join(output_dir, 'calibration_curve.png'))
    # Confusion matrix at a sensible default threshold (middle threshold if present)
    thr_to_use = 0.3
    if thresholds:
        try:
            thr_to_use = thresholds[len(thresholds) // 2]
        except Exception:
            thr_to_use = thresholds[0]
    y_pred_bin = (y_pred_proba >= thr_to_use).astype(int)
    plot_confusion_matrix(y_true, y_pred_bin, os.path.join(output_dir, 'confusion_matrix.png'))
    return metrics
