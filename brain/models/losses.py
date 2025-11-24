"""Custom loss functions for XGBoost (focal loss) and helpers.
"""
from typing import Tuple, Dict
import numpy as np

def focal_loss_objective(preds: np.ndarray, dtrain, alpha: float = 0.25, gamma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Focal loss for binary classification as XGBoost custom objective.

    preds: raw scores (logits) from XGBoost
    dtrain: DMatrix with labels
    returns: grad, hess arrays
    """
    labels = dtrain.get_label()
    # transform logits to probability
    preds_prob = 1.0 / (1.0 + np.exp(-preds))
    eps = 1e-9
    preds_prob = np.clip(preds_prob, eps, 1.0 - eps)

    p_t = preds_prob * labels + (1 - preds_prob) * (1 - labels)
    # focal weight
    focal_weight = alpha * ((1 - p_t) ** gamma)

    # gradient and hessian
    grad = (preds_prob - labels) * focal_weight
    # approximate hessian
    hess = focal_weight * preds_prob * (1.0 - preds_prob)
    # stability
    grad = np.nan_to_num(grad)
    hess = np.nan_to_num(hess) + 1e-6
    return grad, hess


def focal_loss_eval(preds: np.ndarray, dtrain, alpha: float = 0.25, gamma: float = 2.0) -> Tuple[str, float]:
    labels = dtrain.get_label()
    preds_prob = 1.0 / (1.0 + np.exp(-preds))
    eps = 1e-9
    preds_prob = np.clip(preds_prob, eps, 1.0 - eps)
    p_t = preds_prob * labels + (1 - preds_prob) * (1 - labels)
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)
    return 'focal_loss', float(np.mean(loss))


def get_focal_loss_params(config) -> Dict[str, float]:
    cfg = config.get('training', {}).get('xgboost', {})
    return {
        'alpha': float(cfg.get('focal_loss_alpha', 0.25)),
        'gamma': float(cfg.get('focal_loss_gamma', 2.0)),
    }


def focal_loss_keras(alpha: float = 0.25, gamma: float = 2.0):
    """Return a Keras-compatible focal loss function for binary classification.

    Usage: model.compile(loss=focal_loss_keras(alpha=..., gamma=...))
    """
    import tensorflow as _tf

    def loss(y_true, y_pred):
        eps = _tf.keras.backend.epsilon()
        y_pred = _tf.clip_by_value(y_pred, eps, 1.0 - eps)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_factor = alpha * _tf.pow(1.0 - p_t, gamma)
        ce = - (y_true * _tf.math.log(y_pred) + (1 - y_true) * _tf.math.log(1 - y_pred))
        return _tf.reduce_mean(focal_factor * ce)

    return loss


def custom_objects_registry():
    """Return a dict of custom objects that Keras may need when loading models.

    This helps `compare_models` map known custom losses to callable objects.
    """
    return {
        'focal_loss_keras': focal_loss_keras,
    }
