"""Sequence preparation utilities for deep learning models.

Provides helpers to convert windowed sensor data into fixed-length
sequences for LSTM/CNN training, normalization, sliding windows,
and a simple data generator for augmentation.

These implementations are intentionally lightweight and dependency-free
(numpy only) so they can be imported in test environments. TensorFlow
`tf.data` helpers can be added on top when TF is available.
"""
from typing import List, Tuple, Optional
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def pad_or_truncate(seq: np.ndarray, seq_len: int, padding_value: float = 0.0) -> np.ndarray:
    """Pad or truncate a 2D sequence to `seq_len` along axis 0.

    Args:
        seq: array shape (T, C)
        seq_len: desired length T'
        padding_value: value used for padding rows when seq is shorter

    Returns:
        array shape (seq_len, C)
    """
    if seq is None:
        return np.full((seq_len, 0), padding_value)
    seq = np.asarray(seq)
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)
    T, C = seq.shape
    if T == seq_len:
        return seq
    if T > seq_len:
        return seq[-seq_len:]
    # pad at the beginning to preserve most recent timesteps at the end
    pad = np.full((seq_len - T, C), padding_value, dtype=seq.dtype)
    return np.vstack((pad, seq))


def create_sliding_windows(seq: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Create overlapping windows from a single sequence.

    Returns shape (n_windows, window, C).
    """
    seq = np.asarray(seq)
    if seq.ndim == 1:
        seq = seq.reshape(-1, 1)
    T, C = seq.shape
    if window <= 0:
        raise ValueError("window must be > 0")
    if T < window:
        return np.expand_dims(pad_or_truncate(seq, window), axis=0)
    n = 1 + (T - window) // stride
    out = np.empty((n, window, C), dtype=seq.dtype)
    idx = 0
    for start in range(0, T - window + 1, stride):
        out[idx] = seq[start:start + window]
        idx += 1
    return out


def normalize_sequences(seqs: np.ndarray, method: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    """Normalize sequences per-feature (backwards-compatible wrapper).

    For new workflows that need to compute and reuse normalization statistics,
    prefer `compute_normalization_stats` and `apply_normalization`.
    """
    seqs = np.asarray(seqs)
    if method == "none":
        return seqs
    # delegate to new helpers
    stats = compute_normalization_stats(seqs, method=method, eps=eps)
    return apply_normalization(seqs, stats, method=method, eps=eps)


def compute_normalization_stats(seqs: np.ndarray, method: str = "zscore", eps: float = 1e-8):
    """Compute normalization statistics for `seqs`.

    Returns a dict of stats appropriate for `apply_normalization`.
    """
    seqs = np.asarray(seqs).astype(np.float32)
    # handle empty input: return empty/no-op stats so callers can skip normalization
    if seqs.size == 0:
        return {}
    if method == "none":
        return {}
    if method == "zscore":
        mean = seqs.mean(axis=(0, 1), keepdims=True)
        std = seqs.std(axis=(0, 1), keepdims=True)
        return {'mean': mean, 'std': std}
    if method == "minmax":
        mn = seqs.min(axis=(0, 1), keepdims=True)
        mx = seqs.max(axis=(0, 1), keepdims=True)
        return {'min': mn, 'max': mx}
    raise ValueError(f"Unknown normalization method: {method}")


def apply_normalization(seqs: np.ndarray, stats: dict, method: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    """Apply previously computed normalization stats to `seqs`.

    `stats` should be the dict returned by `compute_normalization_stats`.
    """
    seqs = np.asarray(seqs).astype(np.float32)
    if not stats or method == 'none':
        return seqs
    if method == 'zscore':
        mean = stats.get('mean')
        std = stats.get('std')
        if mean is None or std is None:
            raise ValueError('zscore stats must contain mean and std')
        return (seqs - mean) / (std + eps)
    if method == 'minmax':
        mn = stats.get('min')
        mx = stats.get('max')
        if mn is None or mx is None:
            raise ValueError('minmax stats must contain min and max')
        return (seqs - mn) / (mx - mn + eps)
    raise ValueError(f"Unknown normalization method: {method}")


def prepare_sequences_from_windows(windows: List[dict],
                                    seq_length: int = 10,
                                    padding_value: float = 0.0,
                                    feature_keys: Optional[List[str]] = None,
                                    normalization: str = "zscore",
                                    horizon: Optional[int] = None,
                                    timeseries_key: str = 'sensor_timeseries',
                                    stride: int = 1,
                                    use_sliding: bool = True) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Convert labeled window dicts into sequence arrays and labels.

    Expected window dict structure (flexible):
        {
            'sensor_timeseries' or custom timeseries_key: {'gps_speed': [...], 'imu_ax': [...], ...},
            'features': {...},
            'labels': {'is_accident_h10': 0}
            'trip_id': '...'
        }

        Notes:
                - By default this helper produces overlapping sliding subsequences from
                    each input window (i.e., multiple subsequences per window) controlled
                    by the `stride` parameter. The default behavior (`use_sliding=True`)
                    yields all sliding subsequences of length `seq_length`.
                - To obtain exactly one sequence per input window (useful for
                    multimodal alignment), pass `use_sliding=False`; this will return
                    a single subsequence per window by padding/truncating to the
                    requested `seq_length` and preserving the most recent timesteps.

    Args:
        windows: list of labeled window dicts
        seq_length: desired sequence length
        padding_value: padding value for short sequences
        feature_keys: list of sensor keys to extract and concatenate channels in order
        normalization: normalization method
        horizon: horizon integer to select label key 'is_accident_h{horizon}' if provided
        timeseries_key: key in each window where time-series dict is stored

    Returns:
        X: np.ndarray (N, seq_length, C)
        y: np.ndarray (N,)
        meta: list of dicts (one per sample) with original window and trip_id
    """
    # Determine global ordered keys if not explicitly provided. This avoids
    # per-window channel mismatches when windows have heterogeneous key sets.
    if feature_keys is None:
        union_keys = set()
        for w in windows:
            ts = w.get(timeseries_key, {}) or {}
            union_keys.update(ts.keys())
        keys_ordered = sorted(list(union_keys))
    else:
        keys_ordered = list(feature_keys)

    seqs = []
    ys = []
    meta = []
    for w in windows:
        ts = w.get(timeseries_key, {}) or {}
        # collect feature channels in the global order; missing keys become empty arrays
        channels = []
        for k in keys_ordered:
            if k in ts:
                arr = np.asarray(ts.get(k, []), dtype=float)
            else:
                # missing key: treat as an empty 1D array so padding yields correct column
                arr = np.asarray([], dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            channels.append(arr)

        if len(channels) == 0:
            # extreme case: no keys across all windows -> produce an empty-channel time-series
            data = np.zeros((0, 0))
        else:
            # compute per-window max time length and pad/truncate each channel to that length
            lengths = [c.shape[0] for c in channels]
            max_len = max(lengths) if lengths else 0
            padded = [pad_or_truncate(c, max_len, padding_value) if c.shape[0] != max_len else c for c in channels]
            # horizontally concatenate padded channels to shape (T, C)
            if padded:
                data = np.concatenate(padded, axis=1)
            else:
                data = np.zeros((max_len, 0))

        # guard against zero-channel sequences: skip these windows to avoid downstream shape errors
        if data.ndim == 2 and data.shape[1] == 0:
            LOGGER.warning('Window %s produced zero-channel time-series; skipping this window', w.get('trip_id', '<unknown>'))
            continue
        # create subsequences for this window
        if use_sliding:
            subseqs = create_sliding_windows(data, window=seq_length, stride=stride)
        else:
            # For multimodal or alignment-sensitive usage we may want exactly one
            # sequence per window: take the most recent seq_length timesteps by
            # padding/truncating the raw time-series to seq_length.
            s = pad_or_truncate(data, seq_length, padding_value)
            subseqs = np.expand_dims(s, axis=0)
        label_key = f'is_accident_h{horizon}' if horizon is not None else 'is_accident'
        lbl = w.get('labels', {}).get(label_key, 0)
        for s in subseqs:
            seqs.append(s)
            ys.append(lbl)
            meta.append({'trip_id': w.get('trip_id'), 'window': w})
    # stack sequences into array (N, seq_length, C). If C==0 this yields shape (N, seq_length, 0).
    if len(seqs) == 0:
        # no subsequences produced: return empty arrays consistent with expected shapes
        X = np.zeros((0, seq_length, 0))
    else:
        X = np.stack(seqs, axis=0)
    # preserve backwards-compatible behavior: if normalization requested, apply it here
    if normalization and normalization != 'none':
        X = normalize_sequences(X, method=normalization)
    y = np.asarray(ys, dtype=int)
    return X, y, meta


class SequenceDataGenerator:
    """Simple numpy-based data generator for sequence models.

    Supports basic augmentation: additive gaussian noise and time jitter.
    """
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 augment: bool = False,
                 noise_std: float = 0.0,
                 time_jitter_std: float = 0.0):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.noise_std = noise_std
        self.time_jitter_std = time_jitter_std
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, len(self.X), self.batch_size):
            idx = self.indices[start:start + self.batch_size]
            batch_x = self.X[idx].copy()
            if self.augment:
                if self.noise_std > 0:
                    batch_x += np.random.normal(scale=self.noise_std, size=batch_x.shape)
                # time jitter: small circular shifts per channel
                if self.time_jitter_std > 0:
                    shifts = np.round(np.random.normal(scale=self.time_jitter_std, size=(len(idx),))).astype(int)
                    for i, s in enumerate(shifts):
                        if s == 0:
                            continue
                        batch_x[i] = np.roll(batch_x[i], shift=s, axis=0)
            yield batch_x, self.y[idx]


__all__ = [
    'pad_or_truncate', 'create_sliding_windows', 'normalize_sequences',
    'prepare_sequences_from_windows', 'SequenceDataGenerator',
    'compute_normalization_stats', 'apply_normalization'
]


def prepare_tabular_sequences(windows, feature_keys=None):
    """Extract tabular features matrix from a list of window dicts.

    Returns a tuple (X_tabular, meta) where X_tabular is a NumPy array
    shape (N, C) and meta is a list of metadata dicts per row.
    """
    import numpy as _np
    rows = []
    meta = []
    for w in windows:
        features = w.get('features', {}) or {}
        if feature_keys is None:
            # include all scalar features found
            vals = [v for k, v in features.items() if not isinstance(v, (list, tuple, dict))]
        else:
            vals = [features.get(k, 0.0) for k in feature_keys]
        rows.append(vals)
        meta.append({'trip_id': w.get('trip_id'), 'window': w})
    if len(rows) == 0:
        return _np.zeros((0, 0)), meta
    X = _np.asarray(rows, dtype=float)
    return X, meta
