import numpy as np
import utils.sequence_utils as sequence_utils
from utils.sequence_utils import pad_or_truncate, create_sliding_windows, normalize_sequences, prepare_sequences_from_windows


def test_pad_or_truncate_pad():
    seq = np.array([[1.0], [2.0]])
    out = pad_or_truncate(seq, 4, padding_value=0.0)
    assert out.shape == (4, 1)
    assert out[-2, 0] == 1.0


def test_pad_or_truncate_truncate():
    seq = np.arange(10).reshape(10,1)
    out = pad_or_truncate(seq, 5)
    assert out.shape == (5,1)
    assert out[-1,0] == 9


def test_create_sliding_windows():
    seq = np.arange(6).reshape(6,1)
    windows = create_sliding_windows(seq, window=3, stride=1)
    assert windows.shape == (4,3,1)


def test_normalize_sequences_zscore():
    X = np.random.RandomState(0).randn(10,5,3)
    Xn = normalize_sequences(X, method='zscore')
    mean = Xn.mean()
    assert abs(mean) < 1e-6


def test_prepare_sequences_from_windows_empty():
    windows = [
        {'sensor_timeseries': {'a': [1,2,3]}, 'labels': {'is_accident': 0}, 'trip_id': 't1'},
        {'sensor_timeseries': {'a': []}, 'labels': {'is_accident': 1}, 'trip_id': 't2'},
    ]
    X, y, meta = prepare_sequences_from_windows(windows, seq_length=4, timeseries_key='sensor_timeseries')
    assert X.shape[0] == 2
    assert X.shape[1] == 4
    assert list(y) == [0,1]


def test_compute_apply_normalization_constant_and_empty():
    # constant input: std=0 -> apply_normalization should handle eps and produce zeros
    X = np.ones((4, 10, 3), dtype=float) * 5.0
    stats = sequence_utils.compute_normalization_stats(X, method='zscore')
    assert 'mean' in stats and 'std' in stats
    Xn = sequence_utils.apply_normalization(X, stats, method='zscore')
    assert (abs(Xn) < 1e-6).all()

    # empty input returns empty stats and unchanged array
    Xempty = np.zeros((0, 10, 0))
    stats2 = sequence_utils.compute_normalization_stats(Xempty, method='zscore')
    assert stats2 == {}
    Xn2 = sequence_utils.apply_normalization(Xempty, stats2, method='zscore')
    assert Xn2.shape[0] == 0


def test_sequence_data_generator_iteration_and_augmentation():
    X = np.random.randn(25, 10, 2)
    y = np.random.randint(0, 2, size=(25,))
    gen = sequence_utils.SequenceDataGenerator(X, y, batch_size=8, shuffle=False, augment=True, noise_std=0.1, time_jitter_std=0.0)
    batches = list(iter(gen))
    assert len(batches) == int(np.ceil(25 / 8))
    bx, by = batches[0]
    assert bx.shape[1:] == X.shape[1:]


def test_prepare_tabular_sequences_alignment_empty_callable():
    X, meta = sequence_utils.prepare_tabular_sequences([], feature_keys=['f1', 'f2'])
    assert X.shape[0] == 0
    assert isinstance(meta, list)
