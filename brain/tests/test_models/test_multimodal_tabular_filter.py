import pandas as pd


def test_tabular_feature_filtering_excludes_labels():
    # construct a dataframe mimicking FeatureExtractor output
    df = pd.DataFrame({
        'window_id': [1, 2],
        'trip_id': ['t1', 't2'],
        'label_is_accident': [0, 1],
        'is_accident_h10': [0, 0],
        'feature_a': [0.1, 0.2],
        'feature_b': [1.0, 2.0],
    })

    # emulate the filtering logic used by MultimodalTrainer
    all_cols = list(df.columns)

    def is_label_col(c: str) -> bool:
        if not isinstance(c, str):
            return False
        if c == 'is_accident' or c.startswith('label_') or c.startswith('is_accident_h'):
            return True
        if c in ('window_id', 'trip_id', 'window_timestamp'):
            return True
        return False

    tab_cols = [c for c in all_cols if not is_label_col(c)]
    assert 'feature_a' in tab_cols and 'feature_b' in tab_cols
    assert 'label_is_accident' not in tab_cols
    assert 'is_accident_h10' not in tab_cols
    assert 'window_id' not in tab_cols
