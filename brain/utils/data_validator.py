import hashlib
import json
import logging
from typing import Optional, Dict

import pandas as pd

# SciPy optional import: provide fallbacks when not available
try:
    from scipy import stats  # type: ignore
    HAVE_SCIPY = True
except Exception:
    stats = None
    HAVE_SCIPY = False

logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.min_samples = cfg.get('min_samples', 200)
        self.max_missing_ratio = cfg.get('max_missing_ratio', 0.2)
        self.drift_threshold = cfg.get('drift_threshold', 0.3)
        self.label_balance_min_ratio = cfg.get('label_balance_min_ratio', 0.01)
        self.min_temporal_coverage_days = cfg.get('min_temporal_coverage_days', 7)

    def _dataset_hash(self, df: pd.DataFrame) -> str:
        # lightweight hash of column names + row count + sample of data
        m = hashlib.sha256()
        m.update(str(list(df.columns)).encode('utf-8'))
        m.update(str(len(df)).encode('utf-8'))
        sample = df.head(10).to_csv(index=False).encode('utf-8')
        m.update(sample)
        return m.hexdigest()

    def validate_schema(self, df: pd.DataFrame, required_columns) -> Dict:
        missing = [c for c in required_columns if c not in df.columns]
        return {'ok': len(missing) == 0, 'missing_columns': missing}

    def check_feature_distribution_drift(self, df_new: pd.DataFrame, df_baseline: pd.DataFrame) -> Dict:
        if df_baseline is None or df_baseline.empty:
            return {'skipped': True, 'reason': 'no_baseline'}
        drift = {}
        numeric_cols = df_new.select_dtypes(include=['number']).columns
        for c in numeric_cols:
            try:
                if HAVE_SCIPY and hasattr(stats, 'ks_2samp'):
                    stat, p = stats.ks_2samp(df_baseline[c].dropna(), df_new[c].dropna())
                    drift[c] = {'ks_stat': float(stat), 'p_value': float(p), 'drift': p < self.drift_threshold, 'scipy': True}
                else:
                    # SciPy not available: approximate drift with difference of means and stds
                    b = df_baseline[c].dropna()
                    n = df_new[c].dropna()
                    mean_diff = float(n.mean() - b.mean()) if not n.empty and not b.empty else 0.0
                    std_diff = float(n.std() - b.std()) if not n.empty and not b.empty else 0.0
                    approx_stat = abs(mean_diff)
                    drift_flag = abs(mean_diff) > (self.drift_threshold * (abs(b.mean()) + 1e-6))
                    drift[c] = {'approx_stat': approx_stat, 'mean_diff': mean_diff, 'std_diff': std_diff, 'drift': drift_flag, 'scipy': False}
            except Exception:
                drift[c] = {'error': 'failed_stat'}
        return {'drift': drift}

    def check_label_balance(self, df: pd.DataFrame, label_col: str = 'label') -> Dict:
        if label_col not in df.columns:
            return {'ok': False, 'reason': 'no_label_column'}
        counts = df[label_col].value_counts(dropna=True).to_dict()
        total = len(df)
        pos = counts.get(1, counts.get('1', 0) )
        pos_ratio = pos / total if total else 0.0
        return {'ok': pos_ratio >= self.label_balance_min_ratio, 'positive_ratio': pos_ratio, 'counts': counts}

    def check_temporal_coverage(self, df: pd.DataFrame, ts_col: str = 'timestamp') -> Dict:
        if ts_col not in df.columns:
            return {'ok': False, 'reason': 'no_timestamp'}
        try:
            s = pd.to_datetime(df[ts_col], unit='s')
            span_days = (s.max() - s.min()).days
            return {'ok': span_days >= self.min_temporal_coverage_days, 'span_days': int(span_days)}
        except Exception:
            return {'ok': False, 'reason': 'bad_timestamp'}

    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        missing = df.isnull().mean().to_dict()
        too_many = {k: v for k, v in missing.items() if v > self.max_missing_ratio}
        dup = df.duplicated().sum()
        return {'missing_ratio': missing, 'features_exceeding_missing_ratio': too_many, 'duplicates': int(dup)}

    def generate_validation_report(self, df: pd.DataFrame, baseline_df: Optional[pd.DataFrame] = None, required_columns=None) -> Dict:
        required_columns = required_columns or []
        report = {'passed': True, 'checks': {}}
        if len(df) < self.min_samples:
            report['checks']['min_samples'] = {'ok': False, 'samples': len(df), 'required': self.min_samples}
            report['passed'] = False
        else:
            report['checks']['min_samples'] = {'ok': True, 'samples': len(df)}

        schema = self.validate_schema(df, required_columns)
        report['checks']['schema'] = schema
        if not schema.get('ok'):
            report['passed'] = False

        dq = self.check_data_quality(df)
        report['checks']['data_quality'] = dq
        if dq.get('features_exceeding_missing_ratio'):
            report['passed'] = False

        lb = self.check_label_balance(df)
        report['checks']['label_balance'] = lb
        if not lb.get('ok'):
            report['passed'] = False

        tc = self.check_temporal_coverage(df)
        report['checks']['temporal_coverage'] = tc
        if not tc.get('ok'):
            report['passed'] = False

        drift = self.check_feature_distribution_drift(df, baseline_df)
        report['checks']['drift'] = drift
        # don't fail on drift by default; only warn
        report['dataset_hash'] = self._dataset_hash(df)
        return report
