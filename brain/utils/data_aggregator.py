"""DataAggregator: prepare retraining datasets from telemetry logs."""
from __future__ import annotations

import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

import pandas as pd

from config.config_loader import ConfigLoader
from feature_engineering.extractor import FeatureExtractor
from utils import encryption
import logging

logger = logging.getLogger(__name__)


class DataAggregator:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.output_dir = str(self.config.get('retraining.output_dir', 'data/retraining'))
        self.balancing_strategy = str(self.config.get('retraining.balancing_strategy', 'smote'))
        self.min_samples_per_class = int(self.config.get('retraining.min_samples_per_class', 100))
        self.include_near_misses = bool(self.config.get('retraining.include_near_misses', True))
        self.validation_split = float(self.config.get('retraining.validation_split', 0.2))
        os.makedirs(self.output_dir, exist_ok=True)
        self.extractor = FeatureExtractor(config)
        self.last_ts_path = os.path.join('data', 'telemetry', 'last_aggregation.json')

    def _load_last_ts(self) -> Optional[str]:
        try:
            if os.path.exists(self.last_ts_path):
                with open(self.last_ts_path, 'r', encoding='utf-8') as fh:
                    j = json.load(fh)
                    return j.get('last_ts')
        except Exception:
            return None
        return None

    def _save_last_ts(self, ts: str) -> None:
        try:
            with open(self.last_ts_path, 'w', encoding='utf-8') as fh:
                json.dump({'last_ts': ts}, fh)
        except Exception:
            pass

    def load_telemetry_logs(self, telemetry_dir: str, start_date: Optional[str] = None, end_date: Optional[str] = None, incremental: bool = False) -> List[Dict[str, Any]]:
        files = [f for f in os.listdir(telemetry_dir) if f.startswith('telemetry_') and (f.endswith('.jsonl') or f.endswith('.jsonl.enc'))]
        all_entries: List[Dict[str, Any]] = []
        last_ts = None
        if incremental:
            last_ts = self._load_last_ts()
        for fn in files:
            path = os.path.join(telemetry_dir, fn)
            try:
                if fn.endswith('.enc'):
                    tmp = path + '.tmp'
                    key = self._get_key()
                    if not key:
                        continue
                    try:
                        encryption.decrypt_file(path, tmp, key)
                        src = tmp
                    except Exception:
                        continue
                else:
                    src = path
                with open(src, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        try:
                            e = json.loads(line)
                            ts = e.get('timestamp')
                            if ts and (start_date or end_date or last_ts):
                                try:
                                    dt = datetime.fromisoformat(ts)
                                except Exception:
                                    try:
                                        dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')
                                    except Exception:
                                        dt = None
                                if dt:
                                    if start_date and dt < datetime.fromisoformat(start_date):
                                        continue
                                    if end_date and dt > datetime.fromisoformat(end_date):
                                        continue
                                    if last_ts:
                                        try:
                                            if dt <= datetime.fromisoformat(last_ts):
                                                continue
                                        except Exception:
                                            pass
                            all_entries.append(e)
                        except Exception:
                            continue
            finally:
                if fn.endswith('.enc') and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
        return all_entries

    def _get_key(self) -> Optional[bytes]:
        env = str(self.config.get('telemetry.encryption_key_env', 'TELEMETRY_ENCRYPTION_KEY'))
        val = os.environ.get(env)
        if val:
            try:
                return bytes.fromhex(val)
            except Exception:
                pass
            try:
                import base64

                return base64.b64decode(val)
            except Exception:
                pass
        key_file = self.config.get('telemetry.encryption_key_file', None)
        if key_file and os.path.exists(str(key_file)):
            try:
                with open(str(key_file), 'rb') as fh:
                    data = fh.read().strip()
                    try:
                        return bytes.fromhex(data.decode('utf-8'))
                    except Exception:
                        pass
                    try:
                        import base64

                        return base64.b64decode(data)
                    except Exception:
                        pass
            except Exception:
                return None
        return None

    def match_predictions_to_outcomes(self, all_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        predictions = [e for e in all_entries if e.get('type') == 'prediction']
        outcomes = [e for e in all_entries if e.get('type') == 'outcome']
        false_alarms = [e for e in all_entries if e.get('type') == 'false_alarm']
        matched: List[Dict[str, Any]] = []
        for p in predictions:
            found = False
            for o in outcomes:
                try:
                    if o.get('prediction_id') and o.get('prediction_id') == p.get('prediction_id'):
                        p['actual_outcome'] = o.get('actual_outcome')
                        p['time_to_event_s'] = o.get('time_to_event_s')
                        p['severity'] = o.get('severity')
                        found = True
                        break
                    # temporal proximity match
                    pts = p.get('timestamp')
                    ots = o.get('timestamp')
                    if pts and ots:
                        try:
                            pdt = datetime.fromisoformat(pts)
                        except Exception:
                            try:
                                pdt = datetime.strptime(pts, '%Y-%m-%dT%H:%M:%SZ')
                            except Exception:
                                pdt = None
                        try:
                            odt = datetime.fromisoformat(ots)
                        except Exception:
                            try:
                                odt = datetime.strptime(ots, '%Y-%m-%dT%H:%M:%SZ')
                            except Exception:
                                odt = None
                        if pdt and odt and abs((odt - pdt).total_seconds()) < 10:
                            p['actual_outcome'] = o.get('actual_outcome')
                            p['time_to_event_s'] = o.get('time_to_event_s')
                            p['severity'] = o.get('severity')
                            found = True
                            break
                except Exception:
                    continue
            if not found:
                p['label'] = 0
            else:
                ao = p.get('actual_outcome')
                if ao in ['near_miss'] and self.include_near_misses:
                    p['label'] = 1
                elif ao == 'accident':
                    p['label'] = 1
                else:
                    p['label'] = 0
            matched.append(p)

        # attach false alarm flags
        for fa in false_alarms:
            for m in matched:
                try:
                    if fa.get('prediction_id') and fa.get('prediction_id') == m.get('prediction_id'):
                        m['false_alarm'] = True
                        m['label'] = 0
                        break
                except Exception:
                    continue

        return matched

    def extract_features_and_labels(self, matched: List[Dict[str, Any]]) -> pd.DataFrame:
        features_list = []
        labels = []
        for m in matched:
            if m.get('label') is None:
                continue
            try:
                # build a minimal window for extractor; extractor should tolerate missing fields
                window = {'features': m.get('features', {}), 'metadata': m.get('metadata', {}), 'camera': [], 'gps': [], 'imu': [], 'vehicle_controls': []}
                feat = self.extractor.extract(window)
                features_list.append(feat)
                labels.append(int(m.get('label')))
            except Exception:
                continue
        if not features_list:
            return pd.DataFrame()
        df = pd.DataFrame(features_list)
        df['label'] = labels
        return df

    def apply_balancing(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if self.balancing_strategy == 'smote':
            try:
                from imblearn.over_sampling import SMOTE

                X = df.drop(columns=['label'])
                y = df['label']
                if sum(y == 1) < 2:
                    return df
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X, y)
                df_res = pd.DataFrame(X_res, columns=X.columns)
                df_res['label'] = y_res
                return df_res
            except Exception:
                return df
        elif self.balancing_strategy == 'undersample':
            try:
                from imblearn.under_sampling import RandomUnderSampler

                X = df.drop(columns=['label'])
                y = df['label']
                rus = RandomUnderSampler(random_state=42)
                X_res, y_res = rus.fit_resample(X, y)
                df_res = pd.DataFrame(X_res, columns=X.columns)
                df_res['label'] = y_res
                return df_res
            except Exception:
                return df
        else:
            return df

    def aggregate_for_retraining(self, telemetry_dir: str, output_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None, incremental: bool = False) -> str:
        logs = self.load_telemetry_logs(telemetry_dir, start_date, end_date, incremental)
        matched = self.match_predictions_to_outcomes(logs)
        df = self.extract_features_and_labels(matched)
        if df.empty or len(df) < self.min_samples_per_class:
            raise ValueError(f'Insufficient samples: {len(df)}')
        df_bal = self.apply_balancing(df)
        if self.validation_split > 0:
            from sklearn.model_selection import train_test_split

            train, val = train_test_split(df_bal, test_size=self.validation_split, random_state=42, stratify=df_bal['label'])
            train_path = output_path.replace('.csv', '_train.csv')
            val_path = output_path.replace('.csv', '_val.csv')
            train.to_csv(train_path, index=False)
            val.to_csv(val_path, index=False)
            self._save_last_ts(datetime.now(timezone.utc).isoformat())
            return train_path
        else:
            df_bal.to_csv(output_path, index=False)
            self._save_last_ts(datetime.now(timezone.utc).isoformat())
            return output_path

    def generate_retraining_report(self, df: pd.DataFrame, output_dir: str) -> str:
        stats = {
            'total_samples': len(df),
            'positive_ratio': float(df['label'].mean()) if 'label' in df else None,
            'class_distribution': df['label'].value_counts().to_dict() if 'label' in df else {},
            'features': [c for c in df.columns if c != 'label'],
            'temporal_coverage': {'start': None, 'end': None},
        }
        report_path = os.path.join(output_dir, f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(report_path, 'w', encoding='utf-8') as fh:
                json.dump(stats, fh, default=str)
        except Exception:
            pass
        return report_path
