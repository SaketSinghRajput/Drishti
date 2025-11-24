"""Cloud sync manager: aggregate telemetry summaries and upload to remote endpoint."""
from __future__ import annotations

import requests
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from config.config_loader import ConfigLoader
from utils import encryption


class CloudSyncManager:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.endpoint: str = str(self.config.get('telemetry.cloud_sync.endpoint_url', ''))
        self.api_key: str = str(self.config.get('telemetry.cloud_sync.api_key', ''))
        self.enabled: bool = bool(self.config.get('telemetry.cloud_sync.enabled', False))
        self.interval_hours: int = int(self.config.get('telemetry.cloud_sync.sync_interval_hours', 24))
        self.local_dir: str = str(self.config.get('telemetry.output_dir', 'data/telemetry'))
        self.retry_attempts: int = int(self.config.get('telemetry.cloud_sync.retry_attempts', 3))
        self.retry_backoff_s: int = int(self.config.get('telemetry.cloud_sync.retry_backoff_s', 60))
        os.makedirs(self.local_dir, exist_ok=True)

    def get_pending_files(self) -> List[str]:
        files: List[str] = []
        for fn in os.listdir(self.local_dir):
            if not fn.startswith('telemetry_'):
                continue
            if fn.endswith('.synced'):
                continue
            if fn.endswith('.jsonl') or fn.endswith('.jsonl.enc'):
                files.append(os.path.join(self.local_dir, fn))
        return sorted(files)

    def _read_file_lines(self, path: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        try:
            if path.endswith('.enc'):
                # decrypt to temp
                tmp = path + '.decrypted.tmp'
                key = None
                # try to obtain key via telemetry config keys
                key = self._get_key()
                if not key:
                    return []
                try:
                    encryption.decrypt_file(path, tmp, key)
                    src = tmp
                except Exception:
                    return []
            else:
                src = path

            with open(src, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        continue
        finally:
            if path.endswith('.enc') and 'tmp' in locals() and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        return entries

    def prepare_summary(self, files: List[str], period: str = 'daily') -> Dict[str, Any]:
        all_entries: List[Dict[str, Any]] = []
        for f in files:
            try:
                entries = self._read_file_lines(f)
                all_entries.extend(entries)
            except Exception:
                continue

        total_predictions = sum(1 for e in all_entries if e.get('type') == 'prediction')
        accident_count = sum(1 for e in all_entries if e.get('actual_outcome') == 'accident' or e.get('outcome') == 'accident')
        false_alarm_count = sum(1 for e in all_entries if e.get('type') == 'false_alarm' or e.get('false_alarm'))
        avg_risk_score = 0.0
        if total_predictions:
            avg_risk_score = sum(float(e.get('risk_score', 0) or 0) for e in all_entries if e.get('type') == 'prediction') / total_predictions

        timestamps = [e.get('timestamp') for e in all_entries if e.get('timestamp')]
        period_start = min(timestamps) if timestamps else None
        period_end = max(timestamps) if timestamps else None

        # sensor_health aggregation: simple approach - merge keys and take latest values
        sensor_health_summary: Dict[str, Any] = {}
        for e in all_entries:
            sh = e.get('sensor_health') or {}
            for k, v in sh.items():
                sensor_health_summary[k] = v

        # anonymize: remove PII-like fields
        for e in all_entries:
            if 'prediction_id' in e:
                e.pop('prediction_id', None)
            if 'metadata' in e and isinstance(e['metadata'], dict):
                # remove ids but keep non-PII
                e['metadata'].pop('vehicle_hash', None)
                e['metadata'].pop('trip_id', None)

        summary = {
            'period': period,
            'period_start': period_start,
            'period_end': period_end,
            'total_predictions': total_predictions,
            'accident_count': accident_count,
            'false_alarm_count': false_alarm_count,
            'avg_risk_score': avg_risk_score,
            'sensor_health_summary': sensor_health_summary,
            'file_count': len(files),
            'generated_at': datetime.utcnow().isoformat(),
        }
        return summary

    def _get_key(self) -> Optional[bytes]:
        # leverage same key resolution as telemetry
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
        if key_file:
            try:
                with open(str(key_file), 'rb') as f:
                    data = f.read().strip()
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

    def upload_summary(self, summary: Dict[str, Any]) -> bool:
        if not self.endpoint:
            return False
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        for attempt in range(max(1, self.retry_attempts)):
            try:
                resp = requests.post(self.endpoint, json=summary, headers=headers, timeout=30)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep((2 ** attempt) * 1)
        return False

    def sync(self) -> Dict[str, Any]:
        if not self.enabled:
            return {'status': 'disabled', 'synced': 0}
        files = self.get_pending_files()
        if not files:
            return {'status': 'no_files', 'synced': 0}

        summary = self.prepare_summary(files)
        ok = self.upload_summary(summary)
        synced = 0
        log_entries: List[Dict[str, Any]] = []
        if ok:
            for f in files:
                try:
                    new = f + '.synced'
                    os.rename(f, new)
                    size = os.path.getsize(new)
                    log_entries.append({'timestamp': datetime.utcnow().isoformat(), 'file': new, 'status': 'synced', 'bytes': size})
                    synced += 1
                except Exception:
                    log_entries.append({'timestamp': datetime.utcnow().isoformat(), 'file': f, 'status': 'rename_failed'})
        else:
            # upload failed; write attempt to log
            for f in files:
                try:
                    size = os.path.getsize(f)
                except Exception:
                    size = 0
                log_entries.append({'timestamp': datetime.utcnow().isoformat(), 'file': f, 'status': 'upload_failed', 'bytes': size})

        log_path = os.path.join(self.local_dir, 'sync_log.jsonl')
        try:
            with open(log_path, 'a', encoding='utf-8') as fh:
                for le in log_entries:
                    fh.write(json.dumps(le, default=str) + '\n')
        except Exception:
            pass

        return {'status': 'ok' if ok else 'failed', 'synced': synced, 'file_count': len(files)}
