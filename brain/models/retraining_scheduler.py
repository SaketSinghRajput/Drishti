import json
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import requests
import threading


class RetrainingScheduler:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = cfg.get('enabled', False)
        self.schedule = cfg.get('schedule', 'monthly')
        self.approval_required = cfg.get('approval_required', True)
        self.approval_method = cfg.get('approval_method', 'file')
        self.approval_file_path = cfg.get('approval_file_path', 'data/retraining_approval.flag')
        self.approval_webhook_url = cfg.get('approval_webhook_url', '')
        self.check_interval_hours = cfg.get('check_interval_hours', 24)
        self.last_run_path = cfg.get('last_run_path', 'data/last_retraining.json')
        self.approval_timeout_hours = cfg.get('approval_timeout_hours', 48)

    def _read_last_run(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.last_run_path):
            return None
        try:
            with open(self.last_run_path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            return None

    def _write_last_run(self, info: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.last_run_path), exist_ok=True)
        with open(self.last_run_path, 'w', encoding='utf-8') as fh:
            json.dump(info, fh)

    def should_trigger_retraining(self) -> bool:
        if not self.enabled:
            return False
        now = datetime.now(timezone.utc)
        lr = self._read_last_run()
        if self.schedule == 'monthly':
            if now.day == 1:
                # allow once per month
                if not lr:
                    return True
                last_ts = datetime.fromisoformat(lr.get('last_run'))
                if last_ts.month != now.month:
                    return True
        if self.schedule == 'weekly':
            if now.weekday() == 0:
                if not lr:
                    return True
                last_ts = datetime.fromisoformat(lr.get('last_run'))
                if last_ts.isocalendar()[1] != now.isocalendar()[1]:
                    return True
        return False

    def check_approval(self) -> bool:
        if not self.approval_required:
            return True
        if self.approval_method == 'file':
            if os.path.exists(self.approval_file_path):
                try:
                    with open(self.approval_file_path, 'r', encoding='utf-8') as fh:
                        val = fh.read().strip().lower()
                    os.remove(self.approval_file_path)
                    return val == 'approved'
                except Exception:
                    return False
            return False
        # webhook approval: try to POST a request and then poll for approval
        if self.approval_method == 'webhook' and self.approval_webhook_url:
            try:
                return self.request_approval({'action': 'retraining_request', 'timestamp': datetime.now(timezone.utc).isoformat()})
            except Exception:
                return False
        return False

    def request_approval(self, payload: Dict[str, Any]) -> bool:
        """Send a request to the configured approval webhook and wait for approval.

        Returns True if approved within the timeout window, False otherwise.
        """
        if not self.approval_webhook_url:
            return False
        try:
            # Post the request (best-effort)
            try:
                requests.post(self.approval_webhook_url, json=payload, timeout=5)
            except Exception:
                # ignore post failures and rely on polling endpoint
                pass
            return self.wait_for_approval(timeout_hours=self.approval_timeout_hours)
        except Exception:
            return False

    def wait_for_approval(self, timeout_hours: int = None) -> bool:
        timeout_hours = timeout_hours or self.approval_timeout_hours
        deadline = time.time() + (timeout_hours * 3600)
        poll_interval = max(60, int(self.check_interval_hours * 3600 / 24))
        while time.time() < deadline:
            try:
                r = requests.get(self.approval_webhook_url, timeout=5)
                if r.status_code == 200:
                    try:
                        j = r.json()
                        status = j.get('status') if isinstance(j, dict) else None
                    except Exception:
                        status = r.text.strip().lower()
                    if status == 'approved' or status == 'true' or status == 'yes':
                        return True
            except Exception:
                pass
            # wait a short while before polling again
            time.sleep(min(300, poll_interval))
        return False

    def update_last_run(self):
        self._write_last_run({'last_run': datetime.now(timezone.utc).isoformat()})
