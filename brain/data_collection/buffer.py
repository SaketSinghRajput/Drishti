"""Circular buffer implementation for storing JSON windows to disk."""
from __future__ import annotations

import json
import os
import threading
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from config.config_loader import ConfigLoader
from utils.time_utils import get_utc_timestamp, parse_timestamp
from utils.validation import validate_window


class CircularBuffer:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.retention_hours = float(self.config.get("data_collection.buffer_retention_hours", 24))
        self.storage_path = str(self.config.get("data_collection.storage_path", "data/windows"))
        os.makedirs(self.storage_path, exist_ok=True)
        self._lock = threading.Lock()

    def write_window(self, window: Dict[str, Any]) -> str:
        # optional validation
        ok, errors = validate_window(window)
        strict = bool(self.config.get("data_collection.strict_schema", False))
        if not ok:
            # log validation errors
            logging.getLogger(__name__).warning("Window failed schema validation: %s", errors)
            if strict:
                # in strict mode, do not write invalid windows
                raise ValueError(f"Window failed schema validation: {errors}")
            # still write but include errors in filename for debugging
            fname = f"window_invalid_{uuid.uuid4().hex}.json"
        else:
            fname = f"window_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.json"

        path = os.path.join(self.storage_path, fname)
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(window, f, indent=2)
        # return absolute path to avoid caller confusion about relative paths
        return os.path.abspath(path)

    def cleanup_old_files(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        with self._lock:
            for fn in os.listdir(self.storage_path):
                path = os.path.join(self.storage_path, fn)
                try:
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(path)).replace(tzinfo=timezone.utc)
                    if mtime < cutoff:
                        os.remove(path)
                except Exception:
                    # ignore errors during cleanup
                    pass

    def get_file_count(self) -> int:
        return len(os.listdir(self.storage_path))

    def get_total_size_mb(self) -> float:
        total = 0
        for fn in os.listdir(self.storage_path):
            total += os.path.getsize(os.path.join(self.storage_path, fn))
        return total / (1024 * 1024)

    def list_files(self) -> list:
        """Return a list of files in storage_path sorted by modification time (oldest->newest)."""
        with self._lock:
            files = [os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path) if os.path.isfile(os.path.join(self.storage_path, f))]
            files.sort(key=lambda p: os.path.getmtime(p))
            return files

    def read_window(self, path: str) -> Dict[str, Any]:
        """Read a JSON window file. 'path' may be an absolute path or a filename under storage_path."""
        # normalize path
        if not os.path.isabs(path):
            # guard against callers passing a path-like string that already
            # contains the storage_path (avoid doubling it). Use basename
            # to extract the filename and join with storage_path.
            fname = os.path.basename(path)
            path = os.path.join(self.storage_path, fname)
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
