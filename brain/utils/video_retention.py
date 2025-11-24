"""Video retention manager for event-triggered frame storage with retention and optional encryption."""
from __future__ import annotations

import os
import time
import threading
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Optional

import numpy as np
import cv2

from config.config_loader import ConfigLoader
from utils.encryption import encrypt_file


class VideoRetentionManager:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.retention_hours = float(self.config.get('privacy.video_retention_hours', 24))
        self.max_frames_per_trip = int(self.config.get('sensors.camera.max_retained_frames_per_trip', self.config.get('sensors.camera.max_retained_frames_per_trip', 100)))
        self.output_dir = str(self.config.get('video_retention.output_dir', 'data/video_frames'))
        os.makedirs(self.output_dir, exist_ok=True)
        self.trip_frames: Dict[str, Dict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
        self.lock = threading.Lock()
        self.cleanup_interval = int(self.config.get('video_retention.cleanup_interval_s', 3600))
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self.encryption_enabled = bool(self.config.get('privacy.encryption_enabled', self.config.get('privacy.encrypt_storage', False)))
        self.encryption_key = None
        if self.encryption_enabled:
            # try to read key like telemetry
            env = str(self.config.get('telemetry.encryption_key_env', 'TELEMETRY_ENCRYPTION_KEY'))
            val = os.environ.get(env)
            if val:
                try:
                    self.encryption_key = bytes.fromhex(val)
                except Exception:
                    try:
                        import base64

                        self.encryption_key = base64.b64decode(val)
                    except Exception:
                        self.encryption_key = None

    def save_frame(self, frame: np.ndarray, metadata: Dict[str, Any], event_type: str, trip_id: str) -> None:
        with self.lock:
            dir_path = os.path.join(self.output_dir, trip_id, event_type)
            os.makedirs(dir_path, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            frame_path = os.path.join(dir_path, f'frame_{timestamp}.jpg')
            # optional anonymization: blur faces or plates if configured
            if bool(self.config.get('privacy.anonymize_video', False)):
                try:
                    # simple placeholder: apply mild Gaussian blur to entire frame
                    frame_to_write = cv2.GaussianBlur(frame, (15, 15), 0)
                except Exception:
                    frame_to_write = frame
            else:
                frame_to_write = frame
            try:
                cv2.imwrite(frame_path, frame_to_write, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception:
                # fallback: skip writing
                return
            metadata['frame_path'] = frame_path
            metadata_json = os.path.join(dir_path, f'metadata_{timestamp}.json')
            try:
                with open(metadata_json, 'w', encoding='utf-8') as fh:
                    json.dump(metadata, fh, default=str)
            except Exception:
                pass

            self.trip_frames[trip_id][event_type].append((frame_path, time.time()))

            # enforce per-trip limit
            while len(self.trip_frames[trip_id][event_type]) > self.max_frames_per_trip:
                try:
                    oldest_path, _ = self.trip_frames[trip_id][event_type].pop(0)
                    try:
                        os.remove(oldest_path)
                    except Exception:
                        pass
                    meta_path = oldest_path.replace('.jpg', '.json')
                    try:
                        os.remove(meta_path)
                    except Exception:
                        pass
                except Exception:
                    break

            # optional encryption
            if self.encryption_enabled and self.encryption_key:
                try:
                    enc_path = frame_path + '.enc'
                    encrypt_file(frame_path, enc_path, self.encryption_key)
                    try:
                        os.remove(frame_path)
                    except Exception:
                        pass
                    # encrypt metadata as well
                    meta_enc = metadata_json + '.enc'
                    encrypt_file(metadata_json, meta_enc, self.encryption_key)
                    try:
                        os.remove(metadata_json)
                    except Exception:
                        pass
                except Exception:
                    pass

    def save_metadata_only(self, metadata: Dict[str, Any], event_type: str, trip_id: str) -> None:
        with self.lock:
            dir_path = os.path.join(self.output_dir, trip_id, event_type)
            os.makedirs(dir_path, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            metadata_json = os.path.join(dir_path, f'metadata_{timestamp}.json')
            try:
                with open(metadata_json, 'w', encoding='utf-8') as fh:
                    json.dump(metadata, fh, default=str)
            except Exception:
                pass
            self.trip_frames[trip_id][event_type].append((metadata_json, time.time()))

    def cleanup_old_frames(self) -> None:
        now = time.time()
        cutoff = now - (self.retention_hours * 3600)
        with self.lock:
            for trip_id in list(self.trip_frames.keys()):
                for event_type in list(self.trip_frames[trip_id].keys()):
                    new_list = []
                    for p, t in list(self.trip_frames[trip_id][event_type]):
                        if t < cutoff:
                            try:
                                if os.path.exists(p):
                                    os.remove(p)
                                meta = p.replace('.jpg', '.json')
                                if os.path.exists(meta):
                                    os.remove(meta)
                                if p.endswith('.enc'):
                                    enc_meta = p.replace('.jpg.enc', '.json.enc')
                                    if os.path.exists(enc_meta):
                                        os.remove(enc_meta)
                            except Exception:
                                pass
                        else:
                            new_list.append((p, t))
                    if new_list:
                        self.trip_frames[trip_id][event_type] = new_list
                    else:
                        del self.trip_frames[trip_id][event_type]
                if not self.trip_frames[trip_id]:
                    del self.trip_frames[trip_id]

    def _cleanup_loop(self) -> None:
        while not self._stop_cleanup.is_set():
            try:
                time.sleep(self.cleanup_interval)
                self.cleanup_old_frames()
            except Exception:
                pass

    def start_cleanup(self) -> None:
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def stop_cleanup(self, join_timeout: float = 5.0) -> None:
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=join_timeout)

    def get_retention_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_frames = sum(len(frames) for trip in self.trip_frames.values() for frames in trip.values())
            sizes = 0
            oldest = None
            for trip in self.trip_frames.values():
                for frames in trip.values():
                    for p, t in frames:
                        try:
                            if os.path.exists(p):
                                sizes += os.path.getsize(p)
                        except Exception:
                            pass
                        if oldest is None or t < oldest:
                            oldest = t
            disk_usage_gb = sizes / (1024.0 ** 3)
            oldest_ts = datetime.fromtimestamp(oldest).isoformat() if oldest else None
        return {'total_frames': total_frames, 'disk_usage_gb': disk_usage_gb, 'oldest_frame_ts': oldest_ts}
