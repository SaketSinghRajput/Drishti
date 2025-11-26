"""Small helper to download MediaPipe TFLite models into the repository.

This script expects a JSON file at `scripts/mediapipe_model_urls.json` with a mapping
from target filenames to download URLs. A sample file `scripts/mediapipe_model_urls.sample.json`
is included — copy it to `scripts/mediapipe_model_urls.json` and fill in valid URLs, or
edit the mapping inline below.

NOTE: Official model URLs change over time; consult MediaPipe docs if downloads fail.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_JSON = SCRIPTS / "mediapipe_model_urls.json"
SAMPLE_JSON = SCRIPTS / "mediapipe_model_urls.sample.json"

# default mapping (empty URLs) - populate via JSON
DEFAULT_TARGETS = {
    "data_collection/sensors/driver_drowsiness/models/mediapipe/face_detection_short_range.tflite": "",
    "data_collection/sensors/driver_drowsiness/models/mediapipe/face_landmark_lite.tflite": "",
    "data_collection/sensors/driver_drowsiness/models/mediapipe/iris_landmark.tflite": "",
    "data_collection/sensors/camera_handler/models/pose_detection.tflite": "",
    "data_collection/sensors/camera_handler/models/pose_landmark_full.tflite": "",
}


def load_urls():
    if DEFAULT_JSON.exists():
        with open(DEFAULT_JSON, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data
            except Exception:
                print("Failed to parse mediapipe_model_urls.json", file=sys.stderr)
                return DEFAULT_TARGETS
    else:
        print(f"No {DEFAULT_JSON} found. Create it from {SAMPLE_JSON} and add model URLs.")
        return DEFAULT_TARGETS


def download(url: str, dest: Path):
    if not requests:
        raise RuntimeError("requests package not installed. Install with `pip install requests`")
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")


def main():
    mapping = load_urls()
    for target, url in mapping.items():
        if not url:
            print(f"Skipping {target}: no URL provided")
            continue
        dest = ROOT / target
        try:
            download(url, dest)
        except Exception as e:
            print(f"Failed to download {url}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
