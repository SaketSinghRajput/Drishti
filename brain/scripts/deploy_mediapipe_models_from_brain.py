"""Copy MediaPipe TFLite models from `brain/models/` into sensor model folders.

This script performs the local copy (no downloads) using existing model binaries
found in `brain/models/`. It will rename files as needed to match the expected
filenames under `data_collection/sensors/.../models/mediapipe/` and compute SHA256
checksums. Optionally, it can update README placeholders with the computed checksums.

Usage:
    python scripts/deploy_mediapipe_models_from_brain.py [--patch-readmes]

Run locally in your checkout (PowerShell example):
    python .\scripts\deploy_mediapipe_models_from_brain.py --patch-readmes
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "brain" / "models"
DEST_DRIVER = ROOT / "data_collection" / "sensors" / "driver_drowsiness" / "models" / "mediapipe"
DEST_CAMERA = ROOT / "data_collection" / "sensors" / "camera_handler" / "models"

MAPPINGS = {
    "face_detection_short_range.tflite": (SRC_DIR / "face_detection_short_range.tflite", DEST_DRIVER / "face_detection_short_range.tflite"),
    # rename face_landmark.tflite -> face_landmark_lite.tflite for expected name
    "face_landmark_lite.tflite": (SRC_DIR / "face_landmark.tflite", DEST_DRIVER / "face_landmark_lite.tflite"),
    "iris_landmark.tflite": (SRC_DIR / "iris_landmark.tflite", DEST_DRIVER / "iris_landmark.tflite"),
    "pose_detection.tflite": (SRC_DIR / "pose_detection.tflite", DEST_CAMERA / "pose_detection.tflite"),
    "pose_landmark_full.tflite": (SRC_DIR / "pose_landmark_full.tflite", DEST_CAMERA / "pose_landmark_full.tflite"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs():
    DEST_DRIVER.mkdir(parents=True, exist_ok=True)
    DEST_CAMERA.mkdir(parents=True, exist_ok=True)


def deploy(patch_readmes: bool = False):
    ensure_dirs()
    results = {}
    for name, (src, dst) in MAPPINGS.items():
        if not src.exists():
            print(f"Source not found for {name}: {src}")
            results[name] = {"status": "missing_source", "src": str(src), "dst": str(dst)}
            continue
        try:
            shutil.copy2(src, dst)
            s = dst.stat().st_size
            h = sha256(dst)
            print(f"Copied {src} -> {dst} (size={s}, sha256={h})")
            results[name] = {"status": "copied", "dst": str(dst), "size": s, "sha256": h}
        except Exception as e:
            print(f"Failed to copy {src} -> {dst}: {e}")
            results[name] = {"status": "failed", "error": str(e)}

    if patch_readmes:
        try:
            patch_driver_readme(results)
            patch_camera_readme(results)
        except Exception as e:
            print("Failed to patch READMEs:", e)

    return results


def patch_driver_readme(results: dict):
    rd = DEST_DRIVER / "README.md"
    if not rd.exists():
        print("Driver README not found; skipping patch.")
        return
    txt = rd.read_text(encoding="utf-8")
    for key, info in results.items():
        if info.get("status") == "copied" and key in ("face_detection_short_range.tflite", "face_landmark_lite.tflite", "iris_landmark.tflite"):
            txt = txt.replace(f"{key}: <sha256-placeholder>", f"{key}: {info.get('sha256')}")
    rd.write_text(txt, encoding="utf-8")
    print("Patched driver README with sha256 values where available.")


def patch_camera_readme(results: dict):
    rd = DEST_CAMERA / "README.md"
    if not rd.exists():
        print("Camera README not found; skipping patch.")
        return
    txt = rd.read_text(encoding="utf-8")
    for key, info in results.items():
        if info.get("status") == "copied" and key in ("pose_detection.tflite", "pose_landmark_full.tflite"):
            txt = txt.replace(f"{key}: <sha256-placeholder>", f"{key}: {info.get('sha256')}")
    rd.write_text(txt, encoding="utf-8")
    print("Patched camera README with sha256 values where available.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-readmes", action="store_true", help="Patch README placeholders with computed sha256 values")
    args = parser.parse_args()

    res = deploy(patch_readmes=args.patch_readmes)
    missing = [k for k, v in res.items() if v.get("status") != "copied"]
    if missing:
        print("Some models were not copied. Missing sources:")
        for k in missing:
            print(" -", k, res[k])
        sys.exit(2)
    print("All models copied successfully.")


if __name__ == '__main__':
    main()
