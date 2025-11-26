"""Verify presence and basic initialization of MediaPipe TFLite models.

This script checks that the expected model files exist, prints SHA256 checksums,
attempts to initialize the lightweight loaders in `brain.utils.mediapipe_loaders`,
and runs a single dummy frame through the drowsiness TFLite detector pipeline
to confirm the `mediapipe_tflite` backend path activates when models are present.

Usage:
    python scripts/verify_mediapipe_models.py

Note: This script does not download models; use `download_mediapipe_models.py` to fetch them.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MODELS = {
    "face_detection_short_range.tflite": ROOT / "data_collection" / "sensors" / "driver_drowsiness" / "models" / "mediapipe" / "face_detection_short_range.tflite",
    "face_landmark_lite.tflite": ROOT / "data_collection" / "sensors" / "driver_drowsiness" / "models" / "mediapipe" / "face_landmark_lite.tflite",
    "iris_landmark.tflite": ROOT / "data_collection" / "sensors" / "driver_drowsiness" / "models" / "mediapipe" / "iris_landmark.tflite",
    "pose_detection.tflite": ROOT / "data_collection" / "sensors" / "camera_handler" / "models" / "pose_detection.tflite",
    "pose_landmark_full.tflite": ROOT / "data_collection" / "sensors" / "camera_handler" / "models" / "pose_landmark_full.tflite",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_models():
    print("Checking MediaPipe model files:")
    missing = []
    for name, p in REQUIRED_MODELS.items():
        exists = p.exists()
        print(f" - {name}: {'FOUND' if exists else 'MISSING'} -> {p}")
        if exists:
            try:
                sz = p.stat().st_size
                print(f"    size={sz} bytes, sha256={sha256(p)}")
            except Exception as e:
                print(f"    failed to read {p}: {e}")
        else:
            missing.append(name)
    return missing


def try_initialize_loaders():
    # Attempt to initialize the mediapipe loaders and the TFLite drowsiness detector
    try:
        from brain.utils.mediapipe_loaders import MediaPipeFaceDetector, MediaPipeFaceLandmarker, MediaPipeIrisLandmarker
        from data_collection.sensors.driver_drowsiness.tflite_detector import TFLiteDrowsinessDetector
    except Exception as e:
        print("Failed to import loaders/detector:", e)
        return False

    mp_cfg = {
        "tflite_num_threads": 2,
        "face_detection": {"model_path": str(REQUIRED_MODELS["face_detection_short_range.tflite"])},
        "face_landmark": {"model_path": str(REQUIRED_MODELS["face_landmark_lite.tflite"])},
        "iris_landmark": {"model_path": str(REQUIRED_MODELS["iris_landmark.tflite"])},
    }

    # instantiate loaders where files exist
    try:
        fd = None
        fl = None
        ir = None
        if REQUIRED_MODELS["face_detection_short_range.tflite"].exists():
            fd = MediaPipeFaceDetector(str(REQUIRED_MODELS["face_detection_short_range.tflite"]), mp_cfg)
        if REQUIRED_MODELS["face_landmark_lite.tflite"].exists():
            fl = MediaPipeFaceLandmarker(str(REQUIRED_MODELS["face_landmark_lite.tflite"]), mp_cfg)
        if REQUIRED_MODELS["iris_landmark.tflite"].exists():
            ir = MediaPipeIrisLandmarker(str(REQUIRED_MODELS["iris_landmark.tflite"]), mp_cfg)

        det_cfg = {"use_mediapipe_tasks": True, "mediapipe": mp_cfg}
        det = TFLiteDrowsinessDetector(config=det_cfg)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print("Running a single frame through the TFLite drowsiness detector (this may be slow on first run)...")
        out = det.process_frame(frame)
        print("Detector output keys:", list(out.keys()))
        print("backend_used:", out.get("backend_used"))
        print("iris keys present:", "iris" in out and bool(out.get("iris")))
        if "perclos_iris" in out:
            print("perclos_iris:", out.get("perclos_iris"))
        return True
    except Exception as e:
        print("Loader initialization or detector run failed:", e)
        return False


def main():
    missing = check_models()
    if missing:
        print("Some models are missing. Use scripts/download_mediapipe_models.py to fetch them (edit scripts/mediapipe_model_urls.json first).")
    else:
        print("All models present. Attempting to initialize loaders...")
    ok = try_initialize_loaders()
    if ok:
        print("Initialization check completed. If backend_used == 'mediapipe_tflite' the MediaPipe TFLite task pipeline was activated.")
    else:
        print("MediaPipe TFLite task path unavailable or failed to run. Falling back to heuristic/generic TFLite may occur in runtime.")


if __name__ == '__main__':
    main()
