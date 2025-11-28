"""Quick benchmarking example comparing Python MediaPipe API vs TFLite task loaders.

This is a lightweight script that captures a few frames from the webcam and
runs either the Python MediaPipe processing (if installed) or the TFLite task
loader path implemented in `brain.utils.mediapipe_loaders`.
"""
from __future__ import annotations
import time
import argparse
import cv2
import numpy as np

from brain.utils import mediapipe_loaders
from brain.utils.model_loader_tflite import TFLiteDetectionModel


def run_py_mediapipe(frames=50):
    try:
        import mediapipe as mp
    except Exception:
        print("Python MediaPipe not installed; skipping")
        return None
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    times = []
    cap = cv2.VideoCapture(0)
    for i in range(frames):
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(img_rgb)
        times.append((time.time() - t0) * 1000.0)
    cap.release()
    return times


def run_tflite_task(model_path, frames=50):
    mdl = TFLiteDetectionModel(model_path, config={'tflite_num_threads': 2})
    try:
        mdl.load()
    except Exception as e:
        print('Failed to load TFLite model:', e)
        return None
    cap = cv2.VideoCapture(0)
    times = []
    for i in range(frames):
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        _ = mdl.predict(frame)
        times.append((time.time() - t0) * 1000.0)
    cap.release()
    return times


def summarize(times):
    if not times:
        return {}
    a = np.array(times)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['py','tflite'], default='tflite')
    p.add_argument('--model', default='')
    args = p.parse_args()
    if args.mode == 'py':
        times = run_py_mediapipe(frames=50)
        if times is None:
            # dependency missing — exit cleanly with success code
            print('Python MediaPipe unavailable; exiting')
            import sys

            sys.exit(0)
    else:
        if not args.model:
            print('Provide --model path to a MediaPipe TFLite task model')
            return
        times = run_tflite_task(args.model, frames=50)
        if times is None:
            print('TFLite task model failed to load or runtime missing; exiting')
            import sys

            sys.exit(0)

    print('Summary:', summarize(times))


if __name__ == '__main__':
    main()
