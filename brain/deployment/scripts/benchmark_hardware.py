#!/usr/bin/env python3
"""Benchmark hardware: CPU/memory, camera FPS, and optional TFLite model latency.

Outputs a JSON object to stdout with measured values.
"""
import time
import json
import argparse
try:
    import psutil
except Exception:
    psutil = None
try:
    import cv2
except Exception:
    cv2 = None


def measure_cpu():
    if psutil is None:
        return None
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict()
    }


def measure_camera(device=0, frames=60):
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return {"error": "camera_unavailable"}
    t0 = time.time()
    read = 0
    for _ in range(frames):
        ret, _ = cap.read()
        if not ret:
            break
        read += 1
    t1 = time.time()
    cap.release()
    duration = max(1e-6, t1 - t0)
    return {"frames": read, "seconds": duration, "fps": read / duration}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    out = {"cpu": measure_cpu(), "camera": measure_camera(args.camera, args.frames)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
