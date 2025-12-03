"""Example script for CV-based driver fatigue detection.

Run with `--video` to process a file or without to attempt live camera.
"""
from __future__ import annotations

import argparse
import time
import os
import sys

# Ensure repository `brain/` package root is on sys.path when running the
# example as a script (python examples/driver_fatigue_cv_example.py). When a
# script is executed directly, sys.path[0] is the script directory which means
# sibling packages like `data_collection` are not importable. Prepend the
# parent directory (the `brain` package root) so absolute imports work.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
from config.config_loader import ConfigLoader
import json
import jsonschema
from pathlib import Path
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", help="Path to video file to process", default=None)
    p.add_argument("--camera", type=str, help="Camera device id or device path (e.g. 0 or /dev/video0) (overrides video)", default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--output", help="Path to save annotated video (mp4)", default=None)
    p.add_argument("--config", help="Path to config yaml", default=None)
    args = p.parse_args()

    # load config (allow override)
    if args.config:
        cfg = ConfigLoader(args.config)
    else:
        cfg = ConfigLoader()
    # Enable video file mode if provided
    if args.video:
        cfg._data.setdefault("sensors", {}).setdefault("driver_fatigue", {})
        cfg._data["sensors"]["driver_fatigue"]["enable_video_file_input"] = True
        cfg._data["sensors"]["driver_fatigue"]["video_file_path"] = args.video
    if args.camera is not None:
        cfg._data.setdefault("sensors", {}).setdefault("driver_fatigue", {})
        cam_val = args.camera
        # Accept either integer index or device path like /dev/video0
        if isinstance(cam_val, str):
            if cam_val.isdigit():
                cam_val = int(cam_val)
            elif cam_val.startswith("/dev/"):
                # pass through device path string — CameraInterface should handle it
                cam_val = cam_val
            else:
                # try int conversion as a fallback
                try:
                    cam_val = int(cam_val)
                except Exception:
                    # leave as string; CameraInterface may still accept it
                    pass
        cfg._data["sensors"]["driver_fatigue"]["camera_device_id"] = cam_val

    # Handler accepts either a dict or a ConfigLoader; pass underlying dict for clarity
    handler = DriverFatigueHandler(cfg._data if hasattr(cfg, "_data") else cfg)
    # load window schema for validation (optional)
    schema_path = Path(__file__).resolve().parents[2] / "config" / "schema" / "sensor_window_schema.json"
    schema = None
    try:
        with open(schema_path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except Exception:
        schema = None
    writer = None
    try:
        if args.output is not None:
            # prepare writer when visualizing/saving
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cfg._data.get("sensors", {}).get("driver_fatigue", {}).get("camera_resolution", {}).get("width", 640))
            h = int(cfg._data.get("sensors", {}).get("driver_fatigue", {}).get("camera_resolution", {}).get("height", 480))
            writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))

        while True:
            sample = handler.read()
            if sample:
                if schema is not None:
                    try:
                        wrapper = {"timestamp": sample.get("timestamp"), "driver_fatigue": sample}
                        jsonschema.validate(wrapper, schema)
                    except Exception as e:
                        print("Schema validation failed:", e)

                if args.visualize and getattr(handler, "_camera", None) is not None:
                    frame = handler._camera._last_frame if getattr(handler._camera, "_last_frame", None) is not None else None
                    if frame is not None:
                        vis = frame.copy()
                        text = f"Score:{sample.get('drowsiness_score', 0.0):.2f} BR:{sample.get('blink_rate', 0.0):.1f} PERC:{sample.get('perclos', 0.0):.1f}%"
                        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        hp = sample.get("head_pose", {})
                        cv2.putText(vis, f"Pitch:{hp.get('pitch',0):.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                        cv2.putText(vis, f"Yaw:{hp.get('yaw',0):.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                        cv2.imshow("Driver Fatigue", vis)
                        if writer is not None:
                            try:
                                writer.write(vis)
                            except Exception:
                                pass
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                # print JSON-friendly sample
                try:
                    print(json.dumps(sample))
                except Exception:
                    print(sample)

            time.sleep(1.0 / max(1, getattr(handler, "sample_rate", 1)))
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        try:
            handler.close()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
