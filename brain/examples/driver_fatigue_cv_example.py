"""Example script for CV-based driver fatigue detection.

Run with `--video` to process a file or without to attempt live camera.
"""
from __future__ import annotations

import argparse
import time

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
    p.add_argument("--camera", type=int, help="Camera device id (overrides video)", default=None)
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
        cfg._data["sensors"]["driver_fatigue"]["camera_device_id"] = int(args.camera)

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
