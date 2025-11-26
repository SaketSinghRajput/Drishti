"""Example showing CameraHandler + CameraAdapter usage (simulation mode).

Run this script to collect a few seconds of simulated camera data and print features.
"""
import time
import json
import argparse
from config.config_loader import ConfigLoader
from data_collection.logger import DataLogger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["simulation", "legacy", "mediapipe"], default="simulation")
    args = p.parse_args()

    cfg = ConfigLoader("config/default_config.yaml")
    # model override mapping
    if args.model == "mediapipe":
        cfg.override({
            "sensors": {"camera": {"model_type": "mediapipe_tflite", "enable_simulation": False, "enable_pose_detection": True}},
            "mediapipe": {"enabled": True},
        })
    elif args.model == "legacy":
        # explicit legacy/opencv backend
        cfg.override({
            "sensors": {"camera": {"model_type": "opencv", "enable_simulation": False, "enable_pose_detection": False}},
        })
    else:
        # simulation: use generator and disable webcam
        cfg.override({
            "sensors": {"camera": {"enable_webcam": False, "model_type": "simulation", "enable_simulation": True}},
        })
    logger = DataLogger(cfg)
    logger.start()
    print("Logging for 10 seconds (camera simulation)...")
    time.sleep(10)
    logger.stop()
    # Read most recent window file from buffer
    buf = logger.buffer
    files = buf.list_files() if hasattr(buf, "list_files") else []
    if not files:
        print("No windows written.")
        return
    latest = files[-1]
    print(f"Latest window: {latest}")
    w = buf.read_window(latest) if hasattr(buf, "read_window") else None
    if not w:
        print("Failed to read window")
        return
    cam_samples = w.get("camera", [])
    cam_features = w.get("camera_features")
    print(f"Camera raw samples: {len(cam_samples)} frames")
    print("Camera features:")
    print(json.dumps(cam_features, indent=2))


if __name__ == "__main__":
    main()
