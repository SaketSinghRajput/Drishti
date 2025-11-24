"""Example showing CameraHandler + CameraAdapter usage (simulation mode).

Run this script to collect a few seconds of simulated camera data and print features.
"""
import time
import json
from config.config_loader import ConfigLoader
from data_collection.logger import DataLogger


def main():
    cfg = ConfigLoader("config/default_config.yaml")
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
