"""Example: demonstrate BLE mesh handler + adapter usage with DataLogger.

Runs the DataLogger for a short period using the default config and prints a
recent window's BLE raw samples and aggregate features.
"""
from config.config_loader import ConfigLoader
from data_collection.logger import DataLogger
import time
import json
import os


def main():
    cfg = ConfigLoader("config/default_config.yaml")
    dl = DataLogger(cfg)
    dl.start()
    try:
        print("Logging for 10 seconds...")
        time.sleep(10)
    finally:
        dl.stop()

    storage = cfg.get("data_collection.storage_path", "data/windows")
    # find most recent file
    try:
        files = [os.path.join(storage, f) for f in os.listdir(storage) if f.endswith(".json")]
        if not files:
            print("No window files found in storage")
            return
        latest = max(files, key=os.path.getmtime)
        with open(latest, "r", encoding="utf-8") as fh:
            w = json.load(fh)
        print("Latest window:")
        # print BLE raw
        ble_raw = w.get("ble_mesh")
        print("BLE raw samples (count):", len(ble_raw) if isinstance(ble_raw, list) else 0)
        bf = w.get("ble_mesh_features")
        print("BLE mesh features:", bf)
    except Exception as e:
        print("Failed to read window files:", e)


if __name__ == "__main__":
    main()
