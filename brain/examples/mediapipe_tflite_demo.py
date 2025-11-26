"""Minimal demo comparing MediaPipe Python vs TFLite task pipeline for driver fatigue.

This script toggles the `use_mediapipe_tasks` flag and runs the handler for a number of
frames, printing avg latency and a couple of sample fields (backend_used, perclos_iris).
"""
import time
from config.config_loader import ConfigLoader
from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler


def main():
    cfg = ConfigLoader("config/default_config.yaml")
    # enable mediapipe tasks for the fatigue sensor
    cfg.override({"sensors": {"driver_fatigue": {"enable_cv_detection": True, "use_mediapipe_tasks": True}}})
    handler = DriverFatigueHandler(cfg.get_all(), use_cv=True)
    print("Running mediapipe tflite demo for 30 reads...")
    times = []
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        s = handler.read()
        t1 = time.perf_counter()
        if s is None:
            continue
        times.append((t1 - t0) * 1000.0)
        samples.append(s)
        print(f"backend={s.get('backend_used')} perclos_iris={s.get('perclos_iris', None)} ear={s.get('ear')}")
        time.sleep(0.05)
    if times:
        mean = sum(times) / len(times)
        print(f"Avg latency ms: {mean:.2f}")
    else:
        print("No timing data collected")


if __name__ == '__main__':
    main()
