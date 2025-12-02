"""Generate synthetic sensor window JSON files for testing.

This is a small generator that creates windows matching the schema with noise.
"""
import argparse
import json
import os
import random
from datetime import datetime, timedelta, timezone


def random_window(base_time: datetime, vehicle_hash: str = "vh_demo", scenario: str = "normal"):
    # base minimal window
    wnd = {
        "timestamp": base_time.isoformat() + "Z",
        "vehicle_hash": vehicle_hash,
        "gps": {"lat": 37.0 + random.uniform(-0.01, 0.01), "lon": -122.0 + random.uniform(-0.01, 0.01), "speed_kph": random.uniform(0, 120), "heading": random.uniform(0, 360)},
        "imu": {"yaw_rate": random.uniform(-5, 5), "pitch": random.uniform(-5, 5), "roll": random.uniform(-5, 5), "ax": random.uniform(-3, 3), "ay": random.uniform(-3, 3), "az": 9.8 + random.uniform(-0.5, 0.5)},
        "vehicle_controls": {"steering_angle": random.uniform(-90, 90), "brake_pressure": random.uniform(0, 1), "brake_flag": False, "throttle_position": random.uniform(0, 1)},
        "driver_fatigue": {"drowsiness_score": 0.1, "blink_rate": random.uniform(5, 20), "perclos": 0.1, "head_pose": {"pitch": 0, "yaw": 0, "roll": 0, "confidence": 1.0}, "yawning_count": 0, "micro_sleep_flag": False},
    }
    if scenario == "drowsy":
        wnd["driver_fatigue"]["drowsiness_score"] = random.uniform(0.7, 1.0)
        wnd["driver_fatigue"]["perclos"] = random.uniform(0.6, 1.0)
    elif scenario == "accident":
        wnd["label"] = {"accident_within_10s": True, "accident_type": "collision", "accident_severity": "severe"}
        wnd["vehicle_controls"]["brake_flag"] = True
        wnd["vehicle_controls"]["brake_pressure"] = 1.0
    elif scenario == "traffic":
        wnd["ble_neighbors"] = {"mesh_density": random.randint(5, 30), "neighbors": []}
        for i in range(wnd["ble_neighbors"]["mesh_density"]):
            wnd["ble_neighbors"]["neighbors"].append({
                "id_hash": f"n{i}",
                "rssi": random.uniform(-90, -30),
                "dist_m": random.uniform(0.5, 50.0),
                "rel_heading": random.uniform(0, 360),
                "rel_speed_mps": random.uniform(-10, 10),
                "last_seen_ms": random.randint(0, 10000),
            })
    elif scenario == "weather":
        wnd["weather"] = {"precip_type": "rain", "precip_intensity_mm_hr": random.uniform(0.1, 10.0), "visibility_m": random.uniform(50, 2000), "ambient_light_level": "dusk", "road_surface": "wet"}

    return wnd


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--scenario", choices=["normal", "drowsy", "traffic", "weather", "accident"], default="normal")
    p.add_argument("--trip-length", type=int, default=1)
    p.add_argument("--labeled", action="store_true", help="Generate labeled windows (adds label field with positive rate)")
    p.add_argument("--positive-rate", type=float, default=0.1, help="Positive label rate when --labeled is set (0-1)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    count = 0
    for s in range(args.num_samples):
        base = now + timedelta(seconds=s * args.trip_length)
        for t in range(args.trip_length):
            wnd = random_window(base + timedelta(seconds=t), vehicle_hash=f"vh_{s}", scenario=args.scenario)
            if args.labeled:
                is_pos = random.random() < float(args.positive_rate)
                wnd["label"] = {"accident_within_10s": bool(is_pos), "accident_type": ("collision" if is_pos else "none"), "near_miss_flag": False, "accident_severity": ("severe" if is_pos else "none")}
            fname = os.path.join(args.output_dir, f"window_{s}_{t}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(wnd, f, indent=2)
            count += 1

    print(f"Wrote {count} sample windows to {args.output_dir}")


if __name__ == "__main__":
    main()
