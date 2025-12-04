"""Example showing CameraHandler + CameraAdapter usage (simulation mode).

Run this script to collect a few seconds of simulated camera data and print features.
"""
import time
import json
import argparse
import os
from config.config_loader import ConfigLoader
from data_collection.logger import DataLogger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["simulation", "legacy", "mediapipe"], default="simulation")
    p.add_argument("--resolution", help="Target input resolution WxH (e.g. 320x240)", default=None)
    p.add_argument("--num-threads", type=int, help="TFLite/mediapipe num threads", default=None)
    p.add_argument("--frame-skip", type=int, help="Skip inference on intermediate frames (0=no skip)", default=None)
    p.add_argument("--duration", type=int, help="Run duration seconds", default=10)
    p.add_argument("--visualize", action="store_true", help="Show annotated frames in a window")
    p.add_argument("--output", help="Path to save annotated video (mp4)", default=None)
    args = p.parse_args()

    cfg = ConfigLoader("config/default_config.yaml")

    # Helper to apply nested dict overrides using ConfigLoader.set()
    def apply_overrides(cfg, mapping, prefix=""):
        for k, v in mapping.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                apply_overrides(cfg, v, key)
            else:
                cfg.set(key, v)

    # model override mapping
    if args.model == "mediapipe":
        overrides = {
            "sensors": {"camera": {"model_type": "mediapipe_tflite", "enable_simulation": False, "enable_pose_detection": True}},
            "mediapipe": {"enabled": True},
        }
        apply_overrides(cfg, overrides)
    elif args.model == "legacy":
        # explicit legacy/opencv backend
        overrides = {"sensors": {"camera": {"model_type": "opencv", "enable_simulation": False, "enable_pose_detection": False}}}
        apply_overrides(cfg, overrides)
    else:
        # simulation: use generator and disable webcam
        overrides = {"sensors": {"camera": {"enable_webcam": False, "model_type": "simulation", "enable_simulation": True}}}
        apply_overrides(cfg, overrides)
    # Apply runtime tunables passed on the CLI to the camera config
    runtime_overrides = {}
    cam_over = runtime_overrides.setdefault("sensors", {}).setdefault("camera", {})
    if args.resolution:
        try:
            w, h = args.resolution.split("x")
            cam_over["camera_resolution"] = {"width": int(w), "height": int(h)}
        except Exception:
            print("Invalid --resolution format, expected WxH like 320x240")
    if args.num_threads is not None:
        # this key is read by some TFLite loader helpers
        runtime_overrides.setdefault("mediapipe", {})["tflite_num_threads"] = int(args.num_threads)
        cam_over["tflite_num_threads"] = int(args.num_threads)
    if args.frame_skip is not None:
        cam_over["tflite_frame_skip"] = int(args.frame_skip)

    if runtime_overrides:
        apply_overrides(cfg, runtime_overrides)

    logger = DataLogger(cfg)

    # Quick runtime tuning guidance for low-powered devices
    print("Runtime tuning: if inference latency is high (e.g. >200ms), try: --resolution 320x240 --num-threads 2 --frame-skip 1")
    print(f"Using camera model: {args.model}. To force fallback use --model legacy or --model simulation.")

    # Start logging / capture
    logger.start()
    print(f"Logging for {args.duration} seconds (camera mode={args.model})...")
    try:
        time.sleep(args.duration)
    finally:
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

    # Optionally show visualized frames or save annotated video (simple helper)
    if args.visualize or args.output:
        try:
            # try to read annotated frames buffer if present (some loggers write an _annotated file)
            annotated_path = latest.replace('.json', '_annotated.mp4')
            if args.output:
                # copy the annotated file if it exists
                if os.path.exists(annotated_path):
                    import shutil

                    shutil.copy(annotated_path, args.output)
                    print(f"Annotated video saved to: {args.output}")
                else:
                    print("No annotated video available to save.")
            if args.visualize:
                # attempt to open a window showing frames if DataLogger exposes them
                print("Visualization requested but live windowing is not implemented in this example.")
        except Exception:
            print("Visualization/save helper failed (non-critical)")


if __name__ == "__main__":
    main()
