"""Run a live driver fatigue demo using the webcam (if available).

This script starts the fatigue REST API and a `DriverFatigueHandler` in CV
mode (`use_cv=True`). It prints live samples to stdout and serves the API
on the port configured in `config/default_config.yaml` (fallback 5001).

Requirements:
- Set `PYTHONPATH` to the project root so local packages are importable, e.g.
  in PowerShell: `$env:PYTHONPATH = $PWD`
- Install runtime deps: `pip install Flask requests pyyaml opencv-python numpy mediapipe`

Run:
  $env:PYTHONPATH = $PWD
  python examples/run_fatigue_cv_demo.py
"""
from __future__ import annotations

import threading
import time
import logging

try:
    from config.config_loader import ConfigLoader
except Exception:
    ConfigLoader = None

from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
from data_collection.sensors.fatigue_api import create_app
import argparse


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument('--mediapipe', action='store_true', help='Enable MediaPipe TFLite task backends for detector')
    args = p.parse_args()

    cfg = {}
    if ConfigLoader is not None:
        try:
            cfg = ConfigLoader().config
        except Exception:
            cfg = {}

    # apply CLI override to enable mediapipe tasks if requested
    if args.mediapipe:
        try:
            cfg.setdefault('sensors', {}).setdefault('driver_fatigue', {})['use_mediapipe_tasks'] = True
            cfg.setdefault('mediapipe', {})['enabled'] = True
        except Exception:
            pass

    app = create_app(cfg)
    host = cfg.get("sensors", {}).get("driver_fatigue", {}).get("fatigue_api", {}).get("host", "0.0.0.0")
    port = int(cfg.get("sensors", {}).get("driver_fatigue", {}).get("fatigue_api", {}).get("port", 5001))

    # start API server in background thread
    t = threading.Thread(target=lambda: app.run(host=host, port=port), daemon=True)
    t.start()

    # start driver fatigue handler with CV enabled
    handler = None
    try:
        handler = DriverFatigueHandler(cfg, use_cv=True)
    except Exception:
        logging.exception("Failed to initialize DriverFatigueHandler with CV")
        return

    print(f"Fatigue API running at http://{host}:{port}")
    try:
        import cv2

        # If camera interface is available, show a preview window with overlays
        cam = getattr(handler, "_camera", None)
        det = getattr(handler, "_detector", None)

        while True:
            frame = None
            try:
                if cam is not None:
                    frame = cam.read_frame()
                # fallback to handler.read() which will internally call detector
                sample = handler.read()
            except Exception:
                sample = None

            # print numeric sample to console
            if sample is not None:
                print(sample)

            # show preview if we have a frame
            if frame is not None:
                vis = frame.copy()
                # if detector available, try to get detailed metrics without re-reading
                try:
                    if det is not None:
                        res = det.process_frame(frame, is_ir=getattr(cam, 'is_ir', False))
                        # draw basic text overlay
                        cv2.putText(vis, f"EAR:{res.get('ear',0.0):.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(vis, f"MAR:{res.get('mar',0.0):.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        cv2.putText(vis, f"Score:{sample.get('drowsiness_score',0.0):.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                        # draw simple markers for eye/mouth if landmarks present in utils output (not provided here)
                except Exception:
                    pass

                cv2.imshow('Fatigue Demo', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # no frame; sleep briefly to avoid busy-loop
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down demo")
    finally:
        try:
            handler.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
