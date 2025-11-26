"""Benchmark driver fatigue pipeline latency.

Tries to instantiate DriverFatigueHandler with TFLite enabled and a video
file (if provided). If a real TFLite/model/video is not available, the
script falls back to a synthetic benchmarking path that exercises the
`LatencyProfiler` and reports summary stats.
"""
from __future__ import annotations
import argparse
import time
import sys
import os
import numpy as np

try:
    from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
    from data_collection.sensors.driver_drowsiness.profiler import LatencyProfiler
except Exception:
    DriverFatigueHandler = None
    LatencyProfiler = None


def synthetic_benchmark(frames: int = 100, budget_ms: int = 200):
    prof = LatencyProfiler(budget_ms=budget_ms) if LatencyProfiler is not None else None
    # simulate stages: capture, inference, postprocess
    for i in range(frames):
        t0 = time.perf_counter()
        if prof:
            with prof.profile('frame_read'):
                time.sleep(0.001)
            with prof.profile('inference'):
                time.sleep(0.005)
            with prof.profile('postprocess'):
                time.sleep(0.001)
            total_ms = (time.perf_counter() - t0) * 1000.0
            prof.record_total(total_ms)
        else:
            time.sleep(0.008)
    if prof:
        print('Latency summary (synthetic):')
        print(prof.to_dict())
    else:
        print('No profiler available; synthetic run complete')


def run_handler_benchmark(tflite_model: str = '', video: str = '', frames: int = 300):
    # build a minimal config
    cfg = {
        'sensors.driver_fatigue.use_tflite': True if tflite_model else False,
        'sensors.driver_fatigue': {
            'use_tflite': True if tflite_model else False,
            'tflite_model_path': tflite_model or '',
            'enable_latency_profiling': True,
            'latency_budget_ms': 200,
            'enable_video_file_input': bool(video),
            'video_file_path': video or '',
        }
    }
    try:
        handler = DriverFatigueHandler(cfg, use_cv=True)
    except Exception as e:
        print('Failed to initialize DriverFatigueHandler:', e)
        print('Falling back to synthetic benchmark')
        return synthetic_benchmark(frames)

    collected = 0
    t_start = time.perf_counter()
    while collected < frames:
        s = handler.read()
        if s is None:
            break
        collected += 1
    t_total = time.perf_counter() - t_start
    print(f'Processed {collected} frames in {t_total:.2f}s ({(collected/t_total) if t_total>0 else 0:.2f} fps)')
    health = handler.get_health()
    print('Handler health:')
    print(health)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tflite-model', help='Path to .tflite model to benchmark', default='')
    p.add_argument('--mediapipe-suite', action='store_true', help='Run MediaPipe task suite benchmark (per-stage timing)')
    p.add_argument('--video', help='Path to video file to read frames from', default='')
    p.add_argument('--frames', type=int, default=300)
    args = p.parse_args()
    if DriverFatigueHandler is None or LatencyProfiler is None:
        print('Required modules not available; running synthetic benchmark')
        synthetic_benchmark(args.frames)
        return
    if args.mediapipe_suite:
        # attempt to run mediapipe suite using provided tflite model path as a fallback
        try:
            from brain.utils.model_loader_tflite import TFLiteDetectionModel
            print('Running MediaPipe TFLite task suite (best-effort)')
            # if no model provided, fall back to synthetic
            if not args.tflite_model:
                print('No model provided for mediapipe suite; running synthetic benchmark')
                synthetic_benchmark(args.frames)
            else:
                # simple per-frame timing using TFLiteDetectionModel
                mdl = TFLiteDetectionModel(args.tflite_model, config={'tflite_num_threads': 2})
                try:
                    mdl.load()
                except Exception as e:
                    print('Failed to load mediapipe tflite model:', e)
                    synthetic_benchmark(args.frames)
                    return
                import cv2
                cap = cv2.VideoCapture(0 if not args.video else args.video)
                times = []
                for i in range(args.frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    t0 = time.perf_counter()
                    _ = mdl.predict(frame)
                    times.append((time.perf_counter() - t0) * 1000.0)
                cap.release()
                import numpy as _np
                if times:
                    arr = _np.array(times)
                    print('Mediapipe suite timing (ms): mean=%.2f p50=%.2f p90=%.2f' % (arr.mean(), _np.percentile(arr,50), _np.percentile(arr,90)))
        except Exception as e:
            print('Mediapipe suite failed:', e)
    else:
        run_handler_benchmark(args.tflite_model, args.video, args.frames)


if __name__ == '__main__':
    main()
