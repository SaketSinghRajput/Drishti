"""Example wrapper to invoke the benchmark script.

This launches the benchmark module programmatically.
"""
from scripts.benchmark_fatigue_latency import run_handler_benchmark
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tflite-model', default='')
    p.add_argument('--video', default='')
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--camera', action='store_true', help='Use live camera instead of a video file')
    args = p.parse_args()
    # If --camera is passed, run the benchmark using the live camera (no video file)
    if args.camera:
        run_handler_benchmark(args.tflite_model, video='', frames=args.frames)
    else:
        run_handler_benchmark(args.tflite_model, args.video, args.frames)
