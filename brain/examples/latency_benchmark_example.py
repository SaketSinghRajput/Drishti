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
    args = p.parse_args()
    run_handler_benchmark(args.tflite_model, args.video, args.frames)
