"""Run all examples sequentially and report results.

This script executes a curated list of example scripts under brain/examples
and records pass/fail/skip status with stdout/stderr capture.

Usage:
    python brain/scripts/run_all_examples.py [--stop-on-error] [--filter pattern]

The script writes `data/example_test_results.json` with a summary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUT_DIR / "example_test_results.json"

# curated list (order matters: basic -> sensors -> training -> inference -> deployment)
EXAMPLES = [
    "basic_logger_example.py",
    "battery_alert_demo.py",
    "battery_dashboard_example.py",
    "ble_mesh_example.py",
    "bms_example.py",
    "bms_monitoring_example.py",
    "camera_example.py",
    "cnn_training_example.py",
    "dashboard_example.py",
    "deployment_example.py",
    "driver_fatigue_analysis_example.py",
    "driver_fatigue_cv_example.py",
    "driver_fatigue_state_demo.py",
    "edge_inference_example.py",
    "environment_example.py",
    "explainability_example.py",
    "fatigue_integration_example.py",
    "feature_extraction_example.py",
    "gsm_sms_example.py",
    "hand_pose_example.py",
    "hardware_diagnostics_example.py",
    "labeling_example.py",
    "latency_benchmark_example.py",
    "llm_integration_example.py",
    "lstm_training_example.py",
    "mediapipe_benchmark_example.py",
    "mediapipe_tflite_demo.py",
    "model_comparison_example.py",
    "multimodal_training_example.py",
    "orchestrator_example.py",
    "pedestrian_intent_example.py",
    "retraining_pipeline_example.py",
    "run_fatigue_cv_demo.py",
    "telemetry_example.py",
    "tflite_conversion_example.py",
    "verify_mediapipe_setup.py",
    "voltage_current_example.py",
    "xgboost_training_example.py",
]


def run_example(path: Path, timeout: int = 60):
    cmd = [sys.executable, str(path)]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start
        return proc.returncode, proc.stdout, proc.stderr, duration
    except subprocess.TimeoutExpired as e:
        return -1, "", f"Timeout after {timeout}s", timeout


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--filter", type=str, default=None)
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args(argv)

    results = []
    for fname in EXAMPLES:
        if args.filter and args.filter not in fname:
            continue
        path = EXAMPLES_DIR / fname
        if not path.exists():
            results.append({"example": fname, "status": "missing", "duration": 0, "error": "file not found"})
            continue
        print(f"Running {fname} ...")
        rc, out, err, dur = run_example(path, timeout=args.timeout)
        status = "passed" if rc == 0 else ("timed_out" if rc == -1 else "failed")
        results.append({"example": fname, "status": status, "duration": dur, "returncode": rc, "stdout": out[:2000], "stderr": err[:2000]})
        # write incremental results
        with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        if rc != 0 and args.stop_on_error:
            print(f"Stopping on error: {fname}")
            break

    # summary
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    missing = sum(1 for r in results if r["status"] == "missing")
    timed_out = sum(1 for r in results if r["status"] == "timed_out")
    summary = {"total": len(results), "passed": passed, "failed": failed, "missing": missing, "timed_out": timed_out}
    print("Summary:", summary)
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
