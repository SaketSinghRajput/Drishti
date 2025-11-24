"""Generate a dummy battery anomaly model for testing.

This script trains a simple IsolationForest on synthetic features and writes
the model and normalization params to the `models/` directory.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib


def generate(output_dir: str, n_samples: int = 1000):
    os.makedirs(output_dir, exist_ok=True)
    # features: cell_voltage_mean, voltage_imbalance, temp_gradient, current_rms, soc_mean, soh_mean
    rng = np.random.RandomState(42)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 6))
    # inject anomalies
    n_anom = max(10, int(0.05 * n_samples))
    X[:n_anom] += rng.normal(loc=5.0, scale=2.0, size=(n_anom, 6))

    model = IsolationForest(random_state=42, contamination=0.05)
    model.fit(X)

    model_path = os.path.join(output_dir, "battery_anomaly.pkl")
    joblib.dump(model, model_path)
    print(f"Dummy model saved to {model_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o", default="models", help="Output directory for model files")
    p.add_argument("--n", "-n", type=int, default=1000, help="Number of synthetic samples")
    args = p.parse_args()
    generate(args.output, args.n)


if __name__ == "__main__":
    main()
