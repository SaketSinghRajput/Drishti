"""Minimal example that wires config, DataLogger and EdgeInferenceEngine.

This example is intentionally small; adapt to your system config paths.
"""
import time
import yaml
from data_collection.logger import DataLogger
from inference.edge_inference import EdgeInferenceEngine


def main(cfg_path="config_examples/dev_sim.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    inference = EdgeInferenceEngine(cfg.get("inference", {}))
    logger = DataLogger(cfg.get("sensors", {}), inference=inference)

    logger.start()
    try:
        for _ in range(10):
            time.sleep(1)
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
