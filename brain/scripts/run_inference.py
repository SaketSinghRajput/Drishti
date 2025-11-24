#!/usr/bin/env python3
"""CLI script for running edge inference."""
import argparse
import logging
import time
import sys
from pathlib import Path

from config.config_loader import ConfigLoader
from inference.edge_inference import EdgeInferenceEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run edge inference for accident prediction')
    parser.add_argument('--config', type=str, default='config/default_config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, help='Path to model file (.xgb, .tflite, .onnx). Overrides config.')
    parser.add_argument('--model-type', type=str, choices=['xgboost', 'tflite', 'onnx', 'auto'], default='auto', help='Model type (auto-detect from extension if not specified)')
    parser.add_argument('--duration', type=int, default=60, help='Duration to run inference (seconds)')
    parser.add_argument('--vehicle-id', type=str, default='vehicle_001', help='Vehicle identifier')
    parser.add_argument('--trip-id', type=str, help='Trip identifier (auto-generated if not provided)')
    parser.add_argument('--simulation', action='store_true', help='Use simulated sensor data')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load config
    config = ConfigLoader(args.config)
    logger.info(f'Loaded config from {args.config}')

    # Initialize engine
    engine_kwargs = {'config': config, 'vehicle_hash': args.vehicle_id}
    if args.model:
        engine_kwargs['model_path'] = args.model
        engine_kwargs['model_type'] = args.model_type
    if args.trip_id:
        engine_kwargs['trip_id'] = args.trip_id
    engine = EdgeInferenceEngine(**engine_kwargs)

    logger.info('Starting inference engine...')
    try:
        engine.start()
    except Exception as e:
        logger.error(f'Failed to start engine: {e}')
        sys.exit(1)

    logger.info(f'Running inference for {args.duration} seconds...')
    start_time = time.time()

    try:
        while time.time() - start_time < args.duration:
            if int(time.time() - start_time) % 5 == 0:
                pred = engine.get_latest_prediction()
                if pred:
                    logger.info(f'Risk: {pred["risk_score"]:.3f} | Alert: {pred["alert_level"]} | Uncertain: {pred["uncertainty_flag"]}')
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info('Interrupted by user')

    finally:
        logger.info('Stopping inference engine...')
        engine.stop()
        status = engine.get_status()
        logger.info('Final statistics:')
        logger.info('  Total inferences: %d', status.get('total_inferences', 0))
        logger.info('  Mean latency: %.1fms', status['latency_stats']['mean_ms'])
        logger.info('  P95 latency: %.1fms', status['latency_stats']['p95_ms'])


if __name__ == '__main__':
    main()
