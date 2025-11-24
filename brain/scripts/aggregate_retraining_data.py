"""CLI to aggregate telemetry into retraining datasets using DataAggregator."""
from __future__ import annotations

import argparse
import sys
from config.config_loader import ConfigLoader
from utils.data_aggregator import DataAggregator
import pandas as pd
from tqdm import tqdm


def main():
    cfg = ConfigLoader()
    parser = argparse.ArgumentParser(description='Aggregate telemetry into retraining datasets')
    parser.add_argument('--telemetry-dir', default=cfg.get('telemetry.output_dir', 'data/telemetry'))
    parser.add_argument('--output', required=True, help='Output CSV path (e.g., data/retraining/dataset.csv)')
    parser.add_argument('--start-date', default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--incremental', action='store_true', help='Process only new logs since last aggregation')
    parser.add_argument('--balancing', choices=['smote', 'undersample', 'none'], default=None, help='Override balancing strategy')
    args = parser.parse_args()

    if args.balancing:
        cfg.set('retraining.balancing_strategy', args.balancing)

    aggregator = DataAggregator(cfg)
    try:
        path = aggregator.aggregate_for_retraining(args.telemetry_dir, args.output, args.start_date, args.end_date, args.incremental)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f'Unexpected error: {e}', file=sys.stderr)
        sys.exit(1)

    # generate report
    try:
        df = pd.read_csv(path)
        report = aggregator.generate_retraining_report(df, cfg.get('retraining.output_dir', 'data/retraining'))
    except Exception as e:
        print(f'Warning: failed to generate report: {e}', file=sys.stderr)
        report = None

    positive_ratio = None
    try:
        positive_ratio = float(df['label'].mean()) if 'label' in df else 0.0
    except Exception:
        positive_ratio = 0.0

    print(f'Dataset saved: {path}')
    print(f'Report: {report}')
    print(f'Total samples: {len(df) if df is not None else 0}, Positive ratio: {positive_ratio:.2%}')


if __name__ == '__main__':
    main()
