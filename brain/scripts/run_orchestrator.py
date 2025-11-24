#!/usr/bin/env python3
"""Developer convenience wrapper to run the orchestrator in-process.

Usage:
    python scripts/run_orchestrator.py --mode dev --config config_examples/dev_sim.yaml
"""
from __future__ import annotations

import sys
import argparse
import signal

# Add project root
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import main as orchestrator_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--mode', choices=('dev','prod'), default='dev')
    parser.add_argument('--duration', type=int, default=0)
    args = parser.parse_args()

    # Forward to main.main
    sys.argv = ['main.py', '--config', args.config, '--mode', args.mode, '--duration', str(args.duration)]
    try:
        orchestrator_main()
    except KeyboardInterrupt:
        print('Orchestrator interrupted')


if __name__ == '__main__':
    main()
