#!/usr/bin/env python3
"""Example: run the orchestrator in dev mode for 30 seconds with simulation enabled."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import main

if __name__ == '__main__':
    # This is a quick demonstration; use `--mode dev` and a short duration
    import sys
    sys.argv = ['main.py', '--mode', 'dev', '--duration', '30', '--config', 'config_examples/dev_sim.yaml']
    main()
