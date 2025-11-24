import sys
import threading
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import main


def test_orchestrator_runs_short_duration(monkeypatch):
    # Run orchestrator for 2 seconds in dev mode to exercise startup and one loop
    monkeypatch.setattr('sys.argv', ['main.py', '--mode', 'dev', '--duration', '2', '--config', 'config_examples/dev_sim.yaml'])
    # Run main (it will block until duration elapses)
    main.main()
    # If we reach here, orchestrator started and stopped cleanly
    assert True
