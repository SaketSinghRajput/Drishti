"""Example runner that uses the verification helper to print expected fields.

Run: `python examples/verify_mediapipe_setup.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_mediapipe_models import check_models, try_initialize_loaders


def main():
    missing = check_models()
    print()
    ok = try_initialize_loaders()
    if ok:
        print("Verification: MediaPipe TFLite pipeline appears reachable (see details above).")
    else:
        print("Verification: MediaPipe TFLite pipeline not reachable; check model files and loader logs.")


if __name__ == '__main__':
    main()
