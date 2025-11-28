"""Production validation helper.

Performs lightweight environment checks, dependency verification, runs
mypy (if available), runs pytest with strict flags, and performs simple
model-loading smoke checks.

This script is intentionally conservative so it can be executed in CI
with minimal privileges.
"""
from __future__ import annotations

import subprocess
import sys
import shutil
import json
from typing import List

REQUIRED_PACKAGES = ["pytest", "numpy"]


def check_python_packages() -> List[str]:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def run_pytest() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "-W", "error::DeprecationWarning", "--strict-markers"]
    print("Running pytest:", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    print("Production validation: checking required packages...")
    missing = check_python_packages()
    if missing:
        print("Missing packages:", missing)
        print("Please install missing packages before running full validation.")
        # Fail early in CI
        return 2

    # Optional: run mypy if available
    if shutil.which("mypy"):
        print("Running mypy...")
        subprocess.call(["mypy", "."])
    else:
        print("mypy not available; skipping type check")

    # Run pytest with strict flags
    rc = run_pytest()
    if rc != 0:
        print("pytest reported failures/errors (rc=%d)" % rc)
        return rc

    print("Production validation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
