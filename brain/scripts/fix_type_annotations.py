"""Minimal helper to scan Python files for functions missing annotations.

This is a lightweight reporting tool (does not apply fixes) intended to
be used as a developer aid to identify obvious missing annotations.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def find_unannotated_functions(root: Path):
    report = []
    for p in root.rglob('*.py'):
        try:
            src = p.read_text()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                missing = False
                # check args
                for arg in node.args.args:
                    if arg.annotation is None:
                        missing = True
                        break
                # check return
                if node.returns is None:
                    missing = True
                if missing:
                    report.append(str(p) + ':' + node.name)
    return report


if __name__ == '__main__':
    root = Path('.')
    rep = find_unannotated_functions(root)
    if not rep:
        print('No obvious unannotated functions found')
        sys.exit(0)
    print('Functions missing annotations (sample):')
    for r in rep[:200]:
        print('  ', r)
    sys.exit(1)
