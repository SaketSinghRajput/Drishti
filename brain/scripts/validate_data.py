"""Validate JSON sensor window files against the JSON schema.

Usage: python validate_data.py --input path/to/file_or_dir [--schema path/to/schema.json]
"""
import argparse
import json
import os
import sys
from typing import List

from jsonschema import Draft7Validator, exceptions


def find_json_files(path: str, recursive: bool = False) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = []
    for root, dirs, filenames in os.walk(path):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(root, fn))
        if not recursive:
            break
    return files


def load_schema(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_file(path: str, validator: Draft7Validator, verbose: bool = False):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Failed to parse JSON: {e}"]
    errors = []
    for err in validator.iter_errors(data):
        errors.append(f"{err.message} (at {'/'.join(map(str, err.path))})")
    return len(errors) == 0, errors


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--schema", default="config/schema/sensor_window_schema.json")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--output", help="Path to save JSON report")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if not os.path.exists(args.schema):
        print(f"Schema not found: {args.schema}")
        sys.exit(2)
    schema = load_schema(args.schema)
    validator = Draft7Validator(schema)

    files = find_json_files(args.input, recursive=args.recursive)
    if not files:
        print("No JSON files found to validate.")
        sys.exit(1)

    report = {"total": len(files), "passed": 0, "failed": 0, "details": []}
    for fpath in files:
        ok, errors = validate_file(fpath, validator, verbose=args.verbose)
        report["details"].append({"file": fpath, "ok": ok, "errors": errors})
        if ok:
            report["passed"] += 1
        else:
            report["failed"] += 1
            print(f"{fpath}: FAILED")
            if args.verbose:
                for e in errors:
                    print("  ", e)

    print(f"Checked {report['total']} files: {report['passed']} passed, {report['failed']} failed")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            json.dump(report, out, indent=2)
    if report["failed"] > 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
