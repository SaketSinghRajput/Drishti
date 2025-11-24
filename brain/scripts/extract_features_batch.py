"""Batch feature extraction script.

Usage:
    python scripts/extract_features_batch.py --input-dir data/windows --output-file features.csv
"""
import argparse
import json
import os
import logging
from pathlib import Path

from config.config_loader import ConfigLoader
from feature_engineering import FeatureExtractor

try:
    import pandas as pd
except Exception:
    pd = None

logger = logging.getLogger(__name__)


def load_windows_from_dir(d: str):
    res = []
    p = Path(d)
    for fn in sorted(p.rglob("*.json")):
        if fn.name == "metadata.json":
            continue
        try:
            with open(fn, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            res.append(obj)
        except Exception:
            logger.exception("Failed to load %s", fn)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--format", default="csv", choices=["csv", "parquet", "json"])
    parser.add_argument("--filter-labeled", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    cfg = ConfigLoader(args.config)
    fe = FeatureExtractor(cfg)

    windows = load_windows_from_dir(args.input_dir)
    if args.filter_labeled:
        windows = [w for w in windows if w.get("labels")]

    if not windows:
        print("No windows found to process")
        return

    df = fe.extract_to_dataframe(windows)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        if hasattr(df, "to_csv"):
            df.to_csv(str(out_path), index=False)
        else:
            # fallback: write simple json rows
            with open(str(out_path), "w", encoding="utf-8") as fh:
                for row in df:
                    fh.write(json.dumps(row) + "\n")
    elif args.format == "parquet":
        if hasattr(df, "to_parquet"):
            df.to_parquet(str(out_path), index=False)
        else:
            raise RuntimeError("Parquet output requires pandas with pyarrow or fastparquet")
    else:  # json
        if hasattr(df, "to_dict"):
            df.to_json(str(out_path), orient="records")
        else:
            with open(str(out_path), "w", encoding="utf-8") as fh:
                for row in df:
                    fh.write(json.dumps(row) + "\n")

    print(f"Wrote features to {out_path} (rows={len(windows)})")


if __name__ == "__main__":
    main()
