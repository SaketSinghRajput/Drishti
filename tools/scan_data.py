#!/usr/bin/env python3
"""Small utility to scan local data folders and preview a sample frame.

Usage:
  python tools/scan_data.py         # prints counts and sample filenames
  python tools/scan_data.py --show  # attempts to show first image (requires OpenCV)
"""
from pathlib import Path
import argparse
import sys


def scan(repo_root: Path, show: bool = False):
    data_dir = repo_root / "data"
    video_dir = data_dir / "video_frames"
    telem_dir = data_dir / "telemetry"

    print(f"Repo root: {repo_root}")
    print(f"Data dir: {data_dir}")

    # Telemetry
    if telem_dir.exists():
        files = sorted(telem_dir.glob("**/*"))
        files = [p for p in files if p.is_file()]
        print(f"Telemetry directory found: {telem_dir} ({len(files)} files)")
        for p in files[:5]:
            print(f"  - {p.relative_to(repo_root)}")
    else:
        print(f"Telemetry directory not found at: {telem_dir}")

    # Video frames
    if video_dir.exists():
        imgs = sorted(video_dir.glob("**/*"))
        imgs = [p for p in imgs if p.is_file()]
        print(f"Video frames directory found: {video_dir} ({len(imgs)} files)")
        for p in imgs[:5]:
            print(f"  - {p.relative_to(repo_root)}")

        if show and imgs:
            try:
                import cv2
            except Exception as e:
                print("OpenCV not available (cv2 import failed):", e)
                return
            sample = imgs[0]
            img = cv2.imread(str(sample))
            if img is None:
                print(f"Failed to read sample image: {sample}")
                return
            print(f"Showing sample image: {sample} - press any key to close")
            cv2.imshow("sample", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"Video frames directory not found at: {video_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".", help="Path to repo root (default: .)")
    p.add_argument("--show", action="store_true", help="Show a sample image (requires OpenCV)")
    args = p.parse_args()
    repo_root = Path(args.repo_root).resolve()
    scan(repo_root, show=args.show)


if __name__ == "__main__":
    main()
