"""
Deploy hand/pose models from `brain/models/` into sensor model directories.

Usage:
  python deploy_hand_pose_models.py --source-dir brain/models --target-dir brain/data_collection/sensors/driver_drowsiness/models/mediapipe --model-variant lite
"""
import argparse
import shutil
import os
import sys
import hashlib


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest().upper()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', default='brain/models', help='Source dir with canonical models')
    p.add_argument('--target-dir', default='brain/data_collection/sensors/driver_drowsiness/models/mediapipe', help='Target sensor model dir')
    p.add_argument('--model-variant', choices=['lite', 'full'], default='lite')
    args = p.parse_args()

    src = os.path.abspath(args.source_dir)
    tgt = os.path.abspath(args.target_dir)

    variants = {
        'lite': 'hand_landmark_lite.tflite',
        'full': 'hand_landmark_full.tflite'
    }
    fname = variants[args.model_variant]
    src_path = os.path.join(src, fname)
    if not os.path.isfile(src_path):
        print('Source model not found:', src_path, file=sys.stderr)
        sys.exit(2)

    os.makedirs(tgt, exist_ok=True)
    tgt_path = os.path.join(tgt, fname)

    try:
        shutil.copy2(src_path, tgt_path)
        print('Copied', src_path, '->', tgt_path)
        print('SHA256:', sha256(tgt_path))
    except Exception as e:
        print('Failed to copy:', e, file=sys.stderr)
        if os.path.exists(tgt_path):
            os.remove(tgt_path)
        sys.exit(3)

    # Update README if present
    readme = os.path.join(tgt, 'README.md')
    try:
        with open(readme, 'a') as f:
            f.write('\n- Deployed model: {} (sha256: {})\n'.format(fname, sha256(tgt_path)))
    except Exception:
        # ignore when README not present
        pass

    print('Deployment complete')


if __name__ == '__main__':
    main()
