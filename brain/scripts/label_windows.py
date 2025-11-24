#!/usr/bin/env python
"""CLI to label windows using the labeling subsystem."""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

from config.config_loader import ConfigLoader
from data_collection.labeling import AccidentDetector, NearMissDetector, TemporalLabeler, ManualAnnotator
from utils import dataset_utils

logger = logging.getLogger(__name__)


def _load_windows(input_dir: str) -> List[dict]:
    out = []
    for fn in sorted(os.listdir(input_dir)):
        if not fn.endswith('.json'):
            continue
        p = os.path.join(input_dir, fn)
        try:
            with open(p, 'r', encoding='utf-8') as fh:
                out.append(json.load(fh))
        except Exception:
            logger.exception('failed to load %s', p)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mode', choices=('auto', 'manual', 'review'), default='auto')
    p.add_argument('--prediction-horizons', help='Override labeling prediction horizons, comma-separated ints', default=None)
    p.add_argument('--filter-trip', help='Process only windows from this trip_id', default=None)
    p.add_argument('--dry-run', help='Run pipeline without writing outputs', action='store_true')
    p.add_argument('--config', default='config/default_config.yaml')
    p.add_argument('--balance', action='store_true')
    p.add_argument('--split', action='store_true')
    args = p.parse_args()

    cfg = ConfigLoader(args.config)
    windows = _load_windows(args.input_dir)

    # optional trip filter
    if args.filter_trip:
        windows = [w for w in windows if w.get('trip_id') == args.filter_trip]

    ad = AccidentDetector(cfg)
    nm = NearMissDetector(cfg)

    # parse prediction horizons override
    horizons = None
    if args.prediction_horizons:
        try:
            horizons = [int(x.strip()) for x in args.prediction_horizons.split(',') if x.strip()]
            logger.info('Using override prediction horizons: %s', horizons)
        except Exception:
            logger.exception('Failed to parse --prediction-horizons, using config defaults')

    events = []
    for w in windows:
        a = ad.detect(w)
        if a:
            # ensure event type field for TemporalLabeler
            ev = {**a, 'type': 'accident', 'subtype': a.get('type'), 'trip_id': w.get('trip_id')}
            events.append(ev)
            logger.info('Accident detected at %s', w.get('timestamp'))

        # call NearMissDetector with awareness of accident in same window
        try:
            nm_detected = nm.detect(w, accident_detected=bool(a))
        except TypeError:
            # older signature compatibility
            nm_detected = nm.detect(w)
        if nm_detected:
            ev2 = {**nm_detected, 'type': 'near_miss', 'trip_id': w.get('trip_id')}
            events.append(ev2)
            logger.info('Near-miss detected at %s', w.get('timestamp'))

    tl = TemporalLabeler(cfg, horizons=horizons)
    labeled = tl.label_windows(windows, events)

    # mode handling
    if args.mode == 'manual':
        ma = ManualAnnotator(cfg)
        out_windows = []
        for w in labeled:
            lab = ma.review_window(w, auto_labels=w.get('labels'))
            # embed returned label fragment under the window's `labels` key
            w_out = dict(w)
            w_out.setdefault('labels', {})
            w_out['labels'].update(lab)
            out_windows.append(w_out)
        if not args.dry_run:
            # validate before export
            try:
                from utils.validation import validate_labeled_dataset

                vs = validate_labeled_dataset(out_windows)
                logger.info('Validation summary: %s', vs)
            except Exception:
                logger.exception('Validation failed or unavailable')
            dataset_utils.export_labeled_dataset(out_windows, args.output_dir)
    elif args.mode == 'review':
        # load existing labels and present those requiring review
        ma = ManualAnnotator(cfg)
        try:
            loaded = dataset_utils.load_labeled_dataset(args.output_dir)
        except Exception:
            logger.exception('Failed to load existing labels from %s', args.output_dir)
            loaded = []

        # filter by nested labels: requires_review or low confidence
        thr = float(cfg.get('labeling', {}).get('confidence_threshold', 0.8)) if hasattr(cfg, 'get') else 0.8
        def needs_review(l):
            lab = (l.get('labels') or {}) if isinstance(l, dict) else {}
            # support legacy top-level keys as fallback
            rq = lab.get('requires_review') if 'requires_review' in lab else l.get('requires_review')
            conf = lab.get('confidence') if 'confidence' in lab else l.get('confidence')
            if rq:
                return True
            try:
                if conf is not None and float(conf) < thr:
                    return True
            except Exception:
                return False
            return False

        to_review = [l for l in loaded if needs_review(l)]
        if not to_review:
            logger.info('No labels requiring review found in %s', args.output_dir)
        updated = []
        for lab in to_review:
            # treat loaded objects as full windows with `labels` nested
            upd_labels = ma.review_window(lab, auto_labels=lab.get('labels'))
            # embed updated labels into the original object
            newobj = dict(lab)
            newobj.setdefault('labels', {})
            newobj['labels'].update(upd_labels)
            updated.append(newobj)
        # merge updated back into loaded by window_id
        if updated and not args.dry_run:
            by_id = {l.get('window_id'): l for l in loaded}
            for u in updated:
                by_id[u.get('window_id')] = u
            merged = list(by_id.values())
            try:
                from utils.validation import validate_labeled_dataset

                vs = validate_labeled_dataset(merged)
                logger.info('Validation summary: %s', vs)
            except Exception:
                logger.exception('Validation failed or unavailable')
            dataset_utils.export_labeled_dataset(merged, args.output_dir)
    else:
        # auto mode
        if not args.dry_run:
            try:
                from utils.validation import validate_labeled_dataset

                vs = validate_labeled_dataset(labeled)
                logger.info('Validation summary: %s', vs)
            except Exception:
                logger.exception('Validation failed or unavailable')
            dataset_utils.export_labeled_dataset(labeled, args.output_dir)
            if args.balance:
                balanced = dataset_utils.balance_dataset_smote(labeled)
                dataset_utils.export_labeled_dataset(balanced, os.path.join(args.output_dir, 'balanced'))
            if args.split:
                train, val, test = dataset_utils.temporal_train_test_split(labeled)
                dataset_utils.export_labeled_dataset(train, os.path.join(args.output_dir, 'train'))
                dataset_utils.export_labeled_dataset(val, os.path.join(args.output_dir, 'val'))
                dataset_utils.export_labeled_dataset(test, os.path.join(args.output_dir, 'test'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
