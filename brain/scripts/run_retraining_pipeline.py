#!/usr/bin/env python
"""CLI to run the retraining pipeline (lightweight wrapper).

This script provides subcommands: full, validate, deploy, rollback, schedule.
The implementation here is a thin wrapper around `models.retrain_pipeline.RetrainingPipeline`.
"""
import argparse
import logging
import os
import json

from models.retrain_pipeline import RetrainingPipeline
from config.config_loader import ConfigLoader
import time


def main():
    p = argparse.ArgumentParser(description='Run retraining pipeline')
    sub = p.add_subparsers(dest='cmd')

    s_full = sub.add_parser('full')
    s_full.add_argument('--dataset-csv', required=True)
    s_full.add_argument('--output-dir', required=True)
    s_full.add_argument('--incremental', action='store_true')
    s_full.add_argument('--auto-deploy', action='store_true')
    s_full.add_argument('--horizons', nargs='+', type=int)
    s_full.add_argument('--skip-approval', action='store_true')
    s_full.add_argument('--dry-run', action='store_true')

    s_schedule = sub.add_parser('schedule')
    s_schedule.add_argument('--config', required=False)

    s_rollback = sub.add_parser('rollback')
    s_rollback.add_argument('--version', required=True)

    s_val = sub.add_parser('validate')
    s_val.add_argument('--dataset-csv', required=True)

    s_deploy = sub.add_parser('deploy')
    s_deploy.add_argument('--model-id', type=int, required=True)
    s_deploy.add_argument('--skip-approval', action='store_true')

    s_rollback.add_argument('--horizon', type=int, required=False)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg_path = getattr(args, 'config', None)
    cfg_loader = ConfigLoader(cfg_path) if cfg_path else ConfigLoader()
    cfg = cfg_loader.as_dict()
    pipeline = RetrainingPipeline(cfg)

    if args.cmd == 'full':
        # respect flags
        if getattr(args, 'auto_deploy', False):
            cfg.setdefault('retraining', {}).setdefault('deployment', {})['auto_deploy'] = True
        r = pipeline.run_full_pipeline(args.dataset_csv, args.output_dir, incremental=getattr(args, 'incremental', True))
        print(json.dumps(r, indent=2))
    elif args.cmd == 'validate':
        r = pipeline.run_validation_only(args.dataset_csv)
        print(json.dumps(r, indent=2))
    elif args.cmd == 'deploy':
        r = pipeline.run_deployment_only(args.model_id)
        print(json.dumps(r, indent=2))
    elif args.cmd == 'schedule':
        sched = pipeline.scheduler
        try:
            while True:
                if sched.should_trigger_retraining():
                    ok = True
                    if sched.approval_required and not getattr(args, 'skip_approval', False):
                        ok = sched.check_approval()
                    if ok:
                        logging.info('Triggering scheduled retraining')
                        # dataset path resolution could be configured; use retraining.output_dir as placeholder
                        ds = cfg.get('retraining', {}).get('output_dir') or 'data/retraining'
                        outdir = ds
                        pipeline.run_full_pipeline(ds, outdir)
                        sched.update_last_run()
                time.sleep(max(60, int(sched.check_interval_hours * 3600)))
        except KeyboardInterrupt:
            logging.info('Schedule loop stopped by user')
    elif args.cmd == 'rollback':
        # minimal rollback: find model by version and mark as deployed
        version = args.version
        horizon = getattr(args, 'horizon', None)
        mr = pipeline.registry.get_model_by_version(version, horizon)
        if not mr:
            print(json.dumps({'ok': False, 'reason': 'model_not_found'}))
        else:
            pipeline.registry.update_model_status(mr['id'], 'deployed')
            print(json.dumps({'ok': True, 'model': mr}))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
