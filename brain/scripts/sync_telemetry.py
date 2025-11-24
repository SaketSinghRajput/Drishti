"""CLI to sync telemetry summaries to remote endpoint using CloudSyncManager."""
from __future__ import annotations

import argparse
import sys

from config.config_loader import ConfigLoader
from utils.cloud_sync import CloudSyncManager


def main():
    cfg = ConfigLoader()
    parser = argparse.ArgumentParser(description='Sync telemetry summaries to cloud')
    parser.add_argument('--telemetry-dir', default=cfg.get('telemetry.output_dir', 'data/telemetry'))
    parser.add_argument('--endpoint', default=None)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # override config values if provided
    if args.endpoint:
        cfg.set('telemetry.cloud_sync.endpoint_url', args.endpoint)
    if args.api_key:
        cfg.set('telemetry.cloud_sync.api_key', args.api_key)

    mgr = CloudSyncManager(cfg)
    if args.dry_run:
        files = mgr.get_pending_files()
        print('Pending files:', len(files))
        if files:
            print('\n'.join(files))
        sys.exit(0)

    res = mgr.sync()
    print('Sync result:', res)


if __name__ == '__main__':
    main()
