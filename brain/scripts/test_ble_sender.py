#!/usr/bin/env python3
"""Non-interactive BLE test sender.

Loads the local `ble_scanner_cli.py` module and reuses its
`send_safety_packets_enhanced` function to send packets to a target.

If `--mac` is not provided, reads the saved selection from
`../.selected_ble_target` (same location used by the interactive scanner).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import importlib.util

THIS_DIR = os.path.dirname(__file__)
CLI_PATH = os.path.join(THIS_DIR, "ble_scanner_cli.py")
if not os.path.exists(CLI_PATH):
    print(f"Cannot find ble_scanner_cli.py at expected path: {CLI_PATH}")
    sys.exit(2)

spec = importlib.util.spec_from_file_location("ble_scanner_cli", CLI_PATH)
ble_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(ble_mod)

SELECTED_TARGET_FILE = getattr(ble_mod, "SELECTED_TARGET_FILE", os.path.abspath(os.path.join(THIS_DIR, '..', '.selected_ble_target')))
send_func = getattr(ble_mod, "send_safety_packets_enhanced")


def parse_args():
    p = argparse.ArgumentParser(description="Non-interactive BLE test sender (reuses ble_scanner_cli)")
    p.add_argument("--mac", type=str, default=None, help="Target MAC address (reads saved selection if omitted)")
    p.add_argument("--num-packets", type=int, default=5)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--custom-json", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--retry-attempts", type=int, default=3)
    p.add_argument("--packet-interval", type=float, default=2.0)
    return p.parse_args()


async def _run_send(target_mac: str, args):
    await send_func(
        target_mac,
        num_packets=args.num_packets,
        continuous=args.continuous,
        custom_json_path=args.custom_json,
        verbose=args.verbose,
        retry_attempts=args.retry_attempts,
        packet_interval=args.packet_interval,
    )


def main():
    args = parse_args()
    mac = args.mac
    if not mac:
        if os.path.exists(SELECTED_TARGET_FILE):
            with open(SELECTED_TARGET_FILE, 'r', encoding='utf-8') as fh:
                mac = fh.read().strip()
                print(f"Loaded target MAC from {SELECTED_TARGET_FILE}: {mac}")
        else:
            print("No target MAC provided and no saved selection found. Use --mac or run the interactive scanner first.")
            sys.exit(2)
    try:
        asyncio.run(_run_send(mac, args))
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == '__main__':
    main()
