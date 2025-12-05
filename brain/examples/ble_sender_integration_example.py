#!/usr/bin/env python3
"""Integration example: programmatically scan, select `node-1`, and send packets.

This is a non-interactive example that demonstrates how to reuse the scanner
logic to find a device named `node-1` and then send a small burst of packets.
"""
from __future__ import annotations

import asyncio
import os
import sys
import importlib.util

THIS_DIR = os.path.dirname(os.path.dirname(__file__))
CLI_PATH = os.path.join(THIS_DIR, 'scripts', 'ble_scanner_cli.py')
if not os.path.exists(CLI_PATH):
    print(f"Cannot find ble_scanner_cli.py at expected path: {CLI_PATH}")
    sys.exit(2)

spec = importlib.util.spec_from_file_location("ble_scanner_cli", CLI_PATH)
ble_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(ble_mod)

SCAN_DURATION = 8.0
TARGET_NAME = 'node-1'


async def run_example():
    print(f"Scanning for {SCAN_DURATION}s to find device named '{TARGET_NAME}'...")
    devices = await ble_mod.scan_devices(SCAN_DURATION)
    chosen = None
    for d in devices:
        if (d.name or '').lower() == TARGET_NAME.lower():
            chosen = d
            break
    if not chosen and devices:
        # fallback to first device
        chosen = devices[0]
        print(f"Did not find '{TARGET_NAME}', falling back to first device: {chosen.name} ({chosen.address})")
    if not chosen:
        print("No BLE devices found; ensure adapter is up and device is advertising.")
        return
    print(f"Selected device: {chosen.name} ({chosen.address})")

    # send a small burst of packets
    await ble_mod.send_safety_packets_enhanced(
        chosen.address,
        num_packets=3,
        continuous=False,
        custom_json_path=None,
        verbose=True,
        retry_attempts=2,
        packet_interval=1.0,
    )


if __name__ == '__main__':
    try:
        asyncio.run(run_example())
    except KeyboardInterrupt:
        print("Interrupted by user")
