#!/usr/bin/env python3
"""Automated end-to-end BLE integration test.

Scans for a device named 'node-1', auto-selects it (or falls back to the first
found device), sends a burst of test safety packets using the existing
`ble_scanner_cli` sender logic, and prints instructions to verify the ESP32
Serial Monitor.

Usage:
    python brain/scripts/test_ble_integration.py

This script reuses the scanner and sender functionality implemented in
`ble_scanner_cli.py` so keep both scripts in sync.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time

THIS_DIR = os.path.dirname(__file__)
CLI_PATH = os.path.join(THIS_DIR, "ble_scanner_cli.py")
if not os.path.exists(CLI_PATH):
    print(f"Cannot find ble_scanner_cli.py at expected path: {CLI_PATH}")
    sys.exit(2)

spec = importlib.util.spec_from_file_location("ble_scanner_cli", CLI_PATH)
if spec is None or getattr(spec, "loader", None) is None:
    print(f"Failed to load spec for ble_scanner_cli from: {CLI_PATH}")
    print("Ensure the file exists and is readable. Aborting.")
    sys.exit(2)
ble_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ble_mod)

scan_devices = getattr(ble_mod, "scan_devices")
send_func = getattr(ble_mod, "send_safety_packets_enhanced")

SCAN_DURATION = 15.0
NUM_PACKETS = 10
PACKET_INTERVAL = 1.5
RETRY_ATTEMPTS = 3


async def auto_select_node(scan_duration: float):
    print(f"Scanning for {scan_duration}s to find 'node-1'...")
    devices = await scan_devices(scan_duration)
    if not devices:
        return None
    # Prefer device whose name contains 'node-1' (case-insensitive)
    for d in devices:
        name = getattr(d, 'name', None) or ''
        if 'node-1' in name.lower():
            return d
    # fallback to first device
    print("Did not find 'node-1'; falling back to first discovered device")
    return devices[0]


async def run_test():
    print("=== BLE End-to-End Integration Test ===")
    print("Ensure your ESP32 is powered on, running the latest ble_receiver sketch, and the Serial Monitor is open at 115200 baud.")
    print("This script will scan, auto-select node-1 if available, connect, and send 10 packets.")
    try:
        chosen = await auto_select_node(SCAN_DURATION)
    except Exception as exc:
        print(f"Scan failed: {exc}")
        print("Hint: Ensure Bluetooth adapter is up (sudo hciconfig hci0 up) and bleak is installed in your venv.")
        return

    if not chosen:
        print("No BLE devices found. Ensure the ESP32 is advertising as 'node-1' and in range.")
        return

    name = getattr(chosen, 'name', None) or 'Unknown'
    mac = getattr(chosen, 'address', None)
    print(f"Auto-selected: {name} ({mac})")

    print(f"Sending {NUM_PACKETS} test packets to {mac} (interval {PACKET_INTERVAL}s)...")
    try:
        sent_ok = await send_func(
            mac,
            num_packets=NUM_PACKETS,
            continuous=False,
            custom_json_path=None,
            verbose=True,
            retry_attempts=RETRY_ATTEMPTS,
            packet_interval=PACKET_INTERVAL,
        )
    except KeyboardInterrupt:
        print("Interrupted by user")
        return
    except Exception as exc:
        print(f"Error during send: {exc}")
        return

    if sent_ok:
        print("\nTest complete. Please verify the ESP32 Serial Monitor shows 10 '[EV-PACKET]' lines.")
        print("Expected: one [GATT] Client connected line, 10 [EV-PACKET] lines, and then [GATT] Client disconnected / Advertising restarted.")
    else:
        print("\nTest could not send packets. No packets were written to the target device.")
        print("Check the ESP32 is running the GATT server (service UUID 0000FFE0 and writable characteristic 0000FFE1) and that BLE_SECURITY_MODE allows connections from the Pi.")


if __name__ == '__main__':
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(1)
