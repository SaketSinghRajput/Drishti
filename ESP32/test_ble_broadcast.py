#!/usr/bin/env python3
"""
Simple BLE manufacturer-data broadcaster for testing the ESP32 receiver.

This script targets Linux with BlueZ installed and uses `hcitool`/`hciconfig` HCI commands
to set the advertising data. It constructs an AD structure with Manufacturer Specific
Data (AD type 0xFF), company id 0xFFFF (LSB first), and the JSON payload starting at
byte offset 2 as expected by `ble_receiver.ino`.

Usage:
  python test_ble_broadcast.py --json safety_packet.json --packet full_packet --iface hci0 --interval 5

Notes:
- This script requires `sudo` privileges to run hci commands.
- Advertising payloads are limited to 31 bytes total. Manufacturer payload (JSON) must
  fit within a single AD element (approximately 27 bytes for JSON payload). The script
  will warn and refuse if the payload is too large.
- On non-Linux platforms this script will exit with an explanatory message.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time


def check_linux_tools():
    # Ensure we have the needed tools
    for cmd in ("hciconfig", "hcitool"):
        if not shutil.which(cmd):
            print(f"Required tool '{cmd}' not found in PATH. Install BlueZ tools.")
            return False
    return True


def build_adv_bytes(company_id, json_bytes):
    # company_id is 16-bit int, little-endian in manufacturer data
    comp_lo = company_id & 0xFF
    comp_hi = (company_id >> 8) & 0xFF
    # AD structure: [length][type=0xFF][company_lo][company_hi][json bytes...]
    ad_type = 0xFF
    payload_len = len(json_bytes)
    # within the AD structure, the length byte counts the following bytes (type + data)
    ad_length = 1 + 2 + payload_len  # type (1) + company id (2) + payload
    # Total advertising data (sum of AD structures including each AD length byte) must be <=31
    if ad_length + 1 > 31:  # +1 because ad_length itself consumes one byte
        return None, 31 - (1 + 1 + 2)  # return max payload allowed approx
    # Build advertising data bytes (we will pad to 31 bytes when sending HCI command)
    adv = bytearray()
    adv.append(ad_length)
    adv.append(ad_type)
    adv.append(comp_lo)
    adv.append(comp_hi)
    adv.extend(json_bytes)
    return adv, None


def hci_send_adv_data(iface, adv_bytes):
    # HCI LE Set Advertising Data command (OGF=0x08 OCF=0x0008)
    # Command parameters: Advertising_Data_Length (1 byte) + Advertising_Data (31 bytes)
    adv_len = len(adv_bytes)
    # Advertising_Data must be exactly 31 bytes long for the HCI call; pad with zeros
    adv_padded = adv_bytes + bytes(31 - adv_len)
    # Build command: hcitool -i <iface> cmd 0x08 0x0008 <adv_len> <bytes...>
    cmd = [
        "sudo",
        "hcitool",
        "-i",
        iface,
        "cmd",
        "0x08",
        "0x0008",
        f"{adv_len:02x}",
    ]
    cmd += [f"{b:02x}" for b in adv_padded]
    print("Running:", " ".join(cmd[:6]), "[data bytes elided]")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res


def hci_set_advertise_enable(iface, enable=True):
    # LE Set Advertise Enable (OCF 0x000A) param: 0x00 disable, 0x01 enable
    val = "01" if enable else "00"
    cmd = ["sudo", "hcitool", "-i", iface, "cmd", "0x08", "0x000a", val]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res


def main():
    parser = argparse.ArgumentParser(description="Broadcast manufacturer-data BLE adverts for testing ESP32 receiver")
    parser.add_argument("--json", default="safety_packet.json", help="Path to safety_packet.json")
    parser.add_argument("--packet", default="full_packet", help="Key of packet to broadcast from the JSON file")
    parser.add_argument("--iface", default="hci0", help="HCI interface to use (default hci0)")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between re-issuing advertisement (default 5s)")
    parser.add_argument("--company", type=lambda x: int(x, 0), default=0xFFFF, help="Manufacturer company id (default 0xFFFF)")
    args = parser.parse_args()

    if platform.system() != "Linux":
        print("This broadcaster currently supports Linux with BlueZ (hcitool/hciconfig).")
        print("Use nRF Connect or an alternative advertiser on other platforms.")
        sys.exit(1)

    if not check_linux_tools():
        sys.exit(1)

    if not os.path.exists(args.json):
        print(f"JSON file not found: {args.json}")
        sys.exit(1)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.packet not in data:
        print(f"Packet key '{args.packet}' not found in {args.json}. Available keys: {list(data.keys())}")
        sys.exit(1)

    pkt_obj = data[args.packet]
    # Ensure timestamps are int if present
    if "ts" not in pkt_obj:
        pkt_obj["ts"] = int(time.time() * 1000)

    json_payload = json.dumps(pkt_obj, separators=(",", ":"))
    json_bytes = json_payload.encode("utf-8")
    print(f"Selected packet '{args.packet}', payload {len(json_bytes)} bytes: {json_payload}")

    adv_bytes, max_payload_warn = build_adv_bytes(args.company, json_bytes)
    if adv_bytes is None:
        print("Payload too large for a single advertising AD element.")
        print(f"Approximately {max_payload_warn} bytes available for JSON payload in a single manufacturer AD (31-byte advertising data limit).")
        sys.exit(1)

    # Bring up interface
    print(f"Bringing up {args.iface}...")
    subprocess.run(["sudo", "hciconfig", args.iface, "up"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Set advertising data
    res = hci_send_adv_data(args.iface, adv_bytes)
    if res.returncode != 0:
        print("Failed to set advertising data:")
        print(res.stdout.decode(errors="ignore"))
        print(res.stderr.decode(errors="ignore"))
        sys.exit(1)
    print("Advertising data set.")

    # Enable advertising
    res = hci_set_advertise_enable(args.iface, enable=True)
    if res.returncode != 0:
        print("Failed to enable advertising:")
        print(res.stdout.decode(errors="ignore"))
        print(res.stderr.decode(errors="ignore"))
        sys.exit(1)
    print("Advertising enabled. Broadcasting now.")

    try:
        while True:
            # Re-set advertising periodically so that receivers that start later get the data.
            time.sleep(args.interval)
            res = hci_send_adv_data(args.iface, adv_bytes)
            if res.returncode == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Re-set advertisement payload ({len(json_bytes)} bytes)")
            else:
                print("Warning: failed to re-set advertisement payload")
    except KeyboardInterrupt:
        print("Disabling advertising and exiting...")
        hci_set_advertise_enable(args.iface, enable=False)
        sys.exit(0)


if __name__ == '__main__':
    main()
