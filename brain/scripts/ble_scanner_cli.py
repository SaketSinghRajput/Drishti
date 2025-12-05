#!/usr/bin/env python3
"""
Interactive BLE Scanner CLI using bleak.

Scans nearby BLE devices with Bleak, displays a numbered list sorted by RSSI,
allows the user to select one (1-9), and stores the selected MAC address to
`brain/.selected_ble_target` for use in subsequent phases.

Designed for Raspberry Pi / Linux with BlueZ. Defensive against missing
`bleak` and adapter errors.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import os
from typing import List, Optional

try:
    from bleak import BleakScanner, BleakClient
    from bleak.backends.device import BLEDevice
    from bleak.exc import BleakError
except Exception:
    BleakScanner = None  # type: ignore
    BleakClient = None  # type: ignore
    BLEDevice = None  # type: ignore
    BleakError = Exception  # type: ignore

import json
from datetime import datetime, timezone


SELECTED_TARGET_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '.selected_ble_target')
SELECTED_TARGET_FILE = os.path.abspath(SELECTED_TARGET_FILE)


async def scan_devices(scan_duration: float) -> List[BLEDevice]:
    if BleakScanner is None:
        raise RuntimeError("bleak not installed. Install with: pip install bleak")
    print(f"Scanning for {scan_duration}s...")
    try:
        devices = await BleakScanner.discover(timeout=scan_duration)
    except Exception as exc:
        raise RuntimeError(f"BLE scan failed: {exc}")
    return devices


def display_devices(devices: List[BLEDevice]) -> List[BLEDevice]:
    if not devices:
        print("No BLE devices found.")
        return []

    # Sort by RSSI (best first); some devices may have None for rssi
    devices_sorted = sorted(devices, key=lambda d: (d.rssi if d.rssi is not None else -9999), reverse=True)

    print("\n=== Nearby BLE Devices ===")
    for idx, dev in enumerate(devices_sorted, start=1):
        name = dev.name if dev.name else "Unknown"
        addr = dev.address
        rssi = f"{dev.rssi} dBm" if dev.rssi is not None else "N/A"
        print(f"[{idx}] {name} ({addr}) RSSI: {rssi}")
    return devices_sorted


def input_with_timeout(prompt: str, timeout: int) -> Optional[str]:
    # Works on Unix (Raspberry Pi). Falls back to blocking input on other platforms.
    if sys.platform != "linux":
        try:
            return input(prompt)
        except KeyboardInterrupt:
            return None

    import signal

    def _handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        val = input(prompt)
        signal.alarm(0)
        return val
    except TimeoutError:
        print()  # newline after timeout
        return None
    except KeyboardInterrupt:
        print()
        return None
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


def save_selected_mac(mac: str) -> None:
    try:
        d = os.path.dirname(SELECTED_TARGET_FILE)
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(SELECTED_TARGET_FILE, "w", encoding="utf-8") as fh:
            fh.write(mac.strip())
        print(f"MAC Address stored for transmission: {mac}")
        print(f"(Saved to {SELECTED_TARGET_FILE})")
    except Exception as exc:
        print(f"Warning: failed to save selected MAC to file: {exc}")


async def send_safety_packets(mac_address: str, num_packets: int = 5) -> None:
    """Connect to a device and write fake safety packets to the test characteristic.

    This function is defensive: if the device does not expose the expected
    service/characteristic (ESP32 GATT server not implemented yet), it will
    report a helpful message and exit gracefully.
    """
    if BleakClient is None:
        print("Error: bleak BleakClient not available. Install bleak with: pip install bleak")
        return

    service_uuid = "0000FFE0-0000-1000-8000-00805F9B34FB"
    char_uuid = "0000FFE1-0000-1000-8000-00805F9B34FB"

    # New enhanced sender: parameters will be handled by caller; this legacy signature
    # path delegates to the enhanced implementation below for backward compatibility.
    await send_safety_packets_enhanced(
        mac_address,
        num_packets=num_packets,
        continuous=False,
        custom_json_path=None,
        verbose=False,
        retry_attempts=3,
        packet_interval=2.0,
    )


async def send_safety_packets_enhanced(
    mac_address: str,
    *,
    num_packets: int = 5,
    continuous: bool = False,
    custom_json_path: Optional[str] = None,
    verbose: bool = False,
    retry_attempts: int = 3,
    packet_interval: float = 2.0,
) -> None:
    """Enhanced sender with retries, continuous mode, custom JSON and verbose output."""
    if BleakClient is None:
        print("Error: bleak BleakClient not available. Install bleak with: pip install bleak")
        return

    svc_uuid = "0000FFE0-0000-1000-8000-00805F9B34FB"
    char_uuid = "0000FFE1-0000-1000-8000-00805F9B34FB"

    # prepare packet list
    packets = []
    if custom_json_path:
        try:
            packets = load_custom_packets(custom_json_path)
        except Exception as exc:
            print(exc)
            return
    else:
        scenarios = ["normal", "drowsy", "high_risk", "critical_battery", "multi_neighbor"]
        for i in range(num_packets):
            sc = scenarios[i % len(scenarios)]
            packets.append(generate_fake_packet(sc, packet_id=i + 1))

    total_sent = 0
    total_failed = 0
    start_time = time.time()

    attempt = 0
    while attempt < retry_attempts:
        try:
            if verbose:
                print(f"Attempting to connect to {mac_address} (attempt {attempt+1}/{retry_attempts})")
            async with BleakClient(mac_address, timeout=10.0) as client:
                if not client.is_connected:
                    raise RuntimeError("Failed to connect")
                if verbose:
                    print("Connected. Discovering services...")
                svcs = await client.get_services()
                if verbose:
                    print(f"Discovered {len(svcs)} services")
                # check for characteristic
                char = None
                for s in svcs:
                    if s.uuid.lower() == svc_uuid.lower():
                        for c in s.characteristics:
                            if c.uuid.lower() == char_uuid.lower():
                                char = c
                                break
                        if char:
                            break
                if char is None:
                    print(f"Characteristic {char_uuid} not found under service {svc_uuid}. Ensure ESP32 GATT server is implemented.")
                    return

                print(f"Starting to send {('continuous' if continuous else len(packets))} packets to {mac_address}...")
                sent = 0
                try:
                    while True:
                        for idx, pkt in enumerate(packets):
                            if not validate_packet_schema(pkt):
                                print(f"Skipping invalid packet schema: {pkt}")
                                total_failed += 1
                                continue
                            payload = json.dumps(pkt, separators=(",", ":")).encode("utf-8")
                            if len(payload) > 250:
                                print(f"Warning: payload size {len(payload)} bytes may exceed BLE limits")
                            if verbose:
                                print_packet_preview(pkt, verbose=True)
                                print(f"Writing to characteristic {char_uuid} ({len(payload)} bytes)")
                            try:
                                await client.write_gatt_char(char_uuid, payload)
                                sent += 1
                                total_sent += 1
                                print(f"[SENT] Packet {sent}: ts={pkt.get('ts')} vid={pkt.get('vid') or pkt.get('vehicle_hash')}")
                            except Exception as exc:
                                total_failed += 1
                                print(f"[ERROR] Failed to write packet: {exc}")
                            await asyncio.sleep(packet_interval)
                        if not continuous:
                            break
                except KeyboardInterrupt:
                    print("Interrupted by user; stopping send loop")
                duration = time.time() - start_time
                print(f"Finished sending. Sent: {total_sent}, Failed: {total_failed}, Duration: {duration:.1f}s")
                return
        except Exception as exc:
            attempt += 1
            backoff = 2 ** (attempt - 1)
            print(f"Connection attempt failed: {exc}. Retrying in {backoff}s...")
            time.sleep(backoff)
    print(f"Failed to connect after {retry_attempts} attempts")


def generate_fake_packet(scenario: str, packet_id: Optional[int] = None) -> dict:
    """Generate a fake safety packet for testing.

    Accepts an optional `packet_id` which will be embedded as `pid` when provided.
    This keeps callers that pass `packet_id` compatible while not affecting
    validation which doesn't require `pid`.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    base = {
        "vid": "test123",
        "vehicle_hash": "test123",
        "lat": 12.97,
        "lon": 77.59,
        "speed": 11.1,
        "ts": now_ms,
        "battery_severity": "low",
        "max_cell_temp": 36.5,
        "ble_neighbors": 2,
    }
    if scenario == "normal":
        base.update({"fatigue": 18, "risk": 12})
    elif scenario == "drowsy":
        base.update({"fatigue": 65, "risk": 72})
    elif scenario == "high_risk":
        base.update({"fatigue": 85, "risk": 88})
    elif scenario == "critical_battery":
        base.update({"fatigue": 40, "risk": 55, "battery_severity": "critical", "max_cell_temp": 45.0})
    elif scenario == "multi_neighbor":
        base.update({"fatigue": 30, "risk": 28, "ble_neighbors": 8})
    if packet_id is not None:
        # embed, but do not rely on this field for validation
        base["pid"] = int(packet_id)
    return base


def load_custom_packets(path: str) -> List[dict]:
    """Load custom packets from a JSON file.

    The file may contain either a single JSON object (one packet) or an array
    of objects (multiple packets). Returns a list of packet dicts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Custom JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        if not all(isinstance(x, dict) for x in obj):
            raise ValueError("Custom JSON array must contain objects (dicts)")
        return obj
    raise ValueError("Custom JSON must be an object or an array of objects")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_packet_schema(pkt: dict) -> bool:
    """Validate minimal packet schema.

    Required: one of `vid` or `vehicle_hash` (string), `fatigue` (number),
    `risk` (number), `battery_severity` (string), `max_cell_temp` (number),
    `ble_neighbors` (int/number), `lat` (number), `lon` (number), `speed` (number),
    `ts` (int).
    """
    if not isinstance(pkt, dict):
        return False
    if not (isinstance(pkt.get("vid"), str) or isinstance(pkt.get("vehicle_hash"), str)):
        return False
    # numeric checks
    for key in ("fatigue", "risk", "max_cell_temp", "lat", "lon", "speed"):
        if key not in pkt or not _is_number(pkt.get(key)):
            return False
    if "ts" not in pkt or not isinstance(pkt.get("ts"), int):
        return False
    if "battery_severity" not in pkt or not isinstance(pkt.get("battery_severity"), str):
        return False
    if "ble_neighbors" not in pkt or not _is_number(pkt.get("ble_neighbors")):
        return False
    return True


def print_packet_preview(pkt: dict, verbose: bool = False) -> None:
    """Print a concise packet preview when `verbose` is True.

    Shows timestamp, vid (or vehicle_hash), fatigue, risk and neighbor count.
    """
    if not verbose:
        return
    try:
        ts = pkt.get("ts")
        ts_human = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if isinstance(ts, int) else "?"
        vid = pkt.get("vid") or pkt.get("vehicle_hash") or "<unknown>"
        fatigue = pkt.get("fatigue")
        risk = pkt.get("risk")
        neighbors = pkt.get("ble_neighbors")
        battery = pkt.get("battery_severity")
        print(f"[PKT] {ts_human} vid={vid} fatigue={fatigue} risk={risk} neighbors={neighbors} battery={battery}")
    except Exception:
        # best-effort preview; do not raise
        print(f"[PKT] preview unavailable for packet: {pkt}")


def choose_device_interactively(devices: List[BLEDevice], timeout: int) -> Optional[BLEDevice]:
    if not devices:
        return None
    count = len(devices)
    prompt = f"\nSelect device number (1-{count}) or 'q' to quit [auto-select in {timeout}s]: "
    sel = input_with_timeout(prompt, timeout)
    if sel is None:
        # timed out -> choose first device
        chosen = devices[0]
        print(f"No selection made; defaulting to first device: {chosen.name or 'Unknown'} ({chosen.address})")
        return chosen
    sel = sel.strip().lower()
    if sel == 'q':
        return None
    try:
        idx = int(sel)
    except Exception:
        print("Invalid selection. Please enter a number or 'q'.")
        return None
    if idx < 1 or idx > count:
        print("Selection out of range")
        return None
    return devices[idx - 1]


def print_banner():
    print("🔍 BLE Device Scanner (Raspberry Pi)")


def parse_args():
    p = argparse.ArgumentParser(description="Interactive BLE scanner CLI for Raspberry Pi (uses bleak)")
    p.add_argument("--scan-duration", type=float, default=10.0, help="Seconds to scan for BLE devices (default 10)")
    p.add_argument("--timeout", type=int, default=30, help="Seconds to wait for selection input before auto-selecting first device (default 30)")
    p.add_argument("--send-packets", action="store_true", help="After selecting a device, attempt to send test safety packets via GATT")
    p.add_argument("--num-packets", type=int, default=5, help="Number of packets to send (default 5)")
    p.add_argument("--continuous", action="store_true", help="Send packets continuously until interrupted")
    p.add_argument("--custom-json", type=str, default=None, help="Path to custom JSON file with packets (array or single object)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging for connection and packet payloads")
    p.add_argument("--retry-attempts", type=int, default=3, help="Connection retry attempts (default 3)")
    p.add_argument("--packet-interval", type=float, default=2.0, help="Seconds between packets (default 2.0)")
    return p.parse_args()


def main():
    args = parse_args()
    print_banner()

    if BleakScanner is None:
        print("Error: bleak library not available. Install it with: pip install bleak")
        sys.exit(2)

    try:
        devices = asyncio.run(scan_devices(args.scan_duration))
    except Exception as exc:
        print(f"Error during scan: {exc}")
        print("Hint: ensure Bluetooth adapter is up (`sudo hciconfig hci0 up`) and you have necessary permissions.")
        sys.exit(1)

    if not devices:
        print("No BLE devices discovered. Ensure target device is advertising and within range.")
        sys.exit(0)

    devices_sorted = display_devices(devices)
    chosen = choose_device_interactively(devices_sorted, args.timeout)
    if chosen is None:
        print("No device selected. Exiting.")
        sys.exit(0)

    print(f"✅ Selected: {chosen.name or 'Unknown'} ({chosen.address})")
    save_selected_mac(chosen.address)
    if args.send_packets:
        # Attempt to send test packets to the selected device. This will fail
        # gracefully if the device does not expose the expected GATT server
        # / characteristic (ESP32 GATT server is implemented in a future phase).
        if BleakClient is None:
            print("Cannot send packets: BleakClient is not available. Install bleak with: pip install bleak")
            return
        try:
            # Forward CLI options to the enhanced sender so user selections are respected.
            asyncio.run(
                send_safety_packets_enhanced(
                    chosen.address,
                    num_packets=args.num_packets,
                    continuous=args.continuous,
                    custom_json_path=args.custom_json,
                    verbose=args.verbose,
                    retry_attempts=args.retry_attempts,
                    packet_interval=args.packet_interval,
                )
            )
        except Exception as exc:
            print(f"Error while sending packets: {exc}")
            print("Note: the ESP32 must be running a GATT server advertising the write characteristic (next phase).")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
