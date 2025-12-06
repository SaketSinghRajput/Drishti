# BLE Scanner CLI

Overview
--------

This small interactive CLI scans nearby BLE devices using the `bleak` library,
displays a numbered list sorted by signal strength (RSSI), prompts the user to
select one device, and stores the selected MAC address for use in subsequent
phases (e.g. sending safety packets).

Platform
--------

- Target: Raspberry Pi / Linux with BlueZ
- Requires: Python 3.9+ and `bleak` (`pip install bleak`)

Prerequisites
-------------

- Enable Bluetooth adapter: `sudo hciconfig hci0 up`
- Install bleak: `pip install bleak`
- Ensure Python can access BLE adapter (run as root or set required capabilities)
  - Example: `sudo setcap cap_net_raw,cap_net_admin+eip $(which python3)`

Usage
-----

```bash
# Basic scan (10s default)
python brain/scripts/ble_scanner_cli.py

# Custom scan duration
python brain/scripts/ble_scanner_cli.py --scan-duration 20

# Custom selection timeout (seconds)
python brain/scripts/ble_scanner_cli.py --timeout 60
```

## Sending Safety Packets (GATT Mode)

The scanner will automatically send test safety packets to the selected device by default (auto-send). This uses `bleak.BleakClient` to connect and write JSON payloads to the test write characteristic.

**Important**: The ESP32 must be running a GATT server and advertising the service/characteristic for this to succeed.

Auto-Send Behavior (Default)
----------------------------

By default, after you select a device the scanner will connect and send 5 test packets, then exit. To disable that behavior and perform scan-only operation, use `--no-auto-send`.

Examples:

```bash
# Default: scan -> select -> auto-send 5 packets
python brain/scripts/ble_scanner_cli.py

# Disable auto-send (scan only)
python brain/scripts/ble_scanner_cli.py --no-auto-send

# Legacy flag still supported (equivalent to auto-send):
python brain/scripts/ble_scanner_cli.py --send-packets
```

The script will attempt to write packets (normal, drowsy, high_risk, critical_battery, multi_neighbor) to characteristic `0000FFE1-0000-1000-8000-00805F9B34FB` under service `0000FFE0-0000-1000-8000-00805F9B34FB`.

If the characteristic is not found or the device is not connectable, the script will print a helpful message and exit gracefully.

Packet scenarios (examples):

- Normal:
```json
{"vid":"test123","fatigue":18,"risk":12,"battery_severity":"low","max_cell_temp":36.5,"ble_neighbors":2,"lat":12.97,"lon":77.59,"speed":11.1,"ts":1733318400000}
```
- Drowsy:
```json
{"vid":"test123","fatigue":65,"risk":72,...}
```

Troubleshooting:

- "Characteristic not found": ESP32 GATT server not implemented yet; wait for next phase.
- "Connection timeout": device not advertising GATT or out of range.

CLI Send Options
----------------

The scanner includes a set of CLI options that control the sending behavior when
`--send-packets` is provided. These are now supported by both the interactive
scanner and the non-interactive `test_ble_sender.py` script.

- `--num-packets N`: send N packets (default 5)
- `--continuous`: send packets continuously until interrupted
- `--custom-json PATH`: load packets from a JSON file (object or array)
- `--verbose`: print packet previews and connection details
- `--retry-attempts N`: number of connection retries (default 3)
- `--packet-interval S`: seconds between packets (default 2.0)

Auto-Send Options
-----------------

- `--auto-send`: enable automatic sending after device selection (default enabled)
- `--no-auto-send`: disable automatic sending after device selection (scan-only mode)
- `--send-packets`: legacy flag to trigger sending after selection (kept for compatibility)

Examples:

```bash
# Send 10 packets with 1s between writes
python brain/scripts/ble_scanner_cli.py --send-packets --num-packets 10 --packet-interval 1.0

# Send continuously until Ctrl-C
python brain/scripts/ble_scanner_cli.py --send-packets --continuous

# Use a custom JSON file containing an array of packet objects
python brain/scripts/ble_scanner_cli.py --send-packets --custom-json ./my_packets.json --verbose
```

Non-interactive sender
----------------------

Use `brain/scripts/test_ble_sender.py` to send packets to a previously selected
target (saved in `brain/.selected_ble_target`) or to a specific MAC using
`--mac`. Example:

```bash
python brain/scripts/test_ble_sender.py --num-packets 5 --packet-interval 1.0
```

Expected ESP32 Serial Output
---------------------------

When the ESP32 receiver is extended to accept GATT writes and log packets,
the expected serial output for each received packet follows the `[EV-PACKET]`
format. Example:

```
[EV-PACKET] ts=1733318400000 vid=test123 fatigue=65 risk=72 battery=low neighbors=2
```

This line indicates the device parsed and validated an incoming safety packet.
Use this to confirm end-to-end operation when the ESP32 GATT server is active.


Expected Output
---------------

```
🔍 BLE Device Scanner (Raspberry Pi)
Scanning for 10s...
Found 3 devices

=== Nearby BLE Devices ===
[1] node-1 (AA:BB:CC:DD:EE:FF) RSSI: -65 dBm
[2] iPhone (BB:CC:DD:EE:FF:00) RSSI: -72 dBm
[3] Unknown (CC:DD:EE:FF:00:11) RSSI: -78 dBm

Select device number (1-3) or 'q' to quit [auto-select in 30s]: 1
✅ Selected: node-1 (AA:BB:CC:DD:EE:FF)
MAC Address stored: AA:BB:CC:DD:EE:FF
(Saved to brain/.selected_ble_target)
```

Troubleshooting
---------------

- `No module named 'bleak'`: install bleak with `pip install bleak`.
- `BLE adapter not found`: check `hciconfig` and enable the adapter with `sudo hciconfig hci0 up`.
- `Permission denied`: run the script with `sudo` or set capabilities as described above.
- `No devices found`: ensure target ESP32 or other BLE advertiser is powered and in range. Note that the current ESP32 sketch (`ble_receiver.ino`) is scanner-only and does not advertise; advertising will be added in a later phase.

Integration with ESP32
----------------------

- The scanner can discover any BLE advertiser (phones, laptops, BLE peripherals).
- For the ESP32 to be discovered by this scanner it must be advertising (GATT server + advertise). The current `ble_receiver.ino` firmware configures the local name but does not yet start advertising; discovery of `node-1` will work after the advertising/GATT-server phase is implemented.

Next Steps
----------

- The selected MAC address is saved to `brain/.selected_ble_target` for use by the sender in the next phase (GATT write of safety packets).
- Future work: implement sender that reads the saved MAC and performs GATT writes to transmit safety packets as documented in `brain/data_collection/sensors/ble_fatigue_broadcast.py`.
