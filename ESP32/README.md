# ESP32 BLE Safety Packet Receiver

## Overview

Purpose: Receive EV Safety Guardian BLE broadcasts and display on Serial CLI.
Compatible with: ESP32-WROOM-32, ESP32-DevKitC, ESP32-S3.
Communication: BLE advertisement scanning (no pairing required).
Output: Serial CLI at 115200 baud.

## Device Name, Discoverability and GATT Server

The sketch configures the BLE local name to **"node-1"** via `BLEDevice::init("node-1")` in `ble_receiver.ino` and now runs a BLE GATT server so the board advertises and is discoverable by external scanners. The device exposes a service and writable characteristic (see **GATT Server** section) so a Raspberry Pi or other client can connect and write JSON safety packets.

Verification notes (after upload):

```bash
# List BLE advertisers and look for "node-1"
sudo hcitool lescan

# Use the interactive scanner on the Pi to find and connect to node-1
python brain/scripts/ble_scanner_cli.py --scan-duration 8
```

Other discovery tools:
- nRF Connect (mobile app): Scan and look for "node-1".
- Python Bleak: `asyncio.run(BleakScanner.discover())` and filter results by `device.name == "node-1"`.

To change the advertised name, edit the parameter passed to `BLEDevice::init()` in `ble_receiver.ino` and re-upload the sketch.

## Hardware Requirements

- ESP32 development board (any variant with BLE)
- USB cable for programming and Serial monitor

## Software Requirements

- Arduino IDE 2.x or PlatformIO
- ESP32 board support installed (see Espressif docs)
- ArduinoJson library v6.21+ (PlatformIO uses lib_deps in platformio.ini)

## Installation Steps

### Arduino IDE

1. Add ESP32 board URL in Preferences and install ESP32 cores.
2. Install ArduinoJson (v6.21+).
3. Open `ble_receiver.ino` and upload to your ESP32 board.
4. Open Serial Monitor at 115200 baud.

### PlatformIO

From the `ESP32/` directory:

```powershell
pio run -t upload
pio device monitor
```

## Packet Format

Matches `brain/main.py` `build_safety_packet()` output. Example full packet:

```json
{
  "vid": "ab12cd34",
  "fatigue": 62,
  "risk": 64,
  "battery_severity": "low",
  "max_cell_temp": 41.2,
  "ble_neighbors": 3,
  "lat": 12.9715987,
  "lon": 77.594566,
  "speed": 11.11,
  "ts": 1733318400000
}
```

Compact keys supported: `v` (vid), `f` (fatigue), `r` (risk) for advertisement size optimization.

## Sender Compatibility


The production sender (`brain/main.py` + `brain/utils/ble_utils.py`) uses the field name `"vehicle_hash"` for the vehicle identifier. Test packets may use `"vid"` or the compact `"v"` key. The receiver accepts all three aliases (`vehicle_hash`, `vid`, `v`) so it remains compatible with both sender implementations and test data.

## GATT Server

The ESP32 now runs a BLE GATT server exposing service UUID `0000FFE0-0000-1000-8000-00805F9B34FB` and a writable characteristic UUID `0000FFE1-0000-1000-8000-00805F9B34FB`.

- Service: `0000FFE0-0000-1000-8000-00805F9B34FB`
- Writable characteristic: `0000FFE1-0000-1000-8000-00805F9B34FB` (JSON payload writes accepted)

Behavior:
- The device advertises as `node-1` and continues to scan for manufacturer-advertised packets.
- When a remote client connects and writes JSON to the characteristic, the sketch uses the same `parseAndValidatePacket()` and `printPacket()` code paths as for advertisement-parsed packets. GATT-written packets are treated the same and increment the same packet counters.

This dual-mode setup enables end-to-end testing: the Raspberry Pi can scan, select `node-1`, connect, and write safety packets which will be printed on the ESP32 Serial CLI.

## Security Configuration

For developer convenience the ESP32 is configured to accept connections without requiring pairing/bonding. This is useful for rapid testing with the Raspberry Pi sender but is NOT recommended for production.

- Authentication mode: `ESP_LE_AUTH_NO_BOND` (no bonding/pairing)
- I/O capability: `ESP_IO_CAP_NONE` (no passkey/PIN) 
- Encryption keys: standard init encryption and ID keys (`ESP_BLE_ENC_KEY_MASK | ESP_BLE_ID_KEY_MASK`)

Behavior:
- Clients (for example, the Raspberry Pi `bleak` client) can connect and write to the GATT characteristic without running `bluetoothctl pair` first.
- For production use, remove or change these settings and require bonding/MITM as appropriate.

On boot the ESP32 prints a status line indicating the security configuration:

```
[GATT] Security: No bonding, open connections
```

If you see this line on Serial Monitor, the device is accepting pairing-free connections.

BLE Security Modes
------------------

You can control how the ESP32 configures BLE security at compile time by setting
the `BLE_SECURITY_MODE` macro in `ble_receiver.ino` (default: `1`). Modes:

- `0` = `BLESecurity` disabled (open access). No explicit security setup is performed.
- `1` = Minimal: `ESP_LE_AUTH_NO_BOND`, `ESP_IO_CAP_NONE`, and *no* init encryption keys. Recommended first choice when testing with various BlueZ/bleak stacks.
- `2` = Full (original): `ESP_LE_AUTH_NO_BOND`, `ESP_IO_CAP_NONE`, and `setInitEncryptionKey(ESP_BLE_ENC_KEY_MASK | ESP_BLE_ID_KEY_MASK)`. This may trigger encryption negotiation on some hosts.

To change the mode, edit the top of `ble_receiver.ino`:

```cpp
// #define BLE_SECURITY_MODE 0  // no BLESecurity
#define BLE_SECURITY_MODE 1    // minimal (default)
// #define BLE_SECURITY_MODE 2  // original encryption mask
```

If you experience pairing prompts or connection failures, try mode `1` first, then `0` if problems persist. Rebuild and upload after changing the macro.

## Expected Serial Output

```
[ESP32] EV Safety BLE Receiver Started
[GATT] Server started, advertising as 'node-1'
[EV-PACKET] 2025-12-04T12:34:56+0530 VID: ab12cd34 FATIGUE: 62% RISK: 64.3% (HIGH) BATTERY: max_temp=41.2C severity=low BLE_NEIGHBORS: 3 LOCATION: 12.971599, 77.594566 speed=11.11 m/s RX_LATENCY: 120 ms TS_VALID: yes
[SUMMARY] Unique VIDs last 30s: 3 | Total packets: 15
[ERROR] JSON parse failed: InvalidInput
[WARN] Stale packet: ts=1699999990000 age=12500ms
```

## Testing

- With actual EV Safety Guardian sender (Raspberry Pi): run the sender and watch serial output.
- With simulated packets: use the `safety_packet.json` examples or use nRF Connect to broadcast manufacturer data (0xFFFF).

#### Testing with Real Sender

To verify with the actual EV Safety Guardian sender:

1. On the sender machine (Raspberry Pi or other host), start the sender in development mode:

```powershell
python brain/main.py --config config_examples/pi4_minimal.yaml --mode dev
```

2. On the ESP32, open the Serial Monitor at 115200 baud. Look for `[EV-PACKET]` lines indicating received packets.

3. In the sender logs you should see the vehicle id/hash being broadcast. Verify the `VID` shown on the ESP32 matches the `vehicle_hash` in the sender logs (or the hashed vehicle id printed by the sender).

Troubleshooting tips:
- Keep the ESP32 and sender device within ~10 meters during testing to avoid BLE range issues.
- If no packets appear, confirm the sender is advertising manufacturer data with company id `0xFFFF` and that Bluetooth on the host is enabled.
- For Windows hosts, use the sender's logging and a mobile app (nRF Connect) to verify advertisements if low-level HCI tools are unavailable.

### Testing with Raspberry Pi GATT Sender

To test end-to-end GATT writes from the Raspberry Pi sender to the ESP32 GATT server:

1. On the Raspberry Pi, run the interactive scanner/sender and select `node-1` when prompted:

```powershell
python brain/scripts/ble_scanner_cli.py --send-packets --num-packets 5 --packet-interval 1.0 --verbose
```

2. The Pi will scan, list nearby devices (you should see `node-1`), and connect to the selected device.
3. The Pi writes the configured number of JSON packets to characteristic `0000FFE1` and prints send status.
4. On the ESP32 Serial Monitor you should see one `[GATT] Server started, advertising as 'node-1'` line followed by `[EV-PACKET]` lines for each received packet.

If the Pi cannot find `node-1`, use `sudo hcitool lescan` or nRF Connect to verify the ESP32 is advertising.

Note: With the default test configuration the ESP32 does not require pairing. You should be able to connect from the Pi without `bluetoothctl pair`.

## End-to-End Integration Testing

The repository includes a small automated integration script that scans for `node-1`, auto-selects it (or falls back to the first discovered device), and attempts to send 10 test safety packets to the ESP32 GATT server. This script is intended to verify the full flow quickly from the Raspberry Pi to the ESP32.

Run the integration test on the Raspberry Pi (with `bleak` installed):

```powershell
python brain/scripts/test_ble_integration.py
```

Expected flow: Run on Pi → Check ESP32 Serial for `[EV-PACKET]` output. The script will scan (~15s), connect to `node-1`, and write 10 packets. On the ESP32 Serial Monitor you should observe a `[GATT] Client connected` line followed by 10 `[EV-PACKET]` lines. If the script cannot find the writable characteristic or fails to send packets it will print a clear message — in that case verify the ESP32 is running the GATT server (service `0000FFE0`, characteristic `0000FFE1`) and that `BLE_SECURITY_MODE` allows the Pi to connect.

Quick connection-only test (no writes):

```powershell
python -c "import asyncio; from bleak import BleakClient; asyncio.run(BleakClient('MAC').connect())"
```

Replace `MAC` with the target device MAC. A successful connect will return without an exception. If pairing is still required, verify the ESP32 Serial Monitor shows the security line above and restart both devices.

### DEBUG Mode (raw hex + parsed JSON)

To enable debug output that prints raw advertisement bytes and a parsed-structure echo, open `ble_receiver.ino` and uncomment the `#define DEBUG` line near the top of the file, then recompile and upload the sketch to your ESP32.

When enabled you will see additional lines in the Serial monitor for each advertisement containing the manufacturer data or service data, for example:

```
[DEBUG-HEX] Manufacturer Data: 28 bytes
  0xFF 0xFF 0x7B 0x22 0x76 0x69 0x64 0x22 0x3A 0x22 0x61 0x62 0x31 0x32 0x63 0x64  "..{"vid":"ab12cd"
  0x33 0x34 0x22 0x2C 0x22 0x66 0x61 0x74 0x69 0x67 0x75 0x65 0x22 0x3A 0x36 0x32  "34","fatigue":62"

[DEBUG-JSON] {vid:"ab12cd34", fatigue:62, risk:64.3, battery_severity:"low", max_cell_temp:41.2, ble_neighbors:3, lat:12.971599, lon:77.594566, speed:11.11, ts:1733318400000 }
```

Notes:
- Keep `#define DEBUG` commented out for production builds to avoid extra serial traffic and any runtime overhead.
- The debug hex output shows the raw bytes (company ID 0xFFFF followed by the JSON payload starting at offset 2). The parsed JSON echo shows the fields the receiver extracted for verification.

## Configuration

- `SAFETY_MANUFACTURER_ID` and `PACKET_SERVICE_UUID` are defined in the sketch.
- Adjust `SCAN_INTERVAL_S` and `SUMMARY_INTERVAL_MS` for trade-offs between latency and power.

## Troubleshooting

- If no packets: ensure sender is broadcasting to manufacturer ID 0xFFFF.
- JSON parse errors: verify the sender JSON matches the schema in this README.
- Stale packets: sync ESP32 time via NTP or increase `TS_TOLERANCE_MS`.

## Future Enhancements

- BLE Mesh / ESP-IDF integration
- HMAC authentication of packets
- CBOR encoding for smaller payloads
- Local storage and OLED display

This project is intended to be compatible with the EV Safety Guardian sender implementations in `brain/data_collection/sensors/ble_fatigue_broadcast.py` and `brain/utils/ble_utils.py`.
