# BMS Integration Guide

This document explains integration of the Battery Management System (BMS)
into the data collection pipeline.

## Overview

The BMS integration provides CAN and CSV-based ingestion of battery telemetry
and a `BMSHandler` that exposes validated, schema-compliant readings to the
`DataLogger` and `WindowManager`.

## Configuration

Key config is under `battery_monitor.ingestion` in `config/default_config.yaml`.
- `enable_simulation`: use CSV simulator when True (default for development)
- `csv_simulator_path`: path to generated CSV used for simulation
- `can_interface`, `can_bitrate`: used when `enable_simulation` is False

## Usage

Run the example:

```bash
python examples/bms_example.py
```

This will generate a small CSV file, instantiate `BMSHandler` in simulation
mode and print samples to stdout.

## Schema

Telemetry documents conform to `config/bms_schema.json`. Typical fields:
- timestamp (ISO 8601)
- pack_voltage (V)
- pack_current (A)
- soc (%), soh (%)
- cell_voltages (array of volts)
- cell_temps (array of Celsius)
- balancer_status (array of ints)
- fault_codes (array of strings)

## CAN Mode

When deploying to hardware, set `enable_simulation` to False and configure
`can_interface` (e.g., `can0`) and `can_bitrate`. The `CANIngestion` class
will attempt to open a SocketCAN interface via `python-can`. Ensure the OS
is configured for SocketCAN or equivalent.

## Testing

Unit tests for the components are under `tests/test_data_collection/`.

## Power Bus Monitoring (VoltageCurrentHandler)

The `VoltageCurrentHandler` provides pack-level voltage, current, temperature, and power readings via I2C ADC chips (INA219 for V/I, ADS1115 for temp), complementing cell-level BMS data from CAN. This enables detection of pack-level anomalies like voltage sags, current spikes, and thermal events.

Configuration example:

```yaml
sensors:
	voltage_current:
		enable: true
		enable_simulation: false
		i2c_bus: 1
		ina219_address: 0x40
		ads1115_address: 0x48
```

Simulation mode uses `data/voltage_current_simulation.csv` for dev and CI.

## Pack-level thresholds

The system supports YAML-configurable pack-level thresholds used by the `RuleEngine`. Add or modify these keys in `config/bms_rules.yaml` under `thresholds`:

- `pack_voltage_min_v`, `pack_voltage_max_v`
- `pack_current_spike_a`
- `pack_temp_max_c`
- `voltage_drop_threshold_v`

These thresholds affect pack-level rule checks (voltage out-of-range, current spikes, temp exceedances, and voltage-drop detection).
