"""BMSCANParser: parse LibreSolar bms-c1 CAN frames and normalize telemetry.

This implementation contains a best-effort parser with placeholder CAN ID
maps. Exact CAN ID mappings should be aligned to LibreSolar bms-c1 docs
in a subsequent phase. The parser maintains internal state for multi-frame
cell arrays.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import struct
import logging
import time
from utils.time_utils import get_utc_timestamp

logger = logging.getLogger(__name__)


class BMSCANParser:
    def __init__(self, config=None):
        self.config = config or {}
        # cell count (default 13 for 48V nominal)
        self._cell_count: int = int(self.config.get('battery_monitor.ingestion.cells', 13))
        # internal buffers for multi-frame messages
        self._cell_voltage_buffer: List[float] = []
        self._cell_temp_buffer: List[float] = []
        # last-complete snapshots
        self._last_cell_voltages: Optional[List[float]] = None
        self._last_cell_temps: Optional[List[float]] = None
        # timestamps for last partial receipts (for timeout logic)
        self._cell_voltage_ts: Optional[float] = None
        self._cell_temp_ts: Optional[float] = None
        # last pack-level fields
        self._last_pack_fields: Dict[str, Any] = {}
        # balancer and fault masks / lists
        self._last_balancer_list: Optional[List[int]] = None
        self._last_fault_codes: Optional[List[str]] = None

    def parse_message(self, can_id: int, data: bytes) -> Dict[str, Any]:
        """Parse a CAN frame into a partial telemetry dict.

        NOTE: CAN ID mappings are placeholders and should be replaced with
        accurate values from LibreSolar bms-c1 documentation.
        """
        try:
            # Example mapping (placeholders):
            # 0x100: pack data (voltage mV, current mA, soc promille, soh promille)
            # 0x200: cell voltages (multiple frames)
            # 0x300: cell temps
            if can_id == 0x100:
                # expect 8 bytes: uint16 voltage_mV, int16 current_mA, uint16 soc_promille, uint16 soh_promille
                if len(data) >= 8:
                    v_mv, i_ma, soc_prom, soh_prom = struct.unpack('<HhHH', data[0:8])
                    return {
                        'pack_voltage_mv': int(v_mv),
                        'pack_current_ma': int(i_ma),
                        'soc_promille': int(soc_prom),
                        'soh_promille': int(soh_prom),
                        'timestamp': get_utc_timestamp(),
                    }
            elif 0x200 <= can_id < 0x210:
                # cell voltages frame: each frame holds up to 4 uint16 mV values
                vals = []
                for i in range(0, len(data), 2):
                    if i + 2 <= len(data):
                        (mv,) = struct.unpack_from('<H', data, i)
                        vals.append(mv / 1000.0)
                # append to buffer and record timestamp
                self._cell_voltage_buffer.extend(vals)
                self._cell_voltage_ts = time.time()
                # if we've collected enough values for all cells, snapshot and clear buffer
                if len(self._cell_voltage_buffer) >= self._cell_count:
                    self._last_cell_voltages = self._cell_voltage_buffer[: self._cell_count]
                    # keep any overflow for next cycle
                    self._cell_voltage_buffer = self._cell_voltage_buffer[self._cell_count :]
                    return {'cell_voltages_complete': list(self._last_cell_voltages), 'timestamp': get_utc_timestamp()}
                return {'cell_voltages_partial': vals, 'timestamp': get_utc_timestamp()}
            elif 0x300 <= can_id < 0x310:
                # cell temps frame: int8 per temp in degC
                temps = []
                for b in data:
                    # treat as signed byte
                    t = struct.unpack('<b', bytes([b]))[0]
                    temps.append(float(t))
                self._cell_temp_buffer.extend(temps)
                self._cell_temp_ts = time.time()
                if len(self._cell_temp_buffer) >= self._cell_count:
                    self._last_cell_temps = self._cell_temp_buffer[: self._cell_count]
                    self._cell_temp_buffer = self._cell_temp_buffer[self._cell_count :]
                    return {'cell_temps_complete': list(self._last_cell_temps), 'timestamp': get_utc_timestamp()}
                return {'cell_temps_partial': temps, 'timestamp': get_utc_timestamp()}
            elif can_id == 0x400:
                # balancer status bitmask (example placeholder)
                # interpret as little-endian unsigned integer
                mask = 0
                for i, b in enumerate(data):
                    mask |= (b & 0xFF) << (8 * i)
                bal = [(mask >> i) & 1 for i in range(self._cell_count)]
                self._last_balancer_list = bal
                return {'balancer_mask': mask, 'balancer_list': list(bal), 'timestamp': get_utc_timestamp()}
            elif can_id == 0x410:
                # fault bitmask (placeholder mapping)
                mask = 0
                for i, b in enumerate(data):
                    mask |= (b & 0xFF) << (8 * i)
                # map bits to fault strings (example)
                fault_map = {0: 'OVER_TEMP', 1: 'UNDER_VOLTAGE', 2: 'OVER_VOLTAGE', 3: 'BALANCER_FAIL'}
                codes = [fault_map[i] for i in range(32) if (mask >> i) & 1 and i in fault_map]
                self._last_fault_codes = codes
                return {'fault_mask': mask, 'fault_codes_raw': codes, 'timestamp': get_utc_timestamp()}
            else:
                # unknown id: return raw stub
                return {'_raw_can_id': can_id, 'raw_len': len(data), 'timestamp': get_utc_timestamp()}
        except Exception as e:
            logger.exception('Failed to parse CAN message %s: %s', hex(can_id), e)
            return {}

    def normalize_telemetry(self, raw_data: Dict[str, Any], power_bus_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Normalize parsed raw values into schema-compliant telemetry.

        Accepts dicts returned from parse_message and aggregates partial
        buffers into final arrays when possible. Converts units as needed.
        """
        try:
            out: Dict[str, Any] = {}
            # Update stored pack fields when present
            if 'pack_voltage_mv' in raw_data:
                self._last_pack_fields['pack_voltage'] = raw_data['pack_voltage_mv'] / 1000.0
                self._last_pack_fields['pack_current'] = raw_data['pack_current_ma'] / 1000.0
                self._last_pack_fields['soc'] = raw_data['soc_promille'] / 10.0
                self._last_pack_fields['soh'] = raw_data['soh_promille'] / 10.0

            # Use the latest complete snapshots when available
            if self._last_pack_fields:
                out.update(self._last_pack_fields)

            # Cells: prefer most recent complete snapshot
            if 'cell_voltages_complete' in raw_data and self._last_cell_voltages:
                out['cell_voltages'] = list(self._last_cell_voltages)
            elif self._last_cell_voltages:
                out['cell_voltages'] = list(self._last_cell_voltages)
            else:
                # If we don't yet have a complete snapshot, but have buffered partial values,
                # expose those so callers (e.g., tests or early consumers) can see available cells.
                if self._cell_voltage_buffer:
                    out['cell_voltages'] = list(self._cell_voltage_buffer)

            if 'cell_temps_complete' in raw_data and self._last_cell_temps:
                out['cell_temps'] = list(self._last_cell_temps)
            elif self._last_cell_temps:
                out['cell_temps'] = list(self._last_cell_temps)
            else:
                if self._cell_temp_buffer:
                    out['cell_temps'] = list(self._cell_temp_buffer)

            # Balancer and faults: use last known values if present in raw_data or stored
            if 'balancer_list' in raw_data:
                out['balancer_status'] = raw_data.get('balancer_list', list(self._last_balancer_list or []))
            elif self._last_balancer_list is not None:
                out['balancer_status'] = list(self._last_balancer_list)
            else:
                out['balancer_status'] = []

            if 'fault_codes_raw' in raw_data:
                out['fault_codes'] = raw_data.get('fault_codes_raw', list(self._last_fault_codes or []))
            elif self._last_fault_codes is not None:
                out['fault_codes'] = list(self._last_fault_codes)
            else:
                out['fault_codes'] = []

            # If we don't yet have pack fields, allow emitting cell-only or balancer-only
            # snapshots as long as there is meaningful data (cells, temps, balancer, faults).
            has_pack = any(k in out for k in ('pack_voltage', 'pack_current', 'soc', 'soh'))
            has_cells = any(k in out for k in ('cell_voltages', 'cell_temps'))
            has_misc = bool(out.get('balancer_status') or out.get('fault_codes'))
            # Merge power-bus ADC data (higher-fidelity) if provided
            if power_bus_data and isinstance(power_bus_data, dict):
                try:
                    # pack_voltage_v -> pack_voltage (override CAN-derived)
                    if 'pack_voltage_v' in power_bus_data:
                        out['pack_voltage'] = float(power_bus_data.get('pack_voltage_v'))
                    if 'pack_current_a' in power_bus_data:
                        out['pack_current'] = float(power_bus_data.get('pack_current_a'))
                    if 'max_temp_c' in power_bus_data:
                        out['pack_temp_c'] = float(power_bus_data.get('max_temp_c'))
                    if 'power_w' in power_bus_data:
                        out['power_w'] = float(power_bus_data.get('power_w'))
                except Exception:
                    logger.exception('Failed to merge power_bus_data into telemetry')

            if not (has_pack or has_cells or has_misc):
                # nothing meaningful to emit yet
                return {}

            # timestamp
            out['timestamp'] = raw_data.get('timestamp', get_utc_timestamp())
            return out
        except Exception as e:
            logger.exception('Failed to normalize telemetry: %s', e)
            return {}
