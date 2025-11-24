"""BMS CAN message parser (stub).

Parses CAN messages from LibreSolar bms-c1 BMS.

Reference: https://github.com/LibreSolar/bms-c1

Responsibilities (Phase 2+):
- Parse raw CAN frames (CAN ID, data bytes) and map to telemetry fields.
- Normalize units (V, A, %, °C) and output JSON objects matching
  `config/bms_schema.json`.
- Provide hooks for live ingestion (PiCAN2 / USB-CAN) and offline parsing.

This file contains a lightweight scaffold to guide future implementation.
"""

from typing import Dict, Any


class BMSCANParser:
    """Parser for LibreSolar bms-c1 CAN messages.

    TODOs:
    - Implement CAN ID mapping (e.g., 0x100 pack data, 0x200 cell voltages)
    - Handle multi-frame cell arrays and temperature sensors
    - Provide checksum/validation when applicable
    """

    def __init__(self, config=None):
        self.config = config or {}

    def parse_message(self, can_id: int, data: bytes) -> Dict[str, Any]:
        """Parse a single CAN message and return a partial telemetry dict.

        Args:
            can_id: Integer CAN ID.
            data: Raw bytes payload from the CAN frame.

        Returns:
            A dict with parsed fields (may be partial); keys align with
            `config/bms_schema.json` after normalization.
        """
        # Placeholder implementation
        # TODO: implement mapping per LibreSolar bms-c1 protocol
        return {"_raw_can_id": can_id, "_raw_data_len": len(data)}

    def normalize_telemetry(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed raw values to normalized telemetry.

        Performs unit conversions and arranges arrays for `cell_voltages` and
        `cell_temps` when appropriate.
        """
        # Placeholder: echo through for now
        normalized = dict(raw_data)
        return normalized
