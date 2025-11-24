"""CAN bus ingestion and BMS data parsing subpackage.

Reads telemetry from LibreSolar bms-c1 via PiCAN2/USB-CAN or simulated CSV
files. Exports parser and simulator classes (stubs for Phase 1).
"""

from .can_parser import BMSCANParser
from .csv_simulator import BMSCSVSimulator

__all__ = ["BMSCANParser", "BMSCSVSimulator"]
