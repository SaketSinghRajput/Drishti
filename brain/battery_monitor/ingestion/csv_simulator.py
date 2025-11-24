"""CSV simulator for BMS telemetry (scaffold).

Generates realistic BMS telemetry CSV files for testing and development.
Simulation-first approach allows development without hardware.

Fields align with `config/bms_schema.json`.
"""

import csv
import math
import random
from typing import Dict, Any, Optional, IO
from utils.time_utils import get_utc_timestamp


class BMSCSVSimulator:
    """Generate and stream sample CSV telemetry for a BMS.

    Emits rows compliant with `config/bms_schema.json`. Supports generating
    a file and streaming reads via `read_next_sample()`.
    """

    def __init__(self, config=None, out_path: str = 'data/bms_simulation.csv'):
        self.config = config or {}
        self.out_path = self.config.get('battery_monitor.ingestion.csv_simulator_path', out_path)
        self.sample_rate = int(self.config.get('battery_monitor.ingestion.sample_rate', 1))
        self.cells = int(self.config.get('battery_monitor.ingestion.cells', 13))
        self._fh: Optional[IO] = None

    def generate_sample(self, duration_s: int = 60, sample_rate_hz: int = None, out_path: Optional[str] = None) -> str:
        sample_rate_hz = sample_rate_hz or self.sample_rate
        out_path = out_path or self.out_path
        n_samples = max(1, int(duration_s * sample_rate_hz))
        n_cells = self.cells

        with open(out_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            header = ['timestamp', 'pack_voltage', 'pack_current', 'soc', 'soh', 'cell_voltages', 'cell_temps', 'balancer_status', 'fault_codes']
            writer.writerow(header)
            soc = 50.0
            soh = 99.0
            for i in range(n_samples):
                ts = get_utc_timestamp()
                pack_v = round(48.0 + math.sin(i / 10.0) * 1.5 + random.uniform(-0.02, 0.02), 3)
                current = round(random.uniform(-30, 30), 2)
                # integrate soc roughly by current (very approximate)
                soc = soc - current * 0.0001 + random.uniform(-0.1, 0.1)
                # clamp soc/soh to schema bounds
                soc = max(0.0, min(100.0, soc))
                soh = soh - random.uniform(0, 0.01)
                soh = max(0.0, min(100.0, soh))
                cell_vs = [round(pack_v / n_cells + random.uniform(-0.05, 0.05), 3) for _ in range(n_cells)]
                cell_ts = [round(25.0 + (abs(current) / 30.0) * 10.0 + random.uniform(-1, 1), 2) for _ in range(n_cells)]
                # clamp cell voltages and temps to schema bounds
                cell_vs = [max(2.5, min(4.5, v)) for v in cell_vs]
                cell_ts = [max(-20.0, min(80.0, t)) for t in cell_ts]
                bal = [1 if abs(cell_vs[i] - sum(cell_vs)/len(cell_vs)) > 0.05 else 0 for i in range(n_cells)]
                faults = ''
                # inject occasional fault
                if random.random() < 0.01:
                    faults = 'OVER_TEMP'
                writer.writerow([ts, pack_v, current, round(soc,2), round(soh,2), ';'.join(map(str, cell_vs)), ';'.join(map(str, cell_ts)), ';'.join(map(str, bal)), faults])

        return out_path

    def open(self, path: Optional[str] = None):
        path = path or self.out_path
        self._fh = open(path, 'r', encoding='utf-8')
        # consume header
        next(self._fh)

    def read_next_sample(self) -> Optional[Dict[str, Any]]:
        if self._fh is None:
            try:
                self.open()
            except Exception:
                return None
        line = self._fh.readline()
        if not line:
            return None
        parts = next(csv.reader([line.strip()]))
        if len(parts) < 9:
            return None
        ts, pack_v, current, soc, soh, cell_vs, cell_ts, bal, faults = parts
        return {
            'timestamp': ts,
            'pack_voltage': float(pack_v),
            'pack_current': float(current),
            'soc': float(soc),
            'soh': float(soh),
            'cell_voltages': [float(x) for x in cell_vs.split(';') if x],
            'cell_temps': [float(x) for x in cell_ts.split(';') if x],
            'balancer_status': [int(x) for x in bal.split(';') if x],
            'fault_codes': [faults] if faults else [],
        }

    def reset(self):
        if self._fh:
            self._fh.seek(0)
            next(self._fh)

