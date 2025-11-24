"""Example script demonstrating BMS simulation and handler usage.

Run this script to see simulated BMS telemetry output.
"""

import time
import json
from config.config_loader import ConfigLoader
from battery_monitor.ingestion.csv_simulator import BMSCSVSimulator
from data_collection.sensors.bms_handler import BMSHandler


def main():
    cfg = ConfigLoader()
    cfg.set('battery_monitor.ingestion.enable_simulation', True)
    cfg.set('battery_monitor.ingestion.csv_simulator_path', 'data/bms_simulation.csv')
    sim = BMSCSVSimulator(cfg)
    path = sim.generate_sample(duration_s=10, sample_rate_hz=1)
    print('Generated sample at', path)
    handler = BMSHandler(cfg)
    for i in range(10):
        s = handler.read()
        print(json.dumps(s, indent=2))
        time.sleep(1)


if __name__ == '__main__':
    main()
