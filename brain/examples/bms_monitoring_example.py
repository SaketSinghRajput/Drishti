"""Example demonstrating BMS monitoring (rules + ML) using BMSHandler and BMSMonitor.

This script shows how to collect a 60s window of BMS samples and run the
monitor to produce alerts which are logged via TelemetryCollector.
"""
from __future__ import annotations

import time
from config.config_loader import ConfigLoader
from data_collection.sensors.bms_handler import BMSHandler
from utils.telemetry import TelemetryCollector
from battery_monitor.rules.bms_monitor import BMSMonitor
from battery_monitor.ingestion.csv_simulator import BMSCSVSimulator


def run_example():
    cfg = ConfigLoader()
    tc = TelemetryCollector(cfg)

    # prepare simulator CSV
    sim = BMSCSVSimulator(cfg, out_path="/tmp/bms_sim.csv")
    sim.generate_sample(duration_s=65, sample_rate_hz=1)
    sim.open()

    handler = BMSHandler(cfg)
    monitor = BMSMonitor(cfg, tc)

    samples = []
    for i in range(65):
        s = handler.read()
        if s:
            samples.append(s)
        time.sleep(1)
        if len(samples) >= 10 and i % 10 == 0:
            alerts = monitor.process_telemetry(samples[-60:])
            if alerts:
                print("Alerts:", alerts)

    print("Telemetry stats:", tc.get_stats())


if __name__ == "__main__":
    run_example()
