"""Voltage/current sensor example: reads simulation CSV and demonstrates merging with BMS parser and rule engine."""
from config.config_loader import ConfigLoader
from data_collection.sensors.voltage_current_handler import VoltageCurrentHandler
from battery_monitor.ingestion.bms_parser import BMSCANParser
from battery_monitor.rules.rule_engine import RuleEngine
import time

cfg = ConfigLoader('config/default_config.yaml')
# override minimal config for example
class CfgWrap:
    def __init__(self, base):
        self.base = base
    def get(self, key, default=None):
        if key == 'sensors.voltage_current.enable':
            return True
        if key == 'sensors.voltage_current.enable_simulation':
            return True
        if key == 'sensors.voltage_current.csv_simulator_path':
            return 'data/voltage_current_simulation.csv'
        return self.base.get(key, default)

cfgw = CfgWrap(cfg)
handler = VoltageCurrentHandler(cfgw)
print('Handler available:', handler.is_available())
for i in range(5):
    s = handler.read()
    print('Sample', i, s)
    time.sleep(0.2)

# Demonstrate merge with BMSCANParser
parser = BMSCANParser(cfg)
raw_can = {'pack_voltage_mv': 4200, 'pack_current_ma': 2500, 'soc_promille': 800, 'soh_promille': 950}
power = handler.read()
merged = parser.normalize_telemetry(raw_can, power_bus_data=power)
print('Merged telemetry:', merged)

# Rule engine demo
re = RuleEngine(cfg)
alerts = re.evaluate(merged)
print('Alerts:', alerts)
