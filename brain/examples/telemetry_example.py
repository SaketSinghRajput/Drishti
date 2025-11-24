from config.config_loader import ConfigLoader
from utils.telemetry import TelemetryCollector

config = ConfigLoader()
config.set('telemetry.output_dir', 'data/telemetry')
collector = TelemetryCollector(config)
collector.log_prediction(0.5, 'medium', {'speed': 45.0, 'latitude': 40.7, 'longitude': -74.0}, {'gps': True}, metadata={'vehicle_hash': 'abc123', 'trip_id': 'trip1'})
collector.log_outcome('pred1', 'near_miss', 5.0, 'moderate')
collector.log_false_alarm('pred2', 'driver_dismissed')
print('Buffered stats:', collector.get_stats())
collector.close()

# Inspect written files in configured telemetry output directory
import os, json
out = config.get('telemetry.output_dir')
for fn in os.listdir(out):
    if fn.startswith('telemetry_') and fn.endswith('.jsonl'):
        print('File:', fn)
        with open(os.path.join(out, fn), 'r', encoding='utf-8') as fh:
            for line in fh:
                print(json.loads(line))
