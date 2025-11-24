"""Example: send a simulated SMS via GSMHandler and trigger via AlertManager.

This example demonstrates using the GSMHandler in simulation mode (writes to outbox file)
and shows AlertManager triggering an SMS on a CRITICAL-level escalation.

Run:
    python brain/examples/gsm_sms_example.py

"""
from config.config_loader import ConfigLoader
from data_collection.sensors.gsm_handler import GSMHandler
from inference.alert_manager import AlertManager
import os

# Load config (defaults)
cfg = ConfigLoader('config/default_config.yaml')
# Create a small override so we can enable simulation without editing the global config
override = {
    'sensors': {'gsm': {'enable': True, 'enable_simulation': True, 'outbox_path': 'data/gsm_outbox_example.txt'}}
}

class CombinedConfig:
    def __init__(self, base_cfg, overrides):
        self.base = base_cfg
        self.overrides = overrides

    def get(self, key, default=None):
        # support simple dot-notation for overrides
        parts = key.split('.')
        cur = self.overrides
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                cur = None
                break
        if cur is not None:
            return cur
        try:
            return self.base.get(key, default)
        except Exception:
            return default

combined = CombinedConfig(cfg, override)

# Ensure outbox dir exists
outbox = combined.get('sensors.gsm.outbox_path')
os.makedirs(os.path.dirname(outbox) or '.', exist_ok=True)

gsm = GSMHandler(combined)
print('Sending simulated SMS to outbox:', outbox)
ok = gsm.send_sms('+15551234567', 'Test CRITICAL alert from example')
print('send_sms returned', ok)
print('Outbox contents:')
with open(outbox, 'r', encoding='utf-8') as fh:
    for line in fh:
        print(line.strip())

# Demonstrate AlertManager triggering an SMS
am = AlertManager(combined, gsm_handler=gsm)
decision = {'alert_level': 'critical', 'alert_active': True, 'reason': 'Example critical test', 'sensors': {'gps_lat': 12.34, 'gps_lon': 56.78}}
out = am.process_decision(decision)
print('AlertManager output:', out)
print('Example complete')
