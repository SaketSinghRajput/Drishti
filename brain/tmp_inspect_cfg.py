from pathlib import Path
p = Path('f:/iit/temp_test_cfg.yaml')
p.write_text('data_collection:\n  expected_sensors: [imu, gps, camera]\nhardware_diagnostics:\n  enable_simulation: false\n')
from config.config_loader import ConfigLoader
c = ConfigLoader(str(p))
val = c.get('data_collection.expected_sensors')
print('raw type:', type(val), 'repr:', repr(val))
