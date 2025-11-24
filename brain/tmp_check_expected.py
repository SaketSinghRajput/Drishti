from pathlib import Path
from deployment.hardware_diagnostics.checker import HardwareChecker
from config.config_loader import ConfigLoader
p = Path('f:/iit/temp_test_cfg.yaml')
# ensure the same temp_test_cfg.yaml exists
p.write_text('data_collection:\n  expected_sensors: [imu, gps, camera]\nhardware_diagnostics:\n  enable_simulation: false\n  log_dir: "./tmp_logs"\n')
cfg = ConfigLoader(str(p))
chk = HardwareChecker(cfg)
print('expected_sensors type:', type(chk.expected_sensors), 'value:', chk.expected_sensors)
print('log_dir:', chk.log_dir)
