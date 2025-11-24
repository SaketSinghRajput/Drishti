"""Hardware diagnostics package for EV Safety Guardian.

Provides a boot-time `HardwareChecker` used to detect connected hardware
(peripherals and sensors) before starting the main inference service.

Usage:

    from deployment.hardware_diagnostics import HardwareChecker, detect_hardware
    from config.config_loader import ConfigLoader

    cfg = ConfigLoader('config/default_config.yaml')
    checker = HardwareChecker(cfg)
    status = checker.detect_hardware()

This package includes phase 1 (detection), phase 2 (self-tests), and phase 3 (alerts/integration).

Example usage:

    from deployment.hardware_diagnostics import HardwareChecker
    from config.config_loader import ConfigLoader

    cfg = ConfigLoader('config/default_config.yaml')
    checker = HardwareChecker(cfg)

    # Run self-tests on all detected components
    status = checker.run_all_tests()
    print(status['passed_tests'])  # ['imu', 'gps', ...]
    print(status['failed_tests'])  # ['camera', ...]

    # Test a specific component
    result = checker.self_test('imu')
    if result['status'] == 'OK':
        print(f"IMU test passed: {result['details']}")
    else:
        print(f"IMU test failed: {result['reason']}")
"""

from .checker import HardwareChecker, detect_hardware
from .health_api import create_health_app

__all__ = ["HardwareChecker", "detect_hardware", "create_health_app"]
