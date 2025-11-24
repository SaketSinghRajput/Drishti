#!/usr/bin/env python3
"""Example: Hardware diagnostics with self-tests.

Demonstrates detection and self-testing of all hardware components.
"""
# Note: In production, hardware checks run automatically via systemd:
#   - Boot-time: hardware-diagnostics.service
#   - Periodic: hardware-diagnostics.timer (every 5min)
# This script is for manual testing/development. See `docs/deployment.md` for systemd setup.
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deployment.hardware_diagnostics import HardwareChecker
from config.config_loader import ConfigLoader


def main():
    print("=" * 60)
    print("Hardware Diagnostics Example")
    print("=" * 60)

    # Load config
    cfg = ConfigLoader('config/default_config.yaml')
    checker = HardwareChecker(cfg)

    # Phase 1: Detection
    print("\n[Phase 1] Detecting hardware...")
    detection = checker.detect_hardware()
    print(f"Found: {detection['found_sensors']}")
    print(f"Missing: {detection['missing_sensors']}")

    # Phase 2: Self-tests
    print("\n[Phase 2] Running self-tests...")
    results = checker.run_all_tests()

    print(f"\nPassed tests ({len(results['passed_tests'])}):")
    for comp in results['passed_tests']:
        details = results['test_results'][comp].get('details', '')
        print(f"  ✓ {comp}: {details}")

    print(f"\nFailed tests ({len(results['failed_tests'])}):")
    for comp in results['failed_tests']:
        reason = results['test_results'][comp].get('reason', 'unknown')
        print(f"  ✗ {comp}: {reason}")

    # Test a specific component
    print("\n[Example] Testing IMU specifically...")
    imu_result = checker.self_test('imu')
    print(f"IMU status: {imu_result['status']}")
    if imu_result['status'] == 'OK':
        print(f"Details: {imu_result.get('details', 'N/A')}")
    else:
        print(f"Reason: {imu_result.get('reason', 'N/A')}")

    # Save results to JSON
    output_path = project_root / 'data' / 'hardware_test_results.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    # Phase 3: Alerts, Status, Dashboard Integration
    print("\n=== Phase 3: Alerts and Integration ===")

    # 1. Get system health status
    health = checker.get_system_health_status()
    if health:
        print(f"Overall Status: {health['overall_status']}")
        print(f"Disabled Modules: {health['disabled_modules']}")
        for comp, status in health['components'].items():
            print(f"  {comp}: {status}")
    else:
        print("No health check run yet. Running now...")
        checker.run_all_tests()
        health = checker.get_system_health_status()
        print(f"Overall Status: {health['overall_status']}")

    # 2. Check if specific module is enabled
    if checker.is_module_enabled('camera'):
        print("Camera is enabled and functional")
    else:
        print("Camera is disabled (failed or missing)")

    # 3. Demonstrate telemetry integration (optional)
    try:
        from utils.telemetry import TelemetryCollector
        telemetry = TelemetryCollector(cfg)
        checker_with_telemetry = HardwareChecker(cfg, telemetry=telemetry)
        results = checker_with_telemetry.run_all_tests()
        print(f"Telemetry stats: {telemetry.get_stats()}")
        telemetry.close()
    except Exception:
        pass

    # 4. Optional: Start health API (if enabled in config)
    if cfg.get('hardware_diagnostics.health_api.enable', False):
        from deployment.hardware_diagnostics.health_api import create_health_app
        app = create_health_app(checker)
        print("Starting health API on http://localhost:5002/health")
        # app.run(host='0.0.0.0', port=5002)  # Uncomment to run

    # Exit code based on results
    if results['failed_tests']:
        print("\n⚠️  Some tests failed. Check logs for details.")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
