#!/usr/bin/env python3
"""CLI script to run hardware diagnostics with alerts and dashboard integration.

Usage:
    python scripts/run_hardware_check.py [--config CONFIG_PATH] [--no-alerts] [--no-dashboard]
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deployment.hardware_diagnostics import HardwareChecker
from config.config_loader import ConfigLoader
from utils.telemetry import TelemetryCollector


def main():
    parser = argparse.ArgumentParser(description='Run hardware diagnostics')
    parser.add_argument('--config', default='config/default_config.yaml', help='Config file path')
    parser.add_argument('--no-alerts', action='store_true', help='Disable GPIO alerts')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable dashboard POSTs')
    parser.add_argument('--no-telemetry', action='store_true', help='Disable telemetry logging')
    parser.add_argument('--simulation', action='store_true', help='Enable simulation mode')
    args = parser.parse_args()
    
    # Load config
    cfg = ConfigLoader(args.config)
    
    # Override config with CLI flags using public API
    if args.no_alerts:
        try:
            cfg.set('hardware_diagnostics.alerts.enable', False)
        except Exception:
            pass
    if args.no_dashboard:
        try:
            cfg.set('hardware_diagnostics.dashboard_integration.enable', False)
        except Exception:
            pass
    if args.simulation:
        try:
            cfg.set('hardware_diagnostics.enable_simulation', True)
        except Exception:
            pass
    
    # Create telemetry collector
    telemetry = None if args.no_telemetry else TelemetryCollector(cfg)
    
    # Create checker and run tests
    print("Starting hardware diagnostics...")
    checker = HardwareChecker(cfg, telemetry=telemetry)
    results = checker.run_all_tests()
    
    # Print summary
    print(f"\n=== Hardware Check Summary ===")
    print(f"Timestamp: {results.get('timestamp')}")
    print(f"Expected: {len(results.get('expected_sensors', []))} sensors")
    print(f"Passed: {len(results.get('passed_tests', []))}")
    print(f"Failed: {len(results.get('failed_tests', []))}")
    print(f"Skipped: {len(results.get('skipped_tests', []))}")
    
    if results.get('failed_tests'):
        print(f"\nFailed tests: {', '.join(results.get('failed_tests', []))}")
    if results.get('skipped_tests'):
        print(f"Skipped tests: {', '.join(results.get('skipped_tests', []))}")
    
    # Get health status
    health = checker.get_system_health_status()
    if health:
        print(f"\nOverall Status: {health['overall_status']}")
        print(f"Disabled Modules: {health['disabled_modules']}")
    
    # Cleanup
    if telemetry:
        telemetry.close()
    
    # Exit code: 0 if all passed, 1 if any failed
    sys.exit(0 if not results.get('failed_tests') else 1)


if __name__ == '__main__':
    main()
