"""Example: run dashboard with simulated telemetry.

This example demonstrates creating a dashboard app and pushing telemetry updates.
"""
import time
from battery_monitor.dashboard import create_app


def main():
    app = create_app(config={'battery_monitor': {'dashboard': {'max_history_samples': 120}}}, event_logger=None)
    # run the app in a separate thread if embedding; for demo, run directly
    print('Starting dashboard on http://0.0.0.0:5002')
    app.run(host='0.0.0.0', port=5002)


if __name__ == '__main__':
    main()
