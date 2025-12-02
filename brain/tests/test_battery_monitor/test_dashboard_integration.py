import os
import tempfile
import shutil

from battery_monitor.dashboard.app import create_app
from battery_monitor.storage.event_logger import EventLogger
from data_collection.sensors.bms_handler import BMSHandler
from battery_monitor.rules.bms_monitor import BMSMonitor
from utils.telemetry import TelemetryCollector
import csv
from collections import deque
import time


def test_dashboard_alerts_and_snapshot_roundtrip():
    # Full pipeline integration: CSV -> BMSHandler -> BMSMonitor -> EventLogger -> Dashboard
    td = tempfile.mkdtemp(prefix="bms_test_")
    try:
        db = os.path.normpath(os.path.join(td, 'events.db'))
        csv_path = os.path.normpath(os.path.join(td, 'sim.csv'))
        alerts_log = os.path.normpath(os.path.join(td, 'alerts.log'))
        ble_out = os.path.normpath(os.path.join(td, 'ble.json'))

        # test config (simple nested mapping for SimpleConfig-like access)
        cfg = {
            'battery_monitor': {
                'ingestion': {'enable_simulation': True, 'csv_simulator_path': csv_path, 'sample_rate': 1, 'cells': 6},
                'storage': {'db_path': db},
                'alerts': {'log_path': alerts_log, 'ble_outbox_path': ble_out},
                'dashboard': {}
            }
        }

        class SimpleConfig:
            def __init__(self, data):
                self._data = data

            def get(self, key, default=None):
                parts = key.split('.')
                cur = self._data
                for p in parts:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        return default
                return cur

        sconf = SimpleConfig(cfg)

        # Create a CSV with mostly-normal samples and one high-temp fault row
        n_samples = 30
        cells = cfg['battery_monitor']['ingestion']['cells']
        header = ['timestamp', 'pack_voltage', 'pack_current', 'soc', 'soh', 'cell_voltages', 'cell_temps', 'balancer_status', 'fault_codes']
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for i in range(n_samples):
                ts = f'2025-11-21T10:00:{i:02d}Z'
                pack_v = 48.5
                cur = 0.0
                soc = 80.0
                soh = 98.0
                cell_vs = ';'.join(str(round(pack_v / cells, 3)) for _ in range(cells))
                # inject a high temp on sample 10
                if i == 10:
                    cell_ts = ';'.join('75.0' for _ in range(cells))
                    faults = 'OVER_TEMP'
                else:
                    cell_ts = ';'.join('30.0' for _ in range(cells))
                    faults = ''
                bal = ';'.join('0' for _ in range(cells))
                writer.writerow([ts, pack_v, cur, soc, soh, cell_vs, cell_ts, bal, faults])

        # Initialize handler and monitor
        handler = BMSHandler(sconf)
        # ensure simulator reads our CSV
        if getattr(handler, '_sim', None):
            try:
                handler._sim.open(csv_path)
            except Exception:
                pass

        tel = TelemetryCollector(sconf)
        monitor = BMSMonitor(sconf, tel)

        # process samples quickly: sliding window of up to 60 samples
        window = deque(maxlen=60)
        while True:
            s = handler.read()
            if s is None:
                break
            window.append(s)
            # run monitor occasionally
            if len(window) >= 1:
                monitor.process_telemetry(list(window))

        # create dashboard bound to monitor's EventLogger (same DB)
        app = create_app(config=cfg, event_logger=monitor.event_logger)
        client = app.test_client()

        # alerts should be present
        r = client.get('/api/alerts')
        assert r.status_code == 200
        data = r.get_json()
        assert len(data['alerts']) >= 1
        # at least one should be HIGH/CRITICAL
        severities = {a.get('severity') for a in data['alerts']}
        assert any(s in ('HIGH', 'CRITICAL') for s in severities)

        # history returns flattened samples linked to alerts
        r2 = client.get('/api/telemetry/history')
        assert r2.status_code == 200
        hist = r2.get_json()
        assert 'history' in hist and isinstance(hist['history'], list)
        # ensure at least one history entry references an alert id
        recent_alerts = monitor.event_logger.get_recent_alerts(limit=50)
        alert_ids = [a['id'] for a in recent_alerts]
        assert any(entry.get('alert_id') in alert_ids for entry in hist['history'])

        # post a final sample to /api/telemetry/update and verify computed_metrics
        last_sample = window[-1] if window else {'timestamp': '2025-11-21T10:00:00Z', 'pack_voltage': 48.5}
        pr = client.post('/api/telemetry/update', json=last_sample)
        assert pr.status_code == 200
        r3 = client.get('/api/telemetry/latest')
        assert r3.status_code == 200
        latest = r3.get_json()
        # computed_metrics should contain voltage_diff or be present
        assert 'computed_metrics' in latest

    finally:
        # Windows-safe cleanup: stop components, close handles, wait, then remove dir
        try:
            monitor.stop()
        except Exception:
            pass
        try:
            handler.stop()
        except Exception:
            pass

        # Ensure EventLogger is closed if present on monitor
        try:
            if hasattr(monitor, 'event_logger') and monitor.event_logger:
                try:
                    monitor.event_logger.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Close telemetry collector if it exposes close()
        try:
            if 'tel' in locals() and hasattr(tel, 'close'):
                try:
                    tel.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Small delay to allow Windows to release file locks
        try:
            time.sleep(0.1)
        except Exception:
            pass

        # Remove temporary directory (ignore errors to avoid test failure due to OS locks)
        try:
            shutil.rmtree(td, ignore_errors=True)
        except Exception:
            pass
