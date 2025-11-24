import json
import os
from tempfile import TemporaryDirectory

from battery_monitor.alerts.ble_stub import BLEBroadcastStub


def test_ble_stub_writes_json_and_hashes_vehicle():
    cfg = {"battery_monitor": {"alerts": {"ble_outbox_path": ""}}, "privacy": {"node_id_salt": "s"}}
    with TemporaryDirectory() as td:
        out = os.path.join(td, "out.json")
        cfg["battery_monitor"]["alerts"]["ble_outbox_path"] = out
        stub = BLEBroadcastStub(cfg)
        alert = {"severity": "CRITICAL", "reason": "overtemp", "anomaly_score": 0.95}
        stub.broadcast(alert, vehicle_id="veh123")
        with open(out, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        assert len(lines) == 1
        pkt = json.loads(lines[0])
        assert pkt["severity"] == "CRITICAL"
        assert pkt["reason"] == "overtemp"
        assert "vehicle_id" in pkt and pkt["vehicle_id"] != "veh123"
