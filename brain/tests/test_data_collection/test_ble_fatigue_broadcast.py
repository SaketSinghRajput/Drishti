import json
import threading
from pathlib import Path

import pytest

from data_collection.sensors.ble_fatigue_broadcast import BLEFatigueBroadcast
from utils import ble_utils


def test_broadcast_drowsy_state(tmp_path, monkeypatch):
    cfg = {"sensors": {"driver_fatigue": {"ble_broadcast": {"enable": True, "outbox_path": str(tmp_path / "outbox.json")}}}, "privacy": {"node_id_salt": "salt"}}
    b = BLEFatigueBroadcast(cfg)
    b.broadcast("DROWSY", 0.7, "EAR below threshold", "vehicle123")
    out = Path(b.outbox_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(out) == 1
    pkt = json.loads(out[0])
    assert pkt["state"] == "DROWSY"
    assert pkt["status"] == "FatiguedDriver"
    assert "vehicle_id" in pkt


def test_broadcast_normal_state_skipped(tmp_path):
    cfg = {"sensors": {"driver_fatigue": {"ble_broadcast": {"enable": True, "outbox_path": str(tmp_path / "outbox.json")}}}, "privacy": {"node_id_salt": "salt"}}
    b = BLEFatigueBroadcast(cfg)
    # should not write
    b.broadcast("NORMAL", 0.0, "", "vehicle123")
    assert not Path(b.outbox_path).exists()


def test_thread_safety(tmp_path):
    cfg = {"sensors": {"driver_fatigue": {"ble_broadcast": {"enable": True, "outbox_path": str(tmp_path / "outbox.json")}}}, "privacy": {"node_id_salt": "salt"}}
    b = BLEFatigueBroadcast(cfg)
    def worker(i):
        b.broadcast("DROWSY", 0.6 + i * 0.01, f"reason-{i}", f"v{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = Path(b.outbox_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10


def test_adv_impl_called_on_maybe_broadcast(tmp_path, monkeypatch):
    # ensure that when an advertiser implementation is present it's invoked
    cfg = {"sensors": {"driver_fatigue": {"ble_broadcast": {"enable": True, "outbox_path": str(tmp_path / "outbox.json"), "broadcast_interval_s": 1}}}, "privacy": {"node_id_salt": "salt"}}
    b = BLEFatigueBroadcast(cfg)
    calls = []

    def fake_adv(payload, **kw):
        calls.append((payload, kw))

    b._adv_impl = fake_adv
    b.maybe_broadcast("DROWSY", 0.8, "reason", "veh1")
    assert len(calls) == 1
    # payload should be bytes containing a JSON snippet
    assert isinstance(calls[0][0], (bytes, bytearray))
    # check tx_power passed through kwargs
    assert "tx_power" in calls[0][1]


def test_mesh_handler_delegation(tmp_path):
    cfg = {"sensors": {"driver_fatigue": {"ble_broadcast": {"enable": True, "outbox_path": str(tmp_path / "outbox.json"), "broadcast_interval_s": 1}}}, "privacy": {"node_id_salt": "salt"}}
    calls = []

    class FakeMesh:
        def broadcast_fatigue(self, payload: bytes):
            calls.append(payload)

    b = BLEFatigueBroadcast(cfg, mesh_handler=FakeMesh())
    b.maybe_broadcast("DROWSY", 0.9, "reason", "vehicleX")
    assert len(calls) == 1
    assert isinstance(calls[0], (bytes, bytearray))


def test_hci_subprocess_called_on_linux(tmp_path, monkeypatch):
    # Simulate Linux platform and capture subprocess calls
    cfg = {
        "sensors": {
            "driver_fatigue": {"ble_broadcast": {"enable": True, "outbox_path": str(tmp_path / "outbox.json"), "broadcast_interval_s": 1}}
        },
        "privacy": {"node_id_salt": "salt"},
    }
    calls = []

    def fake_run(cmd, check=False, capture_output=False):
        # record the command invoked
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = b""
            stderr = b""
        return R()

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("subprocess.run", fake_run)

    from data_collection.sensors.ble_fatigue_broadcast import BLEFatigueBroadcast

    b = BLEFatigueBroadcast(cfg)
    # ensure no mesh/impl set so Linux path is used
    b._adv_impl = None
    b._mesh_handler = None
    b.maybe_broadcast("DROWSY", 0.85, "reason", "vehX")

    # Expect at least one subprocess.run call for txpower or hcitool
    assert len(calls) >= 1
    joined = " ".join([str(c) for c in calls])
    assert "hcitool" in joined or "hciconfig" in joined
