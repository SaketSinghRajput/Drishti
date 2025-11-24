import re
import time

from config.config_loader import ConfigLoader
from data_collection.sensors.ble_mesh_handler import BLEMeshHandler


def _make_cfg(path):
    path.write_text("sensors:\n  ble_mesh:\n    scan_interval_ms: 500\n    rssi_threshold: -80\n    max_neighbors: 10\n    smoothing_alpha: 0.3\n")


def test_ble_mesh_handler_basic_read(tmp_path):
    cfgf = tmp_path / "cfg.yaml"
    _make_cfg(cfgf)
    cfg = ConfigLoader(str(cfgf))
    h = BLEMeshHandler(cfg)
    # call read several times
    samples = []
    for _ in range(5):
        s = h.read()
        assert isinstance(s, dict)
        assert "timestamp" in s
        assert isinstance(s.get("neighbors"), list)
        samples.append(s)
        time.sleep(0.01)

    # inspect a neighbor if present
    for s in samples:
        for n in s.get("neighbors", []):
            assert re.fullmatch(r"[0-9a-f]{8}", n.get("id_hash"))
            assert -120 <= n.get("rssi", -200) <= -10
            assert n.get("dist_m", 0) >= 0
            assert 0 <= n.get("rel_heading", 0) <= 360


def test_ble_mesh_handler_health_and_limits(tmp_path):
    cfgf = tmp_path / "cfg.yaml"
    cfgf.write_text("sensors:\n  ble_mesh:\n    scan_interval_ms: 500\n    rssi_threshold: -120\n    max_neighbors: 2\n")
    cfg = ConfigLoader(str(cfgf))
    h = BLEMeshHandler(cfg)
    # produce multiple reads
    for _ in range(10):
        s = h.read()
    health = h.get_health()
    assert "neighbor_count" in health
    assert health["neighbor_count"] <= 2
