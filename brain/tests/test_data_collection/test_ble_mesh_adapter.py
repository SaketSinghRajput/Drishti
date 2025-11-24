from config.config_loader import ConfigLoader
from data_collection.ble_mesh_adapter import BLEMeshAdapter


def _make_cfg(path):
    path.write_text("sensors:\n  ble_mesh:\n    stale_threshold_ms: 5000\n    closing_speed_threshold_mps: 2.0\n    enable_aggregate_features: true\n")


def test_ble_mesh_adapter_empty(tmp_path):
    cfgf = tmp_path / "cfg.yaml"
    _make_cfg(cfgf)
    cfg = ConfigLoader(str(cfgf))
    ad = BLEMeshAdapter(cfg)
    assert ad.analyze([]) is None


def test_ble_mesh_adapter_single_sample(tmp_path):
    cfgf = tmp_path / "cfg.yaml"
    _make_cfg(cfgf)
    cfg = ConfigLoader(str(cfgf))
    ad = BLEMeshAdapter(cfg)

    sample = {
        "timestamp": "2020-01-01T00:00:00Z",
        "neighbors": [
            {"id_hash": "aaa11111", "rssi": -60, "dist_m": 1.0, "rel_speed_mps": -1.0, "last_seen_ms": 0},
            {"id_hash": "bbb22222", "rssi": -70, "dist_m": 5.0, "rel_speed_mps": 0.5, "last_seen_ms": 0},
            {"id_hash": "ccc33333", "rssi": -80, "dist_m": 10.0, "rel_speed_mps": 3.0, "last_seen_ms": 0},
        ],
    }

    feats = ad.analyze([sample])
    assert feats is not None
    assert feats["neighbor_count_mean"] == 3.0
    assert feats["nearest_distance_m"] == 1.0
    assert feats["farthest_distance_m"] == 10.0
    assert feats["unique_neighbors_count"] == 3


def test_ble_mesh_adapter_multiple_samples_and_churn(tmp_path):
    cfgf = tmp_path / "cfg.yaml"
    _make_cfg(cfgf)
    cfg = ConfigLoader(str(cfgf))
    ad = BLEMeshAdapter(cfg)

    s1 = {"timestamp": "t1", "neighbors": [{"id_hash": "a", "rssi": -60, "dist_m": 1.0, "rel_speed_mps": -3.0, "last_seen_ms": 0}, {"id_hash": "b", "rssi": -70, "dist_m": 5.0, "rel_speed_mps": 1.0, "last_seen_ms": 0}]}
    s2 = {"timestamp": "t2", "neighbors": [{"id_hash": "a", "rssi": -61, "dist_m": 1.1, "rel_speed_mps": -2.5, "last_seen_ms": 0}, {"id_hash": "c", "rssi": -65, "dist_m": 3.0, "rel_speed_mps": -1.0, "last_seen_ms": 0}]}
    s3 = {"timestamp": "t3", "neighbors": [{"id_hash": "a", "rssi": -63, "dist_m": 1.2, "rel_speed_mps": -4.0, "last_seen_ms": 0}]}

    feats = ad.analyze([s1, s2, s3])
    assert feats is not None
    # neighbor_count mean should reflect counts [2,2,1]
    assert abs(feats["neighbor_count_mean"] - (2.0 + 2.0 + 1.0) / 3.0) < 1e-6
    # approaching neighbors: entries with rel_speed_mps < -2.0 (threshold 2.0)
    assert feats["approaching_neighbors_count"] >= 1
    assert feats["unique_neighbors_count"] == 3


def test_ble_mesh_adapter_disable_flag(tmp_path):
    cfgf = tmp_path / "cfg.yaml"
    cfgf.write_text("sensors:\n  ble_mesh:\n    enable_aggregate_features: false\n")
    cfg = ConfigLoader(str(cfgf))
    ad = BLEMeshAdapter(cfg)
    s = {"timestamp": "t", "neighbors": [{"id_hash": "a", "rssi": -60, "dist_m": 1.0, "rel_speed_mps": -1.0, "last_seen_ms": 0}]}
    assert ad.analyze([s]) is None


def test_ble_mesh_adapter_stale_filtering(tmp_path):
    cfgf = tmp_path / "cfg.yaml"
    _make_cfg(cfgf)
    cfg = ConfigLoader(str(cfgf))
    ad = BLEMeshAdapter(cfg)

    # neighbors with large last_seen_ms should be filtered
    s = {"timestamp": "t", "neighbors": [{"id_hash": "a", "rssi": -60, "dist_m": 1.0, "rel_speed_mps": -1.0, "last_seen_ms": 10000}, {"id_hash": "b", "rssi": -60, "dist_m": 2.0, "rel_speed_mps": -1.0, "last_seen_ms": 0}]}
    feats = ad.analyze([s])
    # only neighbor b should remain, so unique count 1
    assert feats is not None
    assert feats["unique_neighbors_count"] == 1
