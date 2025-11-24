from utils.ble_utils import rssi_to_distance, smooth_rssi, filter_stale_neighbors, hash_node_id, compute_closing_speed


def test_rssi_to_distance_basic():
    # rssi -59 ~ 1m with default a=-59,b=2.0
    d = rssi_to_distance(-59)
    assert 0.5 <= d <= 2.0
    d2 = rssi_to_distance(-80)
    assert d2 > d


def test_smooth_rssi_and_filter():
    hist = [ -60.0, -62.0, -61.0 ]
    cur = -65.0
    sm = smooth_rssi(cur, hist, alpha=0.3)
    assert -100 < sm < 0

    neighbors = [{"last_seen_ms": 1000}, {"last_seen_ms": 6000}]
    filtered = filter_stale_neighbors(neighbors, max_age_ms=5000)
    assert len(filtered) == 1


def test_hashing_and_closing_speed():
    h1 = hash_node_id("nodeA", salt="s")
    h2 = hash_node_id("nodeA", salt="s")
    assert h1 == h2 and len(h1) == 8
    assert hash_node_id("nodeB") != h1

    n = {"rel_speed_mps": -3.5}
    cs = compute_closing_speed(n)
    assert cs == -3.5


def test_broadcast_safety_packet_writes_outbox(tmp_path, monkeypatch):
    # Ensure broadcast_safety_packet writes to a JSONL outbox in cwd
    monkeypatch.chdir(tmp_path)
    from utils.ble_utils import broadcast_safety_packet, hash_node_id

    vh = hash_node_id('vehicle-123')
    packet = {
        'fatigue': 10,
        'risk': 5,
        'battery_severity': 'OK',
        'max_cell_temp': 35.2,
        'ble_neighbors': 2,
        'lat': 12.34,
        'lon': 56.78,
        'speed': 3.2,
        'ts': 1234567890,
    }
    broadcast_safety_packet(vh, packet)
    outpath = tmp_path / 'data' / 'ble_safety_outbox.jsonl'
    assert outpath.exists()
    lines = outpath.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 1
    import json
    obj = json.loads(lines[0])
    assert 'packet' in obj
    assert obj['packet']['vehicle_hash'] == vh
