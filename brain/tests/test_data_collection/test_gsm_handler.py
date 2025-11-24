import json
from pathlib import Path

from data_collection.sensors.gsm_handler import GSMHandler


class DummyConfig:
    def __init__(self, d):
        self._d = d

    def get(self, key, default=None):
        parts = key.split('.')
        cur = self._d
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur


def test_gsm_simulation_outbox(tmp_path):
    out = str(tmp_path / 'gsm_outbox.txt')
    cfg = DummyConfig({'sensors': {'gsm': {'enable': True, 'enable_simulation': True, 'outbox_path': out}}})
    g = GSMHandler(cfg)

    assert g.get_name() == 'gsm'
    # initial health should be 'ok' after first successful send, but when created and enabled it should not be 'disabled'
    assert g.is_available() is True

    ok = g.send_sms('+15550001111', 'hello')
    assert ok is True
    # verify outbox file written
    lines = Path(out).read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry['phone'] == '+15550001111'
    assert 'hello' in entry['message']


def test_gsm_disabled_health():
    cfg = DummyConfig({'sensors': {'gsm': {'enable': False}}})
    g = GSMHandler(cfg)
    assert g.is_available() is False
    h = g.get_health()
    assert h['status'] == 'disabled' or h['status'] == 'unknown'
