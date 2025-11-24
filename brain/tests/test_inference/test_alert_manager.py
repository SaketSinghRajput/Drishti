import time
from config.config_loader import ConfigLoader
from inference.alert_manager import AlertManager


def test_alert_manager_stage_transitions():
    cfg = ConfigLoader('config/default_config.yaml')
    am = AlertManager(cfg)

    # stage1 immediate for low
    dec = {"alert_level": "low", "alert_active": True, "reason": "test"}
    out = am.process_decision(dec)
    assert out["stage"] in ("stage1", "none")

    # simulate persistence for stage2 by manipulating internal state
    am._last_level = "high"
    am._stage_entry_ts = time.time() - 10
    out2 = am.process_decision({"alert_level": "high", "alert_active": True, "reason": "test"})
    assert out2["stage"] == "stage2"


def test_alert_manager_triggers_gsm_once_on_stage2_transition():
    # Dummy config enabling GSM and making stage2 immediate
    class DummyConfig:
        def get(self, key, default=None):
            mapping = {
                'inference.alert_stage1_duration_s': 0,
                'inference.alert_stage2_duration_s': 0,
                'inference.alert_stage2_min_level': 'medium',
                'inference.alerts.enable_gsm': True,
                'inference.alerts.gsm_phone_number': '+15559990000',
            }
            return mapping.get(key, default)

    class MockGSM:
        def __init__(self):
            self.calls = []

        def send_sms(self, to, msg):
            self.calls.append((to, msg))
            return True

    cfg = DummyConfig()
    mock = MockGSM()
    am = AlertManager(cfg, gsm_handler=mock)

    # First decision to set baseline
    am.process_decision({'alert_level': 'low', 'alert_active': True, 'reason': 'baseline'})

    # Now escalate to high -> should trigger stage2 immediately and send SMS once
    out = am.process_decision({'alert_level': 'high', 'alert_active': True, 'reason': 'escalate', 'sensors': {'gps_lat': 1.2, 'gps_lon': 3.4}})
    assert out['stage'] == 'stage2'
    assert len(mock.calls) == 1
    # Ensure not sent again for repeated calls at stage2
    out2 = am.process_decision({'alert_level': 'high', 'alert_active': True, 'reason': 'still high'})
    assert len(mock.calls) == 1
