from battery_monitor.rules.rule_engine import RuleEngine

class DummyCfg:
    def get(self, key, default=None):
        mapping = {
            'battery_monitor.rules.thresholds.pack_voltage_min_v': 3.0,
            'battery_monitor.rules.thresholds.pack_voltage_max_v': 5.0,
            'battery_monitor.rules.thresholds.pack_current_spike_a': 100.0,
            'battery_monitor.rules.thresholds.pack_temp_max_c': 70.0,
            'battery_monitor.rules.thresholds.voltage_drop_threshold_v': 0.5,
        }
        return mapping.get(key, default)


def test_pack_voltage_low_alert():
    re = RuleEngine(DummyCfg())
    data = {'pack_voltage': 2.5}
    alerts = re.evaluate(data)
    assert any(a['rule_name'] == 'pack_voltage_out_of_range' and a['triggered'] for a in alerts)

def test_current_spike_alert():
    re = RuleEngine(DummyCfg())
    data = {'pack_current': 120}
    alerts = re.evaluate(data)
    assert any(a['rule_name'] == 'current_spike' and a['triggered'] for a in alerts)

def test_pack_temp_high_alert():
    re = RuleEngine(DummyCfg())
    data = {'pack_temp_c': 80}
    alerts = re.evaluate(data)
    assert any(a['rule_name'] == 'pack_temp_high' and a['triggered'] for a in alerts)

def test_voltage_drop_alert():
    re = RuleEngine(DummyCfg())
    # feed multiple calls to trigger voltage drop detection
    samples = [ {'pack_voltage': 4.2}, {'pack_voltage': 3.8}, {'pack_voltage': 3.5} ]
    alerts = []
    for s in samples:
        alerts = re.evaluate(s)
    assert any(a['rule_name'] == 'voltage_drop_detected' and a['triggered'] for a in alerts)


def test_combined_violations_and_severity():
    re = RuleEngine(DummyCfg())
    data = {'pack_voltage': 2.0, 'pack_current': 200}
    alerts = re.evaluate(data)
    names = {a['rule_name']: a for a in alerts}
    # Both pack voltage and current spike should trigger
    assert 'pack_voltage_out_of_range' in names and names['pack_voltage_out_of_range']['triggered']
    assert 'current_spike' in names and names['current_spike']['triggered']
    # Severity labels should be non-empty strings and conform to expected mapping (LOW/MEDIUM/HIGH/CRITICAL)
    valid = set(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
    for k in ('pack_voltage_out_of_range', 'current_spike'):
        sev = names[k]['severity']
        assert isinstance(sev, str) and sev.upper() in valid
