"""Demo for driver fatigue state classification and local alerts.

This example simulates a progression of drowsiness scores and demonstrates
how the `DriverFatigueHandler`'s state classification and the
`FatigueAlertManager` produce events and local alerts. It runs in
simulation mode (no camera required) and prints emitted events.
"""
from time import sleep, time
from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
from utils.fatigue_alert_manager import FatigueAlertManager


def run_demo():
    # minimal config enabling alerts (but in CI DummyGPIO will be used)
    cfg = {
        'sensors.driver_fatigue': {
            'state_classification': {
                'fatigue_low': 0.3,
                'drowsy': 0.6,
                'micro': 0.8,
                'state_hysteresis_margin': 0.05,
                'state_hysteresis_dwell_s': 1.0,
            },
            'alerts': {'enable_alerts': True, 'gpio_buzzer_pin': 18, 'gpio_led_pin': 23, 'enable_tts': False},
        }
    }

    handler = DriverFatigueHandler(cfg, use_cv=False)
    # use the underlying alert manager directly for demonstration
    mgr = handler._alert_mgr
    print('Starting simulated fatigue progression demo')

    scores = [0.1, 0.15, 0.2, 0.35, 0.45, 0.5, 0.65, 0.72, 0.9, 0.85, 0.5]
    for s in scores:
        state = handler._classify_state(s)
        prev = handler._current_state
        allowed = handler._should_transition(prev, state, s, time())
        print(f'Score={s:.2f} prev={prev} -> target={state} allowed={allowed}')
        if allowed and state != prev:
            handler._emit_state_event(prev, state, s, 'demo')
            if handler._alert_mgr:
                handler._alert_mgr.trigger_alert(state, s, 'demo')
            handler._current_state = state
            handler._last_state_change_ts = time()
        sleep(0.5)

    print('Events emitted:')
    evs = handler.get_events(clear=False)
    for e in evs:
        print(e)

    handler.close()


if __name__ == '__main__':
    run_demo()
