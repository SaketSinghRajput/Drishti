# Driver Fatigue State Classification and Local Alerts

This document describes the new local state classification and alerting
features added for driver fatigue detection.

## Overview

- Weighted drowsiness score combining eye closure (EAR), mouth openness (MAR), and head tilt.
- Four discrete states: `NORMAL`, `TIRED`, `DROWSY`, `MICROSLEEP`.
- Hysteresis to prevent rapid flapping: margin + dwell time.
- Local actuation: buzzer, LED, and TTS voice alerts on `DROWSY` and `MICROSLEEP`.
- Events emitted on state transitions and consumed by the `DriverFatigueAdapter`.

## Configuration

The YAML config under `sensors.driver_fatigue` contains a `state_classification`
subsection and `alerts` subsection. Example keys:

```yaml
sensors:
  driver_fatigue:
    state_classification:
      enable: true
      fatigue_low: 0.3
      drowsy: 0.6
      micro: 0.8
      state_hysteresis_margin: 0.05
      state_hysteresis_dwell_s: 2.0
    alerts:
      enable_alerts: true
      gpio_buzzer_pin: 18
      gpio_led_pin: 23
      enable_tts: true
      tts_engine: 'pyttsx3'
```

## Weighted Score

The handler computes a weighted score using the normalized metrics:

```
drowsiness = 0.5 * eye_fatigue + 0.3 * norm_mar + 0.2 * head_tilt_norm
```

EAR is normalized as an open-eye metric `ear_norm` in [0..1] (higher = more open).
The handler converts this to an eye-closure fatigue component `eye_fatigue = 1 - ear_norm`
so that higher values indicate greater eye closure. `norm_mar` and `head_tilt_norm`
remain normalized in [0..1].

## State Machine & Hysteresis

- States: `NORMAL` (< fatigue_low), `TIRED` (fatigue_low..drowsy),
  `DROWSY` (drowsy..micro), `MICROSLEEP` (> micro)
- Transitions require both a margin (e.g. 0.05) around thresholds and a
  dwell time (e.g. 2s) to prevent flapping.
- On allowed transitions, handler emits an event and triggers the AlertManager.

## Alerts

`FatigueAlertManager` implements buzzer/LED/TTS alerts with graceful
fallbacks (DummyGPIO and espeak fallback) so the code runs on development
machines and CI without hardware.

## Events & Adapter

State transition events are dicts: `{timestamp, state, prev_state, score, reason}`.
The `DriverFatigueAdapter.analyze()` accepts an optional `events` parameter and
adds state-derived features to `driver_fatigue_features`.

## Demo

Run `python examples/driver_fatigue_state_demo.py` to simulate a sequence of
scores and see events/alerts printed (and DummyGPIO used when RPi GPIO is not
available).
