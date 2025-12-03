"""
Lightweight wrapper that attempts to import CircuitPython ADC/INA drivers and
exposes them (or None) with a simple `_HAS_CIRCUIT` flag. This allows handlers
to import from `brain.utils.voltage_current_handler` and gracefully fall back
when hardware libraries are not installed.

This module is intentionally minimal and does not provide emulation behaviour.
Sensors should check `_HAS_CIRCUIT` and behave accordingly (use simulation).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import board  # type: ignore
    import busio  # type: ignore
    from adafruit_ina219 import ADCResolution, INA219  # type: ignore
    from adafruit_ads1x15.ads1115 import ADS1115  # type: ignore
    from adafruit_ads1x15.analog_in import AnalogIn  # type: ignore

    _HAS_CIRCUIT = True
except Exception as e:  # pragma: no cover - environment-dependent
    logger.debug('CircuitPython ADC drivers not available: %s', e)
    _HAS_CIRCUIT = False

    # Export names for imports elsewhere; set to None so callers can check
    ADCResolution = None
    INA219 = None
    ADS1115 = None
    AnalogIn = None

__all__ = [
    "_HAS_CIRCUIT",
    "ADCResolution",
    "INA219",
    "ADS1115",
    "AnalogIn",
]
