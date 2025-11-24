try:
    import RPi.GPIO as GPIO
    print("Imported real RPi.GPIO")
except Exception:
    print("Running in SIMULATION mode — hardware not available")
    from unittest.mock import MagicMock
    GPIO = MagicMock()