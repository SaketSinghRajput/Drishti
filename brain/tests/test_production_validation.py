import pytest


def test_smoke_imports():
    # Basic smoke test to ensure core modules import
    try:
        import data_collection
        import brain
    except Exception as e:
        pytest.fail(f"Importing core packages failed: {e}")


@pytest.mark.production
def test_driver_fatigue_handler_minimal_init():
    # Ensure DriverFatigueHandler can be constructed with minimal config
    from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
    cfg = {}
    h = DriverFatigueHandler(cfg, use_cv=False)
    assert h is not None
