"""Battery Health Monitoring & Prediction module for EV Safety Guardian.

Integrates LibreSolar bms-c1 firmware data via CAN bus, provides real-time
dashboard, rule-based alarms, ML inference for anomaly detection and RUL
prediction, BLE broadcast for critical alerts, and event logging.

Public classes (stubs, implemented in later phases):
- CANIngestion
- RuleEngine
- AlertManager
- Dashboard

This package is intentionally a scaffold for Phase 1 (structure, schemas,
and config). Implementations will be provided in subsequent phases.
"""

__version__ = "0.1.0"

# Import implementation-visible components so they are available from the
# package root. Only export names that are actually present to avoid
# AttributeError on `from battery_monitor import X`.
from .rules import RuleEngine  # type: ignore
from .alerts import AlertManager  # type: ignore


class CANIngestion:
    """Placeholder CAN ingestion facade.

    The real implementation will be provided in the ingestion subpackage
    in a later phase. This placeholder keeps `battery_monitor.CANIngestion`
    available for imports and documentation while work proceeds.
    """

    def __init__(self, config=None):
        self.config = config

    def start(self):
        raise NotImplementedError("CANIngestion is a placeholder in Phase 1")


class Dashboard:
    """Placeholder Dashboard facade.

    The real dashboard factory is `battery_monitor.dashboard.create_app`.
    This class is provided so top-level imports remain stable.
    """

    def __init__(self, app=None):
        self.app = app

    def run(self, *args, **kwargs):
        raise NotImplementedError("Dashboard is a placeholder in Phase 1")


__all__ = ["CANIngestion", "RuleEngine", "AlertManager", "Dashboard", "__version__"]
