"""Dashboard package for the Accident Prediction System.

Provides factory `create_app` and re-exports `DashboardState` for tests.
"""
from .app import create_app
from .state import DashboardState

__all__ = ["create_app", "DashboardState"]
