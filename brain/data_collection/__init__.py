"""Data collection package.

Provides:
- DataLogger: orchestrates sensor polling, windowing, and storage
- WindowManager: sliding window aggregation
- CircularBuffer: persistent JSON storage with cleanup
- sensors: sensor handler implementations
"""

from .logger import DataLogger
from .window_manager import WindowManager
from .buffer import CircularBuffer
from .environment_adapter import EnvironmentAdapter

__all__ = ["DataLogger", "WindowManager", "CircularBuffer", "EnvironmentAdapter"]
