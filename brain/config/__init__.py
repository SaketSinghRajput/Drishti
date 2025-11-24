"""Configuration package.

Usage:
- Use `ConfigLoader()` to get the process-wide configuration singleton.
- To load or switch to a different file, call `ConfigLoader(path)` or
	`ConfigLoader().set_path(path)` which will reload the configuration.

The loader validates a set of required top-level sections on load.
"""

from .config_loader import ConfigLoader

__all__ = ["ConfigLoader"]
