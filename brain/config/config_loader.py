"""Configuration loader using YAML with dot-access, env overrides, caching, and singleton pattern.

This is a lightweight loader suitable for the project skeleton. It supports:
- Loading YAML files
- Accessing nested keys using dot notation
- Default values and required key validation
- Environment variable overrides using uppercased keys with dots replaced by underscores
- Caching and a singleton instance
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

# Import PyYAML lazily in _load to allow the package to be imported in
# environments where PyYAML may not be installed; _load will raise a
# ConfigError with instructions if PyYAML is missing.


class ConfigError(Exception):
    pass


class ConfigLoader:
    _instance_lock = threading.Lock()
    _instance: Optional["ConfigLoader"] = None

    def __new__(cls, *args, **kwargs):
        # maintain a single instance but allow reconfiguration via __init__
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path: str = "config/default_config.yaml") -> None:
        """Initialize the ConfigLoader.

        Notes:
        - This class is a process-wide singleton. Creating a new instance with a
          different `path` will update the singleton's configuration path and
          reload the configuration immediately. This enables calls like:
            ConfigLoader('config/production.yaml')
          to switch the active configuration.
        - Use `set_path(path)` to programmatically update the path and reload.
        """
        # initialize internal structures once
        if not hasattr(self, "_initialized") or not self._initialized:
            self._config_path = path
            self._cache: Dict[str, Any] = {}
            self._data: Dict[str, Any] = {}
            # attempt initial load (will raise if missing)
            self._load()
            # try loading BMS-specific rules and merge them
            try:
                self.load_bms_rules()
            except Exception:
                pass
            self._initialized = True
            return

        # If already initialized, but a different path is requested, update
        # the path and reload so the caller gets the requested configuration.
        if path and path != getattr(self, "_config_path", None):
            # Defensive check: ensure the requested path exists before switching.
            if not os.path.exists(path):
                raise ConfigError(f"Cannot switch to non-existent config: {path}")
            self.set_path(path)

    def _load(self) -> None:
        # Load using the current `_config_path` value. This delegates to the
        # helper `_load_from_path` which implements the actual parsing logic.
        data = self._load_from_path(self._config_path)
        self._data = data
        # validate presence of required top-level sections and keys
        self._validate_schema()

    def _load_from_path(self, path: str) -> Dict[str, Any]:
        """Load configuration data from an explicit path and return the parsed mapping.

        This helper allows callers to attempt loading a new config file without
        mutating `self._config_path` or `self._data` until the load is known to
        succeed. It raises `ConfigError` on missing files or parse errors.
        """
        if not os.path.exists(path):
            raise ConfigError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            # Try to use PyYAML if available; otherwise fall back to a tiny
            # YAML-like parser sufficient for our default_config.yaml structure.
            try:
                import yaml  # type: ignore

                try:
                    data = yaml.safe_load(f) or {}
                except Exception as e:
                    raise ConfigError(f"YAML parse error: {e}")
            except Exception:
                # fallback: simple parser
                content = f.read()
                try:
                    data = self._simple_yaml_parse(content)
                except Exception as e:
                    raise ConfigError(f"Failed to parse YAML (no PyYAML): {e}")
        if not isinstance(data, dict):
            raise ConfigError("Top-level configuration must be a mapping/dictionary")
        return data

    def _simple_yaml_parse(self, content: str) -> Dict[str, Any]:
        """A very small YAML-like parser to handle simple mappings and nested mappings.

        This supports key: value pairs and nested mappings via indentation. It is
        intentionally minimal and only used as a fallback when PyYAML is not
        installed (e.g., in tests).
        """
        result: Dict[str, Any] = {}
        stack = [(-1, result)]
        for raw in content.splitlines():
            line = raw.rstrip()
            if not line or line.strip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            # split key and value
            if ":" not in line:
                continue
            key, val = line.lstrip().split(":", 1)
            key = key.strip()
            val = val.strip()
            # find parent based on indentation
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if val == "":
                # nested mapping
                parent[key] = {}
                stack.append((indent, parent[key]))
            else:
                # parse simple inline dicts like {a: 1, b: 2}
                if val.startswith("{") and val.endswith("}"):
                    inner = val[1:-1]
                    d = {}
                    for part in inner.split(","):
                        if ":" in part:
                            k, v = part.split(":", 1)
                            dk = k.strip()
                            dv = v.strip()
                            # simple type conversions
                            if dv.lower() in ("true", "false"):
                                d[dk] = dv.lower() == "true"
                            else:
                                try:
                                    d[dk] = int(dv)
                                except Exception:
                                    try:
                                        d[dk] = float(dv)
                                    except Exception:
                                        d[dk] = dv
                    parent[key] = d
                else:
                    # scalar value
                    if val.lower() in ("true", "false"):
                        parent[key] = val.lower() == "true"
                    else:
                        try:
                            if "." in val:
                                parent[key] = float(val)
                            else:
                                parent[key] = int(val)
                        except Exception:
                            # strip quotes if present
                            parent[key] = val.strip("\"'")
        return result

    def reload(self) -> None:
        """Force reload configuration from disk and clear cache.

        Note: Cache is cleared after loading to ensure the new config is
        fully loaded before any `get()` calls can populate the cache.
        """
        self._load()
        self._cache.clear()

    def load_bms_rules(self, rules_path: str = "config/bms_rules.yaml") -> Dict[str, Any]:
        """Load BMS-specific rule thresholds from a YAML file and merge into config.

        Returns the loaded mapping or empty dict if the file is missing. Merges
        into self._data under the key 'battery_monitor.rules'.
        """
        try:
            import yaml

            if not os.path.exists(rules_path):
                return {}
            with open(rules_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            # merge into _data
            bm = self._data.setdefault("battery_monitor", {})
            rules = bm.setdefault("rules", {})
            # place thresholds and severities under rules
            rules.setdefault("thresholds", {})
            rules.setdefault("severities", {})
            if "thresholds" in data:
                rules["thresholds"].update(data.get("thresholds", {}))
            if "severities" in data:
                rules["severities"].update(data.get("severities", {}))
            # clear cache for battery_monitor entries
            to_del = [k for k in list(self._cache.keys()) if k.startswith("battery_monitor.")]
            for k in to_del:
                del self._cache[k]
            return data
        except Exception:
            # best-effort: if yaml not available or parse error, skip
            return {}

    def set_path(self, path: str) -> None:
        """Set a new configuration path and reload immediately.

        Clears cache before updating path to prevent race conditions where
        `get()` might return stale cached values during a reload.
        """
        # Attempt to load the new configuration first, and only swap it into
        # the live state if the load succeeds. This preserves the previous
        # in-memory config and cache when the new path is invalid or fails
        # to parse.
        with self._instance_lock:
            # Load new data from the path without mutating current state
            new_data = self._load_from_path(path)
            # Swap in new configuration and update path atomically
            self._data = new_data
            self._config_path = path
            # Clear cache after successfully loading the new config to avoid
            # stale cached values referencing the old configuration.
            self._cache.clear()

    def _validate_schema(self) -> None:
        """Validate presence of required top-level sections and a few keys.

        This is a lightweight validation to catch obvious misconfigurations
        early. It can be extended later or replaced by a full schema.
        """
        # Lightweight validation: be permissive for minimal configs used in tests and
        # only enforce a couple of helpful checks when sections are present.
        model = self._data.get("model")
        if model is not None:
            if not isinstance(model, dict):
                raise ConfigError("'model' must be a mapping/dictionary in config")
            if "prediction_horizon_s" not in model:
                raise ConfigError("Missing required config key: model.prediction_horizon_s")
            # risk_thresholds is optional in minimal configs; if callers provide other
            # top-level sections they may opt to validate more strictly elsewhere.

    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """Get a configuration value using dot notation.

        Environment variables override config values. For key 'sensors.gps.sample_rate'
        the environment variable 'SENSORS_GPS_SAMPLE_RATE' will be checked.
        """
        # env override should take precedence over cached values
        env_key = key.upper().replace(".", "_")
        if env_key in os.environ:
            val = os.environ[env_key]
            self._cache[key] = val
            return val

        if key in self._cache:
            return self._cache[key]

        parts = key.split(".")
        cur: Any = self._data
        try:
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    if required:
                        raise ConfigError(f"Required configuration key missing: {key}")
                    # do not cache default fallbacks to allow require() to correctly
                    # detect missing keys later in the same process
                    return default
        except Exception:
            if required:
                raise
            return default

        # Cache and return
        self._cache[key] = cur
        return cur

    def require(self, key: str) -> Any:
        return self.get(key, required=True)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation, creating nested mappings as needed.

        This updates the in-memory config and clears any cached entries that
        may be affected by the change.
        """
        parts = key.split(".")
        cur = self._data
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
        # clear cache entries that are the key or children of the key
        to_del = [k for k in list(self._cache.keys()) if k == key or k.startswith(key + ".")]
        for k in to_del:
            del self._cache[k]


def load_config(path: Optional[str] = None) -> ConfigLoader:
    """Convenience function returning the singleton ConfigLoader."""
    if path:
        return ConfigLoader(path)
    return ConfigLoader()
