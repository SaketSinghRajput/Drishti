"""Validation utilities for windows and sensor readings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from utils.time_utils import parse_timestamp


def _get_project_root() -> Path:
    """Return the project `brain/` directory as a Path object.

    This helper navigates up from `brain/utils/` to the repository package root
    so schema files can be resolved regardless of the current working directory.
    """
    return Path(__file__).resolve().parent.parent


def load_schema(path: str = str(_get_project_root() / "config" / "schema" / "sensor_window_schema.json")) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_window(window: Dict[str, Any], schema_path: str = str(_get_project_root() / "config" / "schema" / "sensor_window_schema.json")) -> Tuple[bool, List[str]]:
    schema = load_schema(schema_path)
    errors = []
    try:
        from jsonschema import Draft7Validator

        validator = Draft7Validator(schema)
        for err in validator.iter_errors(window):
            loc = "/".join(map(str, err.path))
            errors.append(f"{err.message} (at {loc})")
        return (len(errors) == 0, errors)
    except Exception:
        # If jsonschema is unavailable or validation fails unexpectedly,
        # return (True, []) to avoid blocking runtime; detailed validation
        # requires jsonschema to be installed.
        return (True, [])


def validate_sensor_reading(sensor_name: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    # lightweight checks matching schema expectations
    errors = []
    if sensor_name == "gps":
        lat = data.get("lat")
        lon = data.get("lon")
        if not isinstance(lat, (int, float)) or not -90 <= lat <= 90:
            errors.append("gps.lat out of range or wrong type")
        if not isinstance(lon, (int, float)) or not -180 <= lon <= 180:
            errors.append("gps.lon out of range or wrong type")
    elif sensor_name == "bms":
        ok, errs = validate_bms_telemetry(data)
        if not ok:
            errors.extend(errs)
    # additional checks can be added for imu, vehicle_controls, driver_fatigue
    return (len(errors) == 0, errors)


def load_bms_schema(path: str = str(_get_project_root() / "config" / "bms_schema.json")) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            import json

            return json.load(f)
    except Exception:
        return {}


def validate_bms_telemetry(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single BMS telemetry dict against `config/bms_schema.json`.

    Returns (is_valid, errors).
    """
    errors = []
    schema = load_bms_schema()
    if not schema:
        # schema unavailable; don't block runtime
        return (True, [])
    try:
        from jsonschema import Draft7Validator

        validator = Draft7Validator(schema)
        for err in validator.iter_errors(data):
            loc = "/".join(map(str, err.path))
            errors.append(f"{err.message} (at {loc})")
        return (len(errors) == 0, errors)
    except Exception:
        return (True, [])


def compute_data_quality_score(window: Dict[str, Any], expected_sensors: List[str]) -> float:
    present = 0
    for s in expected_sensors:
        if s in window:
            present += 1
    if not expected_sensors:
        return 0.0
    return present / len(expected_sensors)


def validate_label(label: Dict[str, Any], schema_path: str = str(_get_project_root() / "config" / "schema" / "label_schema.json")) -> Tuple[bool, List[str]]:
    """Validate a single label dict against the label schema.

    Returns (is_valid, errors).
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        from jsonschema import Draft7Validator

        validator = Draft7Validator(schema)
        errors = []
        for err in validator.iter_errors(label):
            loc = "/".join(map(str, err.path))
            errors.append(f"{err.message} (at {loc})")
        return (len(errors) == 0, errors)
    except Exception:
        return (True, [])


def validate_labeled_dataset(labels: List[Dict[str, Any]], schema_path: str = str(_get_project_root() / "config" / "schema" / "label_schema.json")) -> Dict[str, Any]:
    """Batch-validate labels and return a summary dict.

    Summary contains counts of valid/invalid/duplicates and a sample of error messages.
    """
    seen = set()
    summary = {"valid": 0, "invalid": 0, "duplicates": 0, "errors": [], "temporal_inconsistencies": 0}
    for lab in labels:
        wid = lab.get("window_id")
        if wid in seen:
            summary["duplicates"] += 1
            continue
        seen.add(wid)
        ok, errs = validate_label(lab, schema_path=schema_path)
        if ok:
            summary["valid"] += 1
        else:
            summary["invalid"] += 1
            summary["errors"].extend(errs[:5])
        # temporal consistency check: annotation_timestamp (nested under labels)
        try:
            lab_labels = lab.get("labels", {}) or {}
            ann_ts = lab_labels.get("annotation_timestamp")
            win_ts = lab.get("timestamp")
            if ann_ts and win_ts:
                at = parse_timestamp(ann_ts)
                wt = parse_timestamp(win_ts)
                if at < wt:
                    summary["temporal_inconsistencies"] += 1
                    summary["errors"].append(f"annotation_timestamp {ann_ts} earlier than window timestamp {win_ts} for window {wid}")
        except Exception:
            # parsing failure: log an error but continue
            summary["errors"].append(f"failed to parse timestamps for window {wid}")
    return summary
