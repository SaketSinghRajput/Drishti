"""Time utility helpers."""
from __future__ import annotations

from datetime import datetime, timezone


def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_timestamp(ts: str) -> datetime:
    # naive parser for ISO 8601 with Z
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def timestamp_to_epoch(ts: str) -> float:
    return parse_timestamp(ts).timestamp()


def is_timestamp_recent(ts: str, max_age_seconds: float) -> bool:
    dt = parse_timestamp(ts)
    now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() <= max_age_seconds
