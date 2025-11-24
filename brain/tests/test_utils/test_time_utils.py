from datetime import datetime, timezone, timedelta

from utils.time_utils import get_utc_timestamp, parse_timestamp, timestamp_to_epoch, is_timestamp_recent


def test_time_utils_basic():
    ts = get_utc_timestamp()
    assert ts.endswith("Z")

    dt = parse_timestamp(ts)
    assert dt.tzinfo is not None

    epoch = timestamp_to_epoch(ts)
    assert isinstance(epoch, float)

    assert is_timestamp_recent(ts, 5)

    # old timestamp should not be recent
    old = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    assert not is_timestamp_recent(old, 60)
