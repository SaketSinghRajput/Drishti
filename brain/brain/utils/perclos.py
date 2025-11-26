"""Simple PERCLOS computation helper.

Provides a tiny helper to compute a naive PERCLOS (proportion of eyelid/iris 'closed')
over a rolling window given iris vertical positions. This is intentionally small and
approximate for demonstration/testing only.
"""
from typing import Iterable, List


def compute_perclos_from_iris_y(iris_y_history: Iterable[float], closed_threshold: float = 0.02) -> float:
    """Compute naive PERCLOS from a history of iris center Y positions.

    Args:
        iris_y_history: iterable of iris center y positions (normalized 0..1)
        closed_threshold: vertical displacement threshold (normalized) below which
                          the eye is considered closed (default 0.02 ~ small value)

    Returns:
        proportion (0..1) of samples considered closed.
    """
    hist = [v for v in iris_y_history if v is not None]
    if not hist:
        return 0.0
    closed = [1 for v in hist if v <= closed_threshold]
    return float(sum(closed)) / float(len(hist))


def rolling_perclos(iris_centers: List[tuple], side: str = "left", window: int = 30, closed_threshold: float = 0.02) -> float:
    """Compute perclos for a sliding window of recent iris centers.

    iris_centers: list of per-frame iris info dicts like `(lx,ly)` or tuples; this function assumes the
    second value is the normalized y.
    """
    ys = []
    for c in iris_centers[-window:]:
        if not c:
            ys.append(None)
            continue
        # accept either dict {'center':(x,y)} or (x,y) tuple
        if isinstance(c, dict) and "center" in c:
            ys.append(c["center"][1])
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            ys.append(c[1])
        else:
            ys.append(None)
    return compute_perclos_from_iris_y(ys, closed_threshold=closed_threshold)
