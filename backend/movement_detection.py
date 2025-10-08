#!/usr/bin/env python3
"""
movement_detection.py

A self-contained module to classify movement status ("moving", "stationary", "unknown")
for bear detections in video footage. Each detection is compared to its neighbors
(with the same bear_id) within a ±window_seconds time window. If the minimum distance
to any neighbor in that window is greater than or equal to the distance_threshold,
the detection is labeled "moving"; otherwise "stationary". If there are no neighbors
in the window, the detection is labeled "unknown".

Features:
- Detection dataclass to represent inputs
- Configurable time window (±window_seconds) and distance_threshold (float)
- Pluggable distance metrics: euclidean (default) and haversine_like
- Grouping by bear_id and per-bear sorting by timestamp
- Efficient neighbor lookup via binary search on sorted timestamps
- Edge-case handling: unordered input, identical timestamps, empty inputs
- Optional smoothing utilities (not applied by default)
- __main__ demo with synthetic data for two bears; prints JSON results

Usage:
- Programmatic:
    from movement_detection import Detection, classify_movements
    detections = [
        Detection(bear_id="B001", timestamp=0.0, x=10, y=10),
        Detection(bear_id="B001", timestamp=8.0, x=12, y=12),
        Detection(bear_id="B001", timestamp=16.0, x=55, y=55),
    ]
    results = classify_movements(detections, window_seconds=10.0, distance_threshold=5.0, metric="euclidean")
    # results -> list[dict] with original fields + movement_status and min_neighbor_distance

- CLI demo:
    python movement_detection.py
    # Prints JSON lines with movement classification for sample data

Interpretation notes:
- The distance_threshold should be tuned to the coordinate scale. For normalized [0..1]
  coordinates, a threshold like 0.05 may be reasonable; for pixel space, a threshold like
  5.0 to 20.0 may be appropriate depending on resolution and motion magnitude.
- haversine_like is included as a generic spherical distance approximation if (x, y) represent
  latitude and longitude (degrees). It is not a strict geodesic implementation but a commonly
  used approximation.
- Optional smoothing utilities can be used by callers to post-process per-bear sequences
  to remove isolated spikes, but they are not invoked by default within classify_movements.

Environment:
- Pure Python; no external dependencies.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import bisect
import json
import math


@dataclass(frozen=True)
class Detection:
    """
    Represents a single detection of a bear at a point in time.

    Fields:
        bear_id: Identifier for the bear (grouping key).
        timestamp: Time in seconds (float). Can be fractional.
        x: X coordinate (pixel or normalized).
        y: Y coordinate (pixel or normalized).
        pose: Optional pose information (e.g., keypoints dict). Carried through but not required.

    Notes:
        - The data class is intentionally minimal; real systems might include
          additional fields such as confidence scores, bounding box, etc.
    """
    bear_id: str
    timestamp: float
    x: float
    y: float
    pose: Optional[dict] = None


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance in a 2D plane."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _haversine_like(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Haversine-like distance assuming inputs are (lat, lon) in degrees.

    Returns:
        Distance in kilometers (approx), using Earth's mean radius ~6371 km.

    Notes:
        - Included as a simple alternative metric. If you pass non-geographic
          coordinates, results will be meaningless.
    """
    # Convert degrees to radians
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(s))
    r = 6371.0  # km
    return r * c


_METRICS: Dict[str, Callable[[Tuple[float, float], Tuple[float, float]], float]] = {
    "euclidean": _euclidean,
    "haversine_like": _haversine_like,
}


def _get_metric(metric: str) -> Callable[[Tuple[float, float], Tuple[float, float]], float]:
    """Resolve metric function from string name; defaults to euclidean."""
    return _METRICS.get(metric.lower(), _euclidean)


def _build_index_by_bear(detections: Iterable[Detection]) -> Dict[str, List[Detection]]:
    """
    Group detections by bear_id and sort each group by timestamp.

    Returns:
        dict: bear_id -> list of Detection sorted by timestamp ascending.
    """
    by_bear: Dict[str, List[Detection]] = {}
    for d in detections:
        if not isinstance(d.bear_id, str):
            # Skip malformed entries
            continue
        # Ensure timestamps are float-compatible
        try:
            _ = float(d.timestamp)
        except Exception:
            continue
        by_bear.setdefault(d.bear_id, []).append(d)

    for bid, lst in by_bear.items():
        lst.sort(key=lambda x: float(x.timestamp))
    return by_bear


def _bounds_for_time_window(timestamps: List[float], center_t: float, window: float) -> Tuple[int, int]:
    """
    Return the slice [lo, hi) indices for timestamps within [center_t - window, center_t + window].
    Uses bisect for efficient boundary search on sorted timestamps.
    """
    start = center_t - window
    end = center_t + window
    lo = bisect.bisect_left(timestamps, start)
    hi = bisect.bisect_right(timestamps, end)
    return lo, hi


def _min_neighbor_distance_for_index(
    positions: List[Tuple[float, float]],
    timestamps: List[float],
    idx: int,
    window_seconds: float,
    dist_fn: Callable[[Tuple[float, float], Tuple[float, float]], float],
) -> Optional[float]:
    """
    Compute minimum distance from positions[idx] to any neighbor within ±window_seconds.

    Excludes the identical index (idx) even if timestamps are identical.

    Returns:
        float: minimum distance, or None if no neighbors in window or if position is missing (NaN).
    """
    if not positions or idx < 0 or idx >= len(positions):
        return None

    p0 = positions[idx]
    if p0 is None or any(math.isnan(v) for v in p0):
        return None

    t0 = timestamps[idx]
    lo, hi = _bounds_for_time_window(timestamps, t0, float(window_seconds))
    min_d: Optional[float] = None
    for j in range(lo, hi):
        if j == idx:
            continue
        p1 = positions[j]
        if p1 is None or any(math.isnan(v) for v in p1):
            continue
        # Exclude identical (x, y) in case of duplicates? We keep them; the rule is index exclusion only.
        d = dist_fn(p0, p1)
        if (min_d is None) or (d < min_d):
            min_d = d
    return min_d


def _safe_tuple(x: float, y: float) -> Tuple[float, float]:
    """Convert to a safe tuple, coercing to float and handling errors."""
    try:
        xf = float(x)
        yf = float(y)
        return (xf, yf)
    except Exception:
        return (math.nan, math.nan)


def _basic_smooth_positions(positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Basic 1D moving-average smoothing (window size 3) applied independently to x and y.

    None/NaN values are skipped; if no valid neighborhood, the original value is preserved.

    Returns:
        New list of smoothed positions (same length).
    """
    n = len(positions)
    if n == 0:
        return positions[:]

    out: List[Tuple[float, float]] = []
    for i in range(n):
        xs: List[float] = []
        ys: List[float] = []
        for j in (i - 1, i, i + 1):
            if 0 <= j < n:
                p = positions[j]
                if p is None:
                    continue
                if any(math.isnan(v) for v in p):
                    continue
                xs.append(p[0])
                ys.append(p[1])
        if xs and ys:
            out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        else:
            out.append(positions[i])
    return out


# PUBLIC_INTERFACE
def classify_movements(
    detections: List[Detection],
    window_seconds: float = 10.0,
    distance_threshold: float = 5.0,
    metric: str = "euclidean",
    apply_smoothing: bool = False,
) -> List[dict]:
    """
    Classify movement status for each detection.

    For each detection, looks at other detections of the same bear within
    [t - window_seconds, t + window_seconds], excluding itself. Computes the minimum
    distance to any neighbor in that window. If min_distance >= distance_threshold,
    label is "moving"; otherwise "stationary". If there are no neighbors in window,
    label is "unknown".

    Args:
        detections: List of Detection objects (unordered allowed).
        window_seconds: Half-window in seconds on each side (float allowed).
        distance_threshold: Threshold for movement classification (float allowed).
        metric: Distance metric to use ("euclidean" or "haversine_like"). Defaults to "euclidean".
        apply_smoothing: If True, applies a simple moving average smoothing on per-bear positions
                        before computing distances. Off by default.

    Returns:
        list[dict]: Each dict includes the original Detection fields plus:
            - movement_status: "moving" | "stationary" | "unknown"
            - min_neighbor_distance: float or None
    """
    if not detections:
        return []

    dist_fn = _get_metric(metric)
    by_bear = _build_index_by_bear(detections)

    enriched_results: List[dict] = []

    for bear_id, group in by_bear.items():
        # Build aligned arrays for this bear
        timestamps = [float(d.timestamp) for d in group]
        positions = [_safe_tuple(d.x, d.y) for d in group]
        if apply_smoothing:
            positions = _basic_smooth_positions(positions)

        for i, d in enumerate(group):
            min_d = _min_neighbor_distance_for_index(
                positions=positions,
                timestamps=timestamps,
                idx=i,
                window_seconds=window_seconds,
                dist_fn=dist_fn,
            )
            if min_d is None:
                status = "unknown"
            else:
                status = "moving" if min_d >= float(distance_threshold) else "stationary"

            # Build output dict with original fields plus computed ones
            out = asdict(d)
            out["movement_status"] = status
            out["min_neighbor_distance"] = min_d
            enriched_results.append(out)

    # Preserve original overall input order as much as possible while still having results for all entries:
    # We sort results by (bear_id, timestamp) to be deterministic, but if stable original order is desired,
    # you can instead build an index map from original detection identities. For simplicity, we sort by time.
    enriched_results.sort(key=lambda r: (r["bear_id"], float(r["timestamp"])))
    return enriched_results


# PUBLIC_INTERFACE
def build_index_by_bear(detections: List[Detection]) -> Dict[str, List[Detection]]:
    """Public helper to group detections by bear_id and sort by timestamp."""
    return _build_index_by_bear(detections)


# PUBLIC_INTERFACE
def smooth_statuses(
    per_bear_results: List[dict],
    min_consensus: int = 2,
) -> List[dict]:
    """
    Optional post-processing to smooth isolated spikes in movement_status per bear.

    Strategy:
    - For each bear_id, look at the sequence of statuses ordered by timestamp.
    - If a "moving" status is surrounded by 'min_consensus' neighbors that are all "stationary",
      flip it to "stationary". Similarly, if a "stationary" status is surrounded by "moving",
      flip to "moving".
    - Boundaries are handled gracefully by only counting available neighbors.

    Params:
        per_bear_results: Output list from classify_movements (same schema).
        min_consensus: Number of neighbors needed on each side to consider a flip.
                       Example: 2 means look at up to 2 neighbors on each side.

    Returns:
        New list[dict] with possibly adjusted movement_status values.
        The order of results is preserved within each bear group (sorted by timestamp).
    """
    if not per_bear_results:
        return []

    # Group by bear
    groups: Dict[str, List[dict]] = {}
    for item in per_bear_results:
        bid = item.get("bear_id")
        if not isinstance(bid, str):
            continue
        groups.setdefault(bid, []).append(item)

    # Sort within each bear by timestamp
    for bid, lst in groups.items():
        lst.sort(key=lambda r: float(r.get("timestamp", 0.0)))

    def _flip_if_isolated(seq: List[dict]) -> List[dict]:
        out = [dict(x) for x in seq]
        n = len(seq)
        for i in range(n):
            cur = seq[i].get("movement_status")
            if cur not in ("moving", "stationary"):
                continue

            # Collect neighbors
            left = []
            right = []
            for j in range(1, min_consensus + 1):
                if i - j >= 0:
                    left.append(seq[i - j].get("movement_status"))
                if i + j < n:
                    right.append(seq[i + j].get("movement_status"))

            if len(left) + len(right) < min_consensus:
                # Not enough neighbors to decide
                continue

            neighbors = [s for s in (left + right) if s in ("moving", "stationary")]
            if not neighbors:
                continue
            # Majority vote among neighbors
            moving_count = sum(1 for s in neighbors if s == "moving")
            stationary_count = sum(1 for s in neighbors if s == "stationary")
            majority = "moving" if moving_count > stationary_count else "stationary"
            if majority != cur and (moving_count >= min_consensus or stationary_count >= min_consensus):
                out[i]["movement_status"] = majority
        return out

    smoothed: List[dict] = []
    for bid, seq in groups.items():
        smoothed.extend(_flip_if_isolated(seq))
    # Keep deterministic ordering: by bear, then timestamp
    smoothed.sort(key=lambda r: (r.get("bear_id"), float(r.get("timestamp", 0.0))))
    return smoothed


def _demo() -> None:
    """
    Demonstration with small synthetic examples for two bears.
    Prints JSON results to stdout.
    """
    print("Demo: movement classification with ±10s window, distance_threshold=5.0 (euclidean)")

    # Bear A: first two close (stationary), third far (moving)
    bear_a = [
        Detection(bear_id="A", timestamp=0.0, x=10.0, y=10.0),
        Detection(bear_id="A", timestamp=8.0, x=12.0, y=12.0),
        Detection(bear_id="A", timestamp=16.0, x=40.0, y=45.0),
    ]

    # Bear B: sparse single point (unknown), two later close points (stationary)
    bear_b = [
        Detection(bear_id="B", timestamp=1.0, x=100.0, y=100.0),
        Detection(bear_id="B", timestamp=30.0, x=101.0, y=100.5),
        Detection(bear_id="B", timestamp=35.0, x=103.0, y=102.0),
    ]

    dets = bear_a + bear_b
    # Shuffle-like unordered input to demonstrate internal sorting handling
    dets = [dets[2], dets[0], dets[5], dets[1], dets[4], dets[3]]

    results = classify_movements(dets, window_seconds=10.0, distance_threshold=5.0, metric="euclidean")
    print(json.dumps(results, indent=2))

    # Optional: show smoothing usage (not applied by default)
    print("\nDemo: optional smoothing (min_consensus=1) for illustration")
    smoothed = smooth_statuses(results, min_consensus=1)
    print(json.dumps(smoothed, indent=2))


if __name__ == "__main__":
    _demo()
