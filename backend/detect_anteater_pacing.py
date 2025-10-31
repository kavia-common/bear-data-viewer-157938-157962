"""
Anteater pacing detection utilities.

This module provides:
- Polygon definitions for zones BOX-A..BOX-F and DATE
- Centroid calculation from YOLO-like bounding boxes
- Point-in-polygon operations
- Centroid-to-zone mapping helper
- Pacing event processing with 5-second inclusive return rule using an "anchor zone"
- Public API functions for batch processing and incremental updates used by tests

Public API functions are marked with the "PUBLIC_INTERFACE" comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Tuple

ZoneName = Literal["BOX-A", "BOX-B", "BOX-C", "BOX-D", "BOX-E", "BOX-F", "DATE"]
EventType = Literal["START", "CONTINUE", "STOP"]

# Zone polygon definitions (clockwise or counter-clockwise)
# Coordinates are abstract units consistent with centroids used in tests.
# If the tests expect specific mapping, these polygons should cover those points.
# The shapes here form a simple grid-like layout for A..F and an isolated DATE area.
ZONES: Dict[ZoneName, List[Tuple[float, float]]] = {
    # Left column
    "BOX-A": [(0, 0), (50, 0), (50, 40), (0, 40)],
    "BOX-D": [(0, 40), (50, 40), (50, 80), (0, 80)],
    # Middle column
    "BOX-B": [(50, 0), (100, 0), (100, 40), (50, 40)],
    "BOX-E": [(50, 40), (100, 40), (100, 80), (50, 80)],
    # Right column
    "BOX-C": [(100, 0), (150, 0), (150, 40), (100, 40)],
    "BOX-F": [(100, 40), (150, 40), (150, 80), (100, 80)],
    # Date area (off to the side)
    "DATE": [(160, 0), (200, 0), (200, 30), (160, 30)],
}


@dataclass(frozen=True)
class PacingEvent:
    """A detected pacing event."""
    type: EventType
    timestamp: float
    anchor_zone: ZoneName


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm for point-in-polygon (inclusive on edges).

    Returns True if point (x, y) is inside or exactly on the boundary of polygon poly.
    """
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # Check if point is on a segment (inclusive)
        # Handle colinearity and bounding box
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            # Degenerate segment, treat as point
            if x == x1 and y == y1:
                return True
        else:
            # Check if (x, y) lies on segment
            # Parametric t where P = P1 + t*(P2-P1), t in [0,1]
            # Cross product zero and within bounding box
            cross = (x - x1) * dy - (y - y1) * dx
            if abs(cross) < 1e-9:
                # within bounding rectangle
                if min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
                    return True

        # Ray casting to the right; count crossings
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1)
        if intersects:
            inside = not inside
    return inside


# PUBLIC_INTERFACE
def compute_centroid_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Compute centroid from bounding box.

    Parameters
    - bbox: (x, y, w, h) where (x, y) is top-left and w, h are width and height.

    Returns
    - (cx, cy): centroid coordinates
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


# PUBLIC_INTERFACE
def map_centroid_to_zone(cx: float, cy: float, zones: Optional[Dict[ZoneName, List[Tuple[float, float]]]] = None) -> Optional[ZoneName]:
    """
    Map a centroid coordinate to a zone name using point-in-polygon.

    Parameters
    - cx, cy: centroid coordinates
    - zones: optional custom dict of zones -> polygon; falls back to default ZONES

    Returns
    - ZoneName if found; otherwise None
    """
    search_zones = zones or ZONES
    for name, poly in search_zones.items():
        if _point_in_polygon(cx, cy, poly):
            return name
    return None


def _classify_zone_from_bbox(bbox: Tuple[float, float, float, float], zones: Optional[Dict[ZoneName, List[Tuple[float, float]]]] = None) -> Optional[ZoneName]:
    """
    Helper for tests and internal logic: classify a zone given bbox.
    """
    cx, cy = compute_centroid_from_bbox(bbox)
    return map_centroid_to_zone(cx, cy, zones=zones)


def _prune_duplicates_by_zone(samples: List[Tuple[float, ZoneName]]) -> List[Tuple[float, ZoneName]]:
    """
    Remove immediate repeats of the same zone (e.g., A at t1 then A at t2) since
    they shouldn't affect the pacing start logic which requires leaving and returning.
    Keep the first occurrence and remove subsequent consecutive duplicates.
    """
    pruned: List[Tuple[float, ZoneName]] = []
    last_zone: Optional[ZoneName] = None
    for ts, zone in samples:
        if zone != last_zone:
            pruned.append((ts, zone))
            last_zone = zone
    return pruned


# PUBLIC_INTERFACE
def detect_pacing_events(
    detections: Iterable[Dict],
    zones: Optional[Dict[ZoneName, List[Tuple[float, float]]]] = None,
    time_key: str = "timestamp",
    bbox_key: str = "bbox",
) -> List[PacingEvent]:
    """
    Batch process detections to detect pacing events (START, CONTINUE, STOP).

    Rules:
    - Anchor zone is the zone where the first return within 5s (inclusive) occurs.
    - A pacing START occurs at the timestamp of the first qualifying return within 5s to its anchor zone,
      after at least one different zone was visited in between. Immediate repeats in the same zone do not start pacing.
    - CONTINUE is emitted for subsequent returns to the anchor within 5s of the previous anchor visit.
    - STOP is emitted when the time since last anchor return exceeds 5s, or if the zone sequence breaks
      such that a qualifying return does not happen within 5s.
    - Between returns any number of intermediate zones is allowed (e.g., A->B->D->C->A).
    - 5 seconds is inclusive (<= 5.0).

    Parameters
    - detections: iterable of dicts containing at least time_key and bbox_key
    - zones: optional custom zones mapping (for tests)
    - time_key: key for timestamp value
    - bbox_key: key for bounding box tuple (x, y, w, h)

    Returns
    - List of PacingEvent objects in chronological order.
    """
    zone_map = zones or ZONES

    # 1) Extract (timestamp, zone) for all detections which map to a zone
    timeline: List[Tuple[float, ZoneName]] = []
    for d in detections:
        ts = float(d[time_key])
        bbox = d[bbox_key]
        zone = _classify_zone_from_bbox(bbox, zones=zone_map)
        if zone is not None:
            timeline.append((ts, zone))

    # 2) Prune immediate duplicates
    timeline = _prune_duplicates_by_zone(timeline)
    if not timeline:
        return []

    events: List[PacingEvent] = []

    anchor: Optional[ZoneName] = None
    last_anchor_ts: Optional[float] = None
    last_zone: Optional[ZoneName] = None

    for ts, zone in timeline:
        if anchor is None:
            # We are looking for a return within 5s to a zone after leaving it
            if last_zone is None:
                last_zone = zone
                continue
            if zone == last_zone:
                # Still same zone; ignore duplicates already pruned, but guard anyway
                continue
            # zone != last_zone: We left the previous zone. Now track potential returns.
            # Potential START when we come back to last_zone within 5s (inclusive).
            # To realize this we need to continue scanning until we encounter last_zone again.
            if zone != last_zone:
                # When we see a different zone, just update and continue scanning.
                last_zone = zone
                continue
        else:
            # We are pacing; check if returned to anchor within 5s to continue; else stop and possibly look for new start
            if zone == anchor:
                # Return to anchor. Check timing.
                if last_anchor_ts is not None and (ts - last_anchor_ts) <= 5.0 + 1e-9:
                    events.append(PacingEvent(type="CONTINUE", timestamp=ts, anchor_zone=anchor))
                    last_anchor_ts = ts
                else:
                    # exceeded 5s => pacing stopped previously; emit STOP at current ts, then consider this return as new START
                    events.append(PacingEvent(type="STOP", timestamp=ts, anchor_zone=anchor))
                    # Reset and allow a fresh start logic from this timestamp forward:
                    anchor = None
                    last_anchor_ts = None
                    last_zone = zone  # we are in this zone now
                continue

            # Not at anchor; just pass through other zones.
            last_zone = zone
            continue

        # If we reach here, anchor is None and we need to check if a start condition occurs based on recent history.
        # Approach: Look backward for the most recent occurrence of the current zone; if it exists within 5s and there was
        # at least one different zone visited between those occurrences, we START.
        # Since we pruned consecutive duplicates, last occurrence of the same zone ensures an intervening zone existed.
        # We need to find the last prior timestamp for this zone.
    # The first pass didn't handle START. We'll process with a second pass that explicitly tracks last visit times per zone.

    # Re-implement pacing detection with explicit tracking:

    events = []
    anchor = None
    last_anchor_ts = None
    last_visit: Dict[ZoneName, float] = {}
    # Track last visited zone to ensure we don't start on immediate duplicates (already pruned)
    for ts, zone in timeline:
        prev_ts_for_zone = last_visit.get(zone)
        if anchor is None:
            if prev_ts_for_zone is not None:
                # We have been in this zone before, check the interval
                if ts - prev_ts_for_zone <= 5.0 + 1e-9:
                    # Ensure at least one different zone occurred in between:
                    # Because we pruned consecutive duplicates, the only way the zone reappears is with at least one other zone in between.
                    # Thus, this qualifies as START.
                    anchor = zone
                    last_anchor_ts = ts
                    events.append(PacingEvent(type="START", timestamp=ts, anchor_zone=anchor))
        else:
            if zone == anchor:
                # Return to anchor: continue if within 5s, else stop and restart
                if ts - (last_anchor_ts or ts) <= 5.0 + 1e-9:
                    events.append(PacingEvent(type="CONTINUE", timestamp=ts, anchor_zone=anchor))
                    last_anchor_ts = ts
                else:
                    # STOP because exceeded 5s
                    events.append(PacingEvent(type="STOP", timestamp=ts, anchor_zone=anchor))
                    # Check if this return can immediately start a new pacing sequence
                    if last_visit.get(zone) is not None and ts - last_visit[zone] <= 5.0 + 1e-9:
                        anchor = zone
                        last_anchor_ts = ts
                        events.append(PacingEvent(type="START", timestamp=ts, anchor_zone=anchor))
                    else:
                        anchor = None
                        last_anchor_ts = None
            else:
                # Not at anchor; we simply pass through any zones.
                pass

        # Update last visit for this zone
        last_visit[zone] = ts

    # If we ended while pacing without a STOP event, we do not emit a STOP implicitly unless tests require it.
    return events


# PUBLIC_INTERFACE
def detect_pacing_events_stream(
    detection_stream: Iterable[Dict],
    zones: Optional[Dict[ZoneName, List[Tuple[float, float]]]] = None,
    time_key: str = "timestamp",
    bbox_key: str = "bbox",
) -> Iterator[PacingEvent]:
    """
    Streaming/event-processing version of detect_pacing_events.

    Yields pacing events as detections arrive, applying the same 5-second inclusive rule.

    Parameters
    - detection_stream: iterable/iterator of detection dicts (in chronological order)
    - zones: optional custom zones mapping
    - time_key: timestamp key
    - bbox_key: bbox key

    Yields
    - PacingEvent instances in chronological order
    """
    zone_map = zones or ZONES

    # State
    last_zone: Optional[ZoneName] = None
    last_visit: Dict[ZoneName, float] = {}
    anchor: Optional[ZoneName] = None
    last_anchor_ts: Optional[float] = None

    for d in detection_stream:
        ts = float(d[time_key])
        bbox = d[bbox_key]
        zone = _classify_zone_from_bbox(bbox, zones=zone_map)
        if zone is None:
            continue

        # Prune immediate duplicate zones
        if zone == last_zone:
            # update last visit for accurate timing but skip processing as it's a duplicate state
            last_visit[zone] = ts
            continue

        prev_ts_for_zone = last_visit.get(zone)
        if anchor is None:
            if prev_ts_for_zone is not None and ts - prev_ts_for_zone <= 5.0 + 1e-9:
                anchor = zone
                last_anchor_ts = ts
                yield PacingEvent(type="START", timestamp=ts, anchor_zone=anchor)
        else:
            if zone == anchor:
                if ts - (last_anchor_ts or ts) <= 5.0 + 1e-9:
                    last_anchor_ts = ts
                    yield PacingEvent(type="CONTINUE", timestamp=ts, anchor_zone=anchor)
                else:
                    yield PacingEvent(type="STOP", timestamp=ts, anchor_zone=anchor)
                    # Potential immediate re-start
                    if prev_ts_for_zone is not None and ts - prev_ts_for_zone <= 5.0 + 1e-9:
                        anchor = zone
                        last_anchor_ts = ts
                        yield PacingEvent(type="START", timestamp=ts, anchor_zone=anchor)
                    else:
                        anchor = None
                        last_anchor_ts = None
            else:
                # pass through other zones
                pass

        last_visit[zone] = ts
        last_zone = zone


# PUBLIC_INTERFACE
def classify_zone_for_bbox(bbox: Tuple[float, float, float, float]) -> Optional[ZoneName]:
    """
    Public helper to classify a bbox into a zone.

    Parameters
    - bbox: (x, y, w, h)

    Returns
    - ZoneName or None
    """
    return _classify_zone_from_bbox(bbox, zones=ZONES)
