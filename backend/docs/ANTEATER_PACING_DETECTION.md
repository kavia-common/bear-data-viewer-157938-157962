# Anteater Pacing Detection - Specification and Examples

## Overview
This document defines a concise, developer-facing specification for detecting anteater pacing between labeled zones using detection events derived from object tracking. The detection uses bounding box centroids to assign each event to a zone and then evaluates sequences for back-and-forth pacing with a “5-second return” rule.

## Definitions

### Zones
- The video frame contains named polygonal zones:
  - BOX-A, BOX-B, BOX-C, BOX-D, BOX-E, BOX-F, and DATE.
- BOX-A..BOX-F correspond to potential pacing areas; DATE is a special overlay region used for OCR date labels and is not considered a pacing zone.
- Zones are defined as polygons in image coordinates (pixels). The exact coordinates are use-case specific and must be provided by the caller or configured in a project config module.

### Detection Event
- Each detection event represents a tracked anteater at a point in time.
- Input fields:
  - timestamp_ms: integer milliseconds since epoch or video-relative time.
  - bbox: an axis-aligned bounding box dictionary with fields:
    - x1, y1: top-left pixel coordinates (inclusive)
    - x2, y2: bottom-right pixel coordinates (exclusive or inclusive; be consistent)
  - Optional: track_id, confidence, frame_index

### Centroid Calculation
- The centroid of a bounding box is:
  - cx = (x1 + x2) / 2
  - cy = (y1 + y2) / 2
- Use floating point arithmetic and retain subpixel accuracy if available.

### Zone Assignment
- Assign a detection event to a zone by testing the centroid point against the polygon of each zone (point-in-polygon).
- If the centroid falls into multiple zones (should not happen with clean, non-overlapping polygons), use a priority or first-match rule; otherwise, treat as ambiguous and ignore.
- If the centroid is not inside any pacing box (BOX-A..F), label the event as OUTSIDE. The DATE zone is excluded from pacing analysis and should be treated as OUTSIDE.

## Pacing Definition

### 5-Second Return Rule
- A pacing “cycle” is defined as moving from an origin box X to another box Y, and returning to X within 5 seconds (inclusive boundary).
- 5 seconds = 5000 ms.
- Exactly 5000 ms qualifies as within 5 seconds.
- More than 5000 ms (>5000) does not qualify.

### Episode Semantics
- A pacing episode begins at the first detection that initiates a valid cycle (X → Y → X within ≤5000 ms).
- Once an episode has started, it continues as long as the anteater repeatedly travels between the same pair of boxes X and Y, each time returning within ≤5000 ms.
- If a return exceeds 5000 ms, the current episode ends at the last successful return time.
- An episode is specific to a pair {X, Y}. If the anteater switches to involve a third box Z, the current episode ends; a new episode may start for a different pair.

### Debouncing (Optional)
- To reduce noise, consecutive detections assigned to the same zone may be collapsed (e.g., keep the first or compute a stable representative) before sequence analysis.
- Smoothing is optional; the core rule set must not rely on debouncing to be correct.

## Input Format

Provide a list of detections (already filtered to the target species/track) as JSON-like structures:

```json
[
  {"timestamp_ms": 1000, "bbox": {"x1": 100, "y1": 200, "x2": 140, "y2": 260}},
  {"timestamp_ms": 1800, "bbox": {"x1": 480, "y1": 210, "x2": 520, "y2": 260}},
  ...
]
```

- The algorithm will compute centroids and assign zones using configured polygons for BOX-A..F and DATE.

## Output Format

The algorithm returns:
- episodes: a list of pacing episodes with:
  - start_time_ms: first timestamp of the episode
  - end_time_ms: last timestamp of the episode (time of the last qualifying return)
  - origin_box: one of BOX-A..F
  - other_box: the paired box distinct from origin_box
  - cycles: integer count of successful X→Y→X returns within the episode
- annotations: optional per-event annotations (aligned with inputs) including:
  - zone: assigned zone or OUTSIDE
  - event_type: one of ENTER_ZONE, LEAVE_ZONE, TRAVERSE, RETURN_SUCCESS, RETURN_TIMEOUT
  - episode_id: linkage to an episode if applicable

Example structure:

```json
{
  "episodes": [
    {
      "start_time_ms": 1000,
      "end_time_ms": 5600,
      "origin_box": "BOX-A",
      "other_box": "BOX-B",
      "cycles": 2
    }
  ],
  "annotations": [
    {"timestamp_ms": 1000, "zone": "BOX-A", "event_type": "ENTER_ZONE", "episode_id": 1},
    {"timestamp_ms": 3000, "zone": "BOX-B", "event_type": "TRAVERSE", "episode_id": 1},
    {"timestamp_ms": 5600, "zone": "BOX-A", "event_type": "RETURN_SUCCESS", "episode_id": 1}
  ]
}
```

## Examples

For brevity, suppose the centroid-based zone assignment yields this event stream:
- Format: [time_ms] zone

### Example 1: Start and continue episode with 5-second rule
- [1000] BOX-A
- [3000] BOX-B
- [5600] BOX-A
Interpretation:
- A→B→A with (5600 − 3000) = 2600 ms (≤5000) → 1 cycle. Episode starts at 1000 and ends at 5600 (so far), cycles=1.

Continue:
- [7000] BOX-B
- [11000] BOX-A
Interpretation:
- A→B→A with (11000 − 7000) = 4000 ms (≤5000) → 2nd cycle. Episode continues; end_time_ms becomes 11000, cycles=2.

### Example 2: Timeout ends episode
Continuing from Example 1:
- [17000] BOX-B
- [23050] BOX-A
Interpretation:
- A→B→A with (23050 − 17000) = 6050 ms (>5000) → timeout. The previous episode ended at 11000 with cycles=2. The A→B at 17000 may seed a new episode later if a timely return occurs.

### Example 3: Exactly 5 seconds qualifies
- [10000] BOX-C
- [13000] BOX-D
- [18000] BOX-C
Interpretation:
- (18000 − 13000) = 5000 ms → qualifies. One cycle for pair C/D. start=10000, end=18000.

### Example 4: OUTSIDE or DATE is ignored
- [2000] OUTSIDE
- [2600] DATE
- [3000] BOX-E
- [7600] BOX-F
- [12601] BOX-E
Interpretation:
- Only consider BOX-E/F events. (12601 − 7600) = 5001 ms → timeout; no completed cycle.

### Example 5: Different pair ends current episode
- [5000] BOX-A
- [8000] BOX-B
- [10000] BOX-A  → completes A/B cycle #1
- [12000] BOX-C  → new box C appears; A/B episode ends at 10000 with cycles=1
- [14000] BOX-A  → potential new A/C pairing starts if return qualifies

### Example 6: Repeated detections in same zone (debounce concept)
Raw:
- [1000] BOX-A
- [1200] BOX-A
- [1500] BOX-A
- [3000] BOX-B
- [5200] BOX-A
Debounced:
- [1000] BOX-A
- [3000] BOX-B
- [5200] BOX-A → (5200 − 3000) = 2200 ms qualifies → 1 cycle

## Algorithm Outline

### Inputs
- zones: dictionary of named polygons for BOX-A..F and DATE; only BOX-A..F used for pacing.
- detections: list of detection events with timestamp_ms and bbox.

### Steps
1. Preprocess detections
   - Compute centroid for each bbox.
   - Assign zone via point-in-polygon.
   - Optionally debounce consecutive same-zone assignments (keep first or representative).
   - Filter to events with zone in {BOX-A..F}; treat DATE and others as OUTSIDE and ignore for pacing.
2. Track zone transitions
   - Maintain current_origin (X) when entering a pacing box and the last other_box (Y) when traversing away from origin.
   - A candidate cycle starts when sequence shows X → Y (Y ≠ X, both in {A..F}).
3. Apply 5-second return rule
   - If the next time the stream re-enters X is within ≤5000 ms of the entry into Y, count one cycle.
   - If the re-entry occurs after >5000 ms, close any ongoing episode at the last successful return; do not count a new cycle.
   - If the stream goes to a third box Z (Z ≠ X, Z ≠ Y), close the current episode (if any) and consider starting a new pairing with the last two zones seen.
4. Episode management
   - Episode starts at the time of the first X in a valid A↔B pattern.
   - Episode end is the timestamp of the last successful return to X.
   - Maintain per-episode cycle count and pair {origin_box, other_box}.
   - Provide annotations for ENTER_ZONE, TRAVERSE, RETURN_SUCCESS, RETURN_TIMEOUT as helpful diagnostics.
5. Output episodes and annotations.

### Point-in-Polygon
- Use a standard ray casting or winding number method for robust point-in-polygon checks.
- Ensure polygons are defined in image coordinate space (origin at top-left, y increasing downward).

### Smoothing/Debouncing (Optional)
- Time-based hysteresis: require that a zone assignment persists for at least N ms to recognize a transition.
- Spatial smoothing: reject micro-jitters within the same zone threshold.
- These options should be configurable and disabled by default to retain clarity of the core rule.

## Edge Cases
- Exactly 5000 ms: qualifies as a valid return.
- >5000 ms: ends the episode; the attempted cycle does not count.
- Rapid oscillation due to jitter near a boundary: consider optional debouncing to avoid false cycles.
- Overlapping polygons: should not be used; if present, define an explicit priority.
- Missing or unordered timestamps: inputs must be sorted by timestamp_ms ascending; otherwise, sorting is required.
- Multiple individuals or track_ids: handle per tracked entity; this spec assumes a single track stream per run.

## Performance Considerations
- Complexity is O(N) over the number of detections after preprocessing.
- Point-in-polygon checks dominate if many zones or events; precompute polygon bounding boxes for quick rejection, then run precise checks.
- Debouncing reduces event count and improves stability but adds minor state management overhead.
- Memory usage is linear in N for annotations; can be streamed if annotations are optional or batched.

## Developer Notes
- DATE zone exists for OCR (date/time overlay) and should not be counted toward pacing.
- Ensure uniform timestamp units (ms) and consistent bbox conventions.
- Keep the inclusive boundary for the 5-second rule clear in code: delta_ms <= 5000 qualifies.

## Minimal Example With Raw Input

Input detections (bbox in pixels):
```json
[
  {"timestamp_ms": 1000, "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}},
  {"timestamp_ms": 3000, "bbox": {"x1": 200, "y1": 10, "x2": 220, "y2": 30}},
  {"timestamp_ms": 5600, "bbox": {"x1": 15, "y1": 15, "x2": 35, "y2": 35}}
]
```

Assuming:
- Centroids map to BOX-A (near x≈20,y≈20) and BOX-B (near x≈210,y≈20)

Derived zones:
- [1000] BOX-A
- [3000] BOX-B
- [5600] BOX-A → Cycle A/B #1; episode start=1000, end=5600, cycles=1

## References to Project Context
- This repository already includes movement detection and OCR components. This pacing detection is an orthogonal analysis module that can consume detection streams and zone definitions (e.g., from a configuration similar to backend/yolo_pipeline/config.py where zones are defined for other workflows).
