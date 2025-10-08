# PUBLIC_INTERFACE
def compute_distance(a, b):
    """Compute Euclidean distance between two points a and b.

    Args:
        a (tuple|list): Point A as (x, y)
        b (tuple|list): Point B as (x, y)

    Returns:
        float: Euclidean distance.

    Notes:
        Basic helper used by movement logic.
    """
    try:
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
    except (TypeError, ValueError, IndexError):
        return float("inf")
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def _parse_timestamp_to_epoch_seconds(ts):
    """Parse timestamp to epoch seconds.

    Accepts:
      - ISO8601 strings (e.g., '2024-10-01T12:00:00Z', or '+00:00' timezone)
      - Numeric epoch seconds (int/float or numeric string)

    Returns:
        float: epoch seconds (UTC). If parsing fails, returns None.
    """
    if ts is None:
        return None
    # numeric fast path
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        # string numeric
        return float(str(ts))
    except Exception:
        pass

    # ISO8601 parsing
    try:
        from datetime import datetime
        s = str(ts).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        # Attempt dateutil if available
        try:
            from dateutil import parser as dateutil_parser  # optional
            return dateutil_parser.isoparse(str(ts)).timestamp()
        except Exception:
            return None


def _centroid_from_pose_keypoints(pose):
    """Compute centroid from pose keypoints if available.

    Args:
        pose: May be:
              - dict with 'keypoints': list of [x,y,(optional visibility/conf)]
              - list of keypoints directly
              - dict with 'x','y' (already centroid)
              - None

    Returns:
        tuple|None: (x, y) centroid if derivable else None.
    """
    if pose is None:
        return None

    # Direct x,y
    if isinstance(pose, dict) and "x" in pose and "y" in pose:
        try:
            return float(pose["x"]), float(pose["y"])
        except Exception:
            pass

    # Keypoints under dict
    kpts = None
    if isinstance(pose, dict) and "keypoints" in pose:
        kpts = pose.get("keypoints")

    # Or list directly
    if kpts is None and isinstance(pose, list):
        kpts = pose

    if not kpts:
        return None

    xs, ys = [], []
    for kp in kpts:
        # kp can be [x,y] or [x,y,conf]
        try:
            x, y = float(kp[0]), float(kp[1])
            # if visibility given and is 0, you may skip; here we accept all visible to keep generic
            xs.append(x)
            ys.append(y)
        except Exception:
            continue

    if not xs or not ys:
        return None

    return sum(xs) / len(xs), sum(ys) / len(ys)


def _position_from_detection(item):
    """Resolve a single detection's (x, y) position.

    Priority:
      1) item['position'] as dict/list/tuple; supports {'x':..,'y':..} or (x,y)
      2) centroid from pose keypoints in item['pose']

    Returns:
        tuple|None: (x, y) or None if unavailable.
    """
    if not isinstance(item, dict):
        return None

    # Try position
    pos = item.get("position")
    if isinstance(pos, dict) and "x" in pos and "y" in pos:
        try:
            return float(pos["x"]), float(pos["y"])
        except Exception:
            pass
    elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return float(pos[0]), float(pos[1])
        except Exception:
            pass

    # Try pose centroid
    return _centroid_from_pose_keypoints(item.get("pose"))


def _basic_smooth_position_sequence(seq):
    """Apply a very basic smoothing over a list of (x,y) positions.

    Strategy:
      - Single pass moving average with window size 3.

    Args:
        seq (list[tuple|None]): positions (x,y) or None

    Returns:
        list[tuple|None]: smoothed positions (None preserved where no data)
    """
    n = len(seq)
    if n == 0:
        return seq

    out = [None] * n
    for i in range(n):
        vals = []
        for j in (i - 1, i, i + 1):
            if 0 <= j < n and seq[j] is not None:
                vals.append(seq[j])
        if not vals:
            out[i] = None
        else:
            sx = sum(v[0] for v in vals) / len(vals)
            sy = sum(v[1] for v in vals) / len(vals)
            out[i] = (sx, sy)
    return out


# PUBLIC_INTERFACE
def is_moving(detections_for_bear, at_index, time_window=10, distance_threshold=15):
    """Determine if a bear is moving for a specific detection index using ±time_window seconds.

    Definition:
      A detection is classified as "moving" if the Euclidean distance between the position at
      index 'at_index' and any other observation for the same bear within the time window
      [t-∆, t+∆] (∆ = time_window seconds) exceeds distance_threshold pixels. Otherwise, "stationary".

    Args:
        detections_for_bear (list[dict]): Detections for a single bear. Each dict should include:
            - 'timestamp' (ISO8601 or epoch seconds)
            - 'position' as {'x','y'} or [x,y] OR 'pose' with 'keypoints' to derive centroid.
            Optional: 'pose' can also be a string (pose label) which is returned in annotate_movement.
        at_index (int): Index within detections_for_bear to evaluate.
        time_window (int|float): Half-window in seconds to consider on either side (default 10).
        distance_threshold (int|float): Distance threshold in pixels (default 15).

    Returns:
        bool: True if moving, False if stationary.

    Edge cases:
        - If timestamp cannot be parsed or position can't be derived, the function
          attempts to use available neighbors. If no valid comparison exists, returns False.
    """
    if not detections_for_bear or at_index < 0 or at_index >= len(detections_for_bear):
        return False

    # Sort by timestamp for consistent neighbor scanning
    items = list(detections_for_bear)
    for it in items:
        it["_epoch"] = _parse_timestamp_to_epoch_seconds(it.get("timestamp"))

    items = [it for it in items if it["_epoch"] is not None]
    items.sort(key=lambda x: x["_epoch"])

    # Re-find our index in sorted list
    # If duplicate dict instances exist, this finds by identity; else match by a stable key if available.
    target = detections_for_bear[at_index]
    target_epoch = _parse_timestamp_to_epoch_seconds(target.get("timestamp"))
    if target_epoch is None:
        return False

    # Build positions aligned to 'items'
    positions = [_position_from_detection(it) for it in items]
    # Basic smoothing
    positions = _basic_smooth_position_sequence(positions)

    # Find index of target_epoch in items
    targ_idx = None
    for i, it in enumerate(items):
        if it["_epoch"] == target_epoch:
            targ_idx = i
            break
    if targ_idx is None:
        return False

    t0 = items[targ_idx]["_epoch"]
    # Scan neighbors within ± time_window
    moved = False
    for i, it in enumerate(items):
        if i == targ_idx:
            continue
        if abs(it["_epoch"] - t0) <= float(time_window):
            p0 = positions[targ_idx]
            p1 = positions[i]
            if p0 is None or p1 is None:
                continue
            if compute_distance(p0, p1) > float(distance_threshold):
                moved = True
                break

    return moved


# PUBLIC_INTERFACE
def annotate_movement(detections, time_window=10, distance_threshold=15):
    """Annotate every detection with movement_status: 'moving' or 'stationary'.

    The function groups detections by 'bear_id' (or 'bearId'), sorts each group by timestamp,
    and for each detection checks if the distance to any neighbor within ±time_window exceeds
    distance_threshold after basic smoothing.

    Args:
        detections (list[dict]): Input detections. Expected fields per item:
            - bear_id or bearId (string)
            - timestamp (ISO string or epoch seconds)
            - position: {'x','y'} or [x,y] (optional)
            - pose: either a label string or a dict with 'keypoints' to derive centroid (optional)
        time_window (int|float): seconds half window size (default 10).
        distance_threshold (int|float): pixels (default 15).

    Returns:
        list[dict]: New list of records with fields:
            - bear_id (string)
            - pose (string if provided else '')
            - timestamp (string)
            - movement_status ('moving'|'stationary')

    Error handling:
        - Skips items missing bear_id and timestamp.
        - If timestamp malformed, item is skipped.
        - If both position and keypoints absent, movement is inferred from available comparisons; with no
          valid neighbors, defaults to 'stationary'.
    """
    if not isinstance(detections, list):
        return []

    # Normalize bear id key and filter invalid entries quickly
    norm = []
    for it in detections:
        if not isinstance(it, dict):
            continue
        bid = it.get("bear_id", it.get("bearId"))
        ts = it.get("timestamp")
        if not bid or ts is None:
            continue
        # Ensure timestamp parseable
        if _parse_timestamp_to_epoch_seconds(ts) is None:
            continue
        norm.append({**it, "bear_id": bid})

    # Group by bear_id
    from collections import defaultdict
    groups = defaultdict(list)
    for it in norm:
        groups[it["bear_id"]].append(it)

    annotated = []
    for bid, items in groups.items():
        # Sort by time ascending
        items.sort(key=lambda x: _parse_timestamp_to_epoch_seconds(x.get("timestamp")) or 0.0)
        for idx, item in enumerate(items):
            moving = is_moving(items, idx, time_window=time_window, distance_threshold=distance_threshold)
            pose_value = item.get("pose")
            # If pose is a dict (e.g., keypoints), return empty string or a summary string
            if isinstance(pose_value, dict):
                # Optional: include a brief marker to signal structured pose present
                pose_out = pose_value.get("label") if isinstance(pose_value.get("label", None), str) else ""
            else:
                pose_out = pose_value if isinstance(pose_value, str) else ""
            annotated.append({
                "bear_id": bid,
                "pose": pose_out,
                "timestamp": item.get("timestamp"),
                "movement_status": "moving" if moving else "stationary",
            })

    # Sort final output by timestamp descending to align with typical UI use
    annotated.sort(key=lambda x: _parse_timestamp_to_epoch_seconds(x.get("timestamp")) or 0.0, reverse=True)
    return annotated
