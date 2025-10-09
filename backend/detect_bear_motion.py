#!/usr/bin/env python3
"""
detect_bear_motion.py

CSV-driven detection loader and simple grouping/filter utility for YOLO detections.

This script reads a CSV file with headers:
  Preferred: frame_time_seconds,label,x1,y1,x2,y2,confidence
  Backward-compatible: time_frame_in_secs,label,x1,y1,x2,y2,confidence (a warning will be printed)

It provides CLI controls to:
  - Filter by minimum confidence
  - Filter by allowed labels
  - Group detections by time_frame_in_secs or return a flat list
  - Optionally write normalized JSON output to a file
  - Print a concise summary of parsed and valid detections

Notes:
- Only Python stdlib is used.
- Video access is optional; the --video flag is accepted for future integration
  (e.g., extracting frames), but this script does not implement heavy video processing.

Example usage:
  python detect_bear_motion.py --csv /path/to/dets.csv
  python detect_bear_motion.py --csv dets.csv --min-confidence 0.5 --labels bear,dog
  python detect_bear_motion.py --csv dets.csv --grouping flat --output out.json
  python detect_bear_motion.py --csv dets.csv --grouping frame --output out.json --quiet
  python detect_bear_motion.py --csv dets.csv --move-threshold 10.0 --match-method centroid
  python detect_bear_motion.py --csv dets.csv --match-method iou --iou-threshold 0.4
  python detect_bear_motion.py --csv dets.csv --no-movement  # disables movement computation

Acceptance/behavior:
- Validates CSV headers: prefers frame_time_seconds; accepts legacy time_frame_in_secs with a warning.
- Parses rows with robust type conversion; skips malformed rows with warnings (unless --quiet).
- Filters by --min-confidence and --labels (case-sensitive by default; see implementation).
- grouping == 'frame' -> detections: { "0.0": [ {...}, ... ], "10.0": [ ... ] }
- grouping == 'flat'  -> detections: [ {...}, {...}, ... ] where each dict has time_frame_in_secs
- JSON schema: {"schema":"bear_motion_v1", "grouping":"frame|flat", "detections": <object|array>, "movements": [ ... ]}
- When movement is enabled (default), associates detections between consecutive frames using centroid or IoU.
- For matched pairs with centroid displacement > --move-threshold, emits BEAR_MOVED lines unless --quiet.

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union


# Updated required headers to new preferred name while maintaining backward compatibility in validation logic
REQUIRED_HEADERS = [
    "frame_time_seconds",
    "label",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
]


def _print(msg: str, quiet: bool) -> None:
    """Print info message unless quiet is True."""
    if not quiet:
        print(msg)


def _warn(msg: str, quiet: bool) -> None:
    """Print a warning to stderr unless quiet is True."""
    if not quiet:
        print(f"[WARN] {msg}", file=sys.stderr)


def _err(msg: str) -> None:
    """Print an error to stderr."""
    print(f"[ERROR] {msg}", file=sys.stderr)


def _parse_float(val: Any) -> Optional[float]:
    """Try to parse a number to float; return None if invalid."""
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _parse_label(val: Any) -> Optional[str]:
    """Parse label as non-empty string; return None if invalid."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _validate_headers(headers: List[str]) -> bool:
    """
    Validate CSV headers. Accept both:
      - New preferred: frame_time_seconds,label,x1,y1,x2,y2,confidence
      - Legacy (backward-compatible): time_frame_in_secs,label,x1,y1,x2,y2,confidence

    Returns:
        bool: True if valid, False otherwise.
    """
    if headers is None:
        return False
    if headers == REQUIRED_HEADERS:
        return True
    legacy = [
        "time_frame_in_secs",
        "label",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    ]
    return headers == legacy


def _normalize_label_list(labels_arg: Optional[str]) -> Optional[List[str]]:
    """Convert comma-separated string into list of labels; return None if not provided."""
    if not labels_arg:
        return None
    arr = [s.strip() for s in labels_arg.split(",")]
    arr = [s for s in arr if s]  # remove empty
    return arr if arr else None


def _row_to_detection(
    row: Dict[str, str], quiet: bool
) -> Optional[Dict[str, Any]]:
    """
    Convert a CSV DictReader row into a normalized detection dict.

    Expected keys:
      - Preferred: frame_time_seconds,label,x1,y1,x2,y2,confidence
      - Legacy accepted with warning: time_frame_in_secs,label,x1,y1,x2,y2,confidence

    Returns:
        dict with keys:
            - time_frame_in_secs (float)
            - label (str)
            - bbox: {x1,y1,x2,y2} (floats or ints depending on input)
            - confidence (float)
        or None if malformed.
    """
    # Parse time from preferred or legacy field
    t_val = row.get("frame_time_seconds")
    legacy_val = row.get("time_frame_in_secs")
    if t_val is None and legacy_val is not None:
        # Legacy key present; warn once per process (we'll set a module flag)
        if not globals().get("_LEGACY_HEADER_WARNED", False):
            _warn("Detected legacy CSV header 'time_frame_in_secs'. Please migrate to 'frame_time_seconds'.", quiet)
            globals()["_LEGACY_HEADER_WARNED"] = True
        t_val = legacy_val
    t = _parse_float(t_val)
    if t is None:
        _warn(f"Skipping row due to invalid frame time value: {t_val!r}", quiet)
        return None

    # Parse label
    label = _parse_label(row.get("label"))
    if label is None:
        _warn("Skipping row due to missing/empty label", quiet)
        return None

    # Parse bbox coords
    x1 = _parse_float(row.get("x1"))
    y1 = _parse_float(row.get("y1"))
    x2 = _parse_float(row.get("x2"))
    y2 = _parse_float(row.get("y2"))
    if None in (x1, y1, x2, y2):
        _warn("Skipping row due to invalid bbox coordinates", quiet)
        return None
    # Ensure bbox is valid: allow x2<x1 or y2<y1? We'll accept as-is; caller logic may handle.
    # If needed, we could normalize to min/max, but spec does not require.

    # Parse confidence
    conf = _parse_float(row.get("confidence"))
    if conf is None:
        _warn("Skipping row due to invalid confidence", quiet)
        return None
    if conf < 0.0 or conf > 1.0:
        _warn(f"Skipping row due to confidence outside [0,1]: {conf}", quiet)
        return None

    det = {
        "time_frame_in_secs": float(t),
        "label": label,
        "bbox": {
            "x1": x1 if x1 is not None else 0.0,
            "y1": y1 if y1 is not None else 0.0,
            "x2": x2 if x2 is not None else 0.0,
            "y2": y2 if y2 is not None else 0.0,
        },
        "confidence": float(conf),
    }
    return det


# PUBLIC_INTERFACE
def load_csv(
    csv_path: str,
    min_confidence: float = 0.0,
    labels: Optional[List[str]] = None,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load and parse a detection CSV.

    PUBLIC_INTERFACE
    Args:
        csv_path: Path to CSV file with headers exactly:
                  time_frame_in_secs,label,x1,y1,x2,y2,confidence
        min_confidence: Minimum confidence to include (0..1).
        labels: Optional list of labels to include (exact match).
        quiet: Suppress warnings/info if True.

    Returns:
        List of detection dicts:
            {
              "time_frame_in_secs": float,
              "label": str,
              "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
              "confidence": float
            }

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If header validation fails.
    """
    if not os.path.exists(csv_path) or not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    detections: List[Dict[str, Any]] = []
    total_rows = 0
    malformed_rows = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty: {csv_path}")

        if not _validate_headers(reader.fieldnames):
            expected = ",".join(REQUIRED_HEADERS)
            legacy = "time_frame_in_secs,label,x1,y1,x2,y2,confidence"
            raise ValueError(
                "Invalid CSV headers. Expected either (preferred):\n"
                f"{expected}\n"
                "or legacy:\n"
                f"{legacy}\n"
                f"Got: {','.join(reader.fieldnames)}"
            )

        for row in reader:
            total_rows += 1
            det = _row_to_detection(row, quiet=quiet)
            if det is None:
                malformed_rows += 1
                continue

            # Filter by confidence
            if float(det["confidence"]) < float(min_confidence):
                continue

            # Filter by labels if provided (exact match)
            if labels is not None and det["label"] not in labels:
                continue

            detections.append(det)

    _print(
        f"Parsed rows: {total_rows}, Malformed: {malformed_rows}, Valid detections after filters: {len(detections)}",
        quiet,
    )
    return detections


# PUBLIC_INTERFACE
def filter_detections(
    detections: Iterable[Dict[str, Any]],
    min_confidence: float = 0.0,
    labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    PUBLIC_INTERFACE
    Filter detections by confidence and labels.

    Args:
        detections: Iterable of detection dicts as produced by load_csv().
        min_confidence: Minimum confidence threshold (inclusive).
        labels: Optional list of allowed labels (exact match).

    Returns:
        Filtered list of detections.
    """
    out: List[Dict[str, Any]] = []
    for det in detections:
        try:
            conf_ok = float(det.get("confidence", 0.0)) >= float(min_confidence)
            label_ok = (labels is None) or (det.get("label") in labels)
            if conf_ok and label_ok:
                out.append(det)
        except Exception:
            # If malformed, skip silently here.
            continue
    return out


# PUBLIC_INTERFACE
def group_by_frame(
    detections: Iterable[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    PUBLIC_INTERFACE
    Group detections by their time_frame_in_secs value (stringified).

    Args:
        detections: Iterable of detection dicts.

    Returns:
        Dict mapping "time_frame_in_secs" (as string) to list of detection dicts
        with the shape: {"label": str, "bbox": {...}, "confidence": float}

    Notes:
        - Keys are strings for JSON friendliness ("0.0" etc.).
        - The items in the grouped lists contain only label, bbox, confidence.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        t = det.get("time_frame_in_secs")
        try:
            t_str = f"{float(t):.6f}"  # stable string representation
        except Exception:
            # If time is invalid, skip
            continue

        grouped.setdefault(t_str, []).append(
            {
                "label": det.get("label"),
                "bbox": {
                    "x1": det.get("bbox", {}).get("x1"),
                    "y1": det.get("bbox", {}).get("y1"),
                    "x2": det.get("bbox", {}).get("x2"),
                    "y2": det.get("bbox", {}).get("y2"),
                },
                "confidence": det.get("confidence"),
            }
        )
    return grouped


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser for the CSV-based detection loader and movement detector.
    """
    parser = argparse.ArgumentParser(
        description="Parse CSV YOLO detections and optionally group, filter, export JSON, and detect movement across frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV requirements:
  Preferred headers: frame_time_seconds,label,x1,y1,x2,y2,confidence
  Legacy accepted with warning: time_frame_in_secs,label,x1,y1,x2,y2,confidence

Examples:
  python detect_bear_motion.py --csv dets.csv
  python detect_bear_motion.py --csv dets.csv --min-confidence 0.5 --labels bear
  python detect_bear_motion.py --csv dets.csv --grouping flat --output out.json
  python detect_bear_motion.py --csv dets.csv --grouping frame --output out.json --quiet
  python detect_bear_motion.py --csv dets.csv --move-threshold 12.0 --match-method centroid
  python detect_bear_motion.py --csv dets.csv --match-method iou --iou-threshold 0.4
""",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the metadata CSV (required).",
    )
    parser.add_argument(
        "--video",
        help="Optional video path; accepted for future frame extraction needs (not used in this script).",
        default=None,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold to include (default: 0.0).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels to include (exact match). Example: 'bear,dog'",
    )
    parser.add_argument(
        "--grouping",
        choices=["frame", "flat"],
        default="frame",
        help="Grouping mode: 'frame' to group detections by time_frame_in_secs, 'flat' to emit all detections (default: frame).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If provided, write normalized JSON to this path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose messages and warnings.",
    )
    # Movement detection options
    parser.add_argument(
        "--move-threshold",
        type=float,
        default=10.0,
        help="Movement distance threshold in pixels for centroid displacement (default: 10.0).",
    )
    parser.add_argument(
        "--match-method",
        type=str,
        choices=["centroid", "iou"],
        default="centroid",
        help="Association method between consecutive frames: centroid or iou (default: centroid).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold used when --match-method iou is selected (default: 0.3).",
    )
    parser.add_argument(
        "--movement",
        action="store_true",
        default=True,
        help="Enable movement computation across consecutive frames (default: enabled). Use --no-movement to disable.",
    )
    # Provide a --no-movement to disable (argparse boolean pair)
    parser.add_argument(
        "--no-movement",
        dest="movement",
        action="store_false",
        help="Disable movement computation.",
    )
    return parser


def _summary(detections: List[Dict[str, Any]], quiet: bool) -> Tuple[int, int, Optional[float], Optional[float]]:
    """
    Compute summary metrics for detections.

    Returns:
        total_rows (int), frames_found (int), min_ts (float|None), max_ts (float|None)
    """
    total_rows = len(detections)
    frames = set()
    min_ts: Optional[float] = None
    max_ts: Optional[float] = None
    for det in detections:
        t = det.get("time_frame_in_secs")
        try:
            tf = float(t)
        except Exception:
            # Skip time aggregation for malformed time
            continue
        frames.add(tf)
        min_ts = tf if min_ts is None else min(min_ts, tf)
        max_ts = tf if max_ts is None else max(max_ts, tf)

    frames_found = len(frames)
    return total_rows, frames_found, min_ts, max_ts


def _write_json(
    path: str,
    grouping: str,
    grouped_or_flat: Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]],
    quiet: bool,
    movements: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Write normalized JSON output to path.

    JSON structure:
      {
        "schema": "bear_motion_v1",
        "grouping": "frame" | "flat",
        "detections": {... or [...]},
        "movements": [ ... ]   # optional, when movement detection enabled
      }
    """
    payload = {
        "schema": "bear_motion_v1",
        "grouping": grouping,
        "detections": grouped_or_flat,
    }
    if movements is not None:
        payload["movements"] = movements
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _print(f"Wrote JSON to: {path}", quiet)
    except Exception as e:
        _err(f"Failed to write JSON to {path}: {e}")


# PUBLIC_INTERFACE
def _centroid(bbox: Dict[str, Any]) -> Tuple[float, float]:
    """Compute centroid (cx, cy) from bbox dict with x1,y1,x2,y2."""
    try:
        x1 = float(bbox.get("x1"))
        y1 = float(bbox.get("y1"))
        x2 = float(bbox.get("x2"))
        y2 = float(bbox.get("y2"))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    except Exception:
        return (float("nan"), float("nan"))


def _iou(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
    """Compute IoU between two bboxes; returns 0.0 if invalid."""
    try:
        ax1, ay1, ax2, ay2 = float(b1["x1"]), float(b1["y1"]), float(b1["x2"]), float(b1["y2"])
        bx1, by1, bx2, by2 = float(b2["x1"]), float(b2["y1"]), float(b2["x2"]), float(b2["y2"])
    except Exception:
        return 0.0

    # Normalize boxes (min/max) to be robust to x2<x1 inputs
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _associate_pairs(
    prev: List[Dict[str, Any]],
    cur: List[Dict[str, Any]],
    method: str = "centroid",
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Greedy association between prev and cur lists based on method.

    Returns:
        list of (prev_index, cur_index) pairs.
    """
    pairs: List[Tuple[int, int]] = []
    used_prev: set = set()
    used_cur: set = set()

    if method == "iou":
        # Build all candidate pairs with IoU >= threshold, sorted by IoU desc
        candidates: List[Tuple[float, int, int]] = []
        for i, a in enumerate(prev):
            for j, b in enumerate(cur):
                iou = _iou(a.get("bbox", {}), b.get("bbox", {}))
                if iou >= float(iou_threshold):
                    candidates.append((iou, i, j))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, i, j in candidates:
            if i in used_prev or j in used_cur:
                continue
            pairs.append((i, j))
            used_prev.add(i)
            used_cur.add(j)
    else:
        # Centroid-based: use nearest neighbor by Euclidean distance
        import math
        # Precompute centroids
        prev_c = [_centroid(d.get("bbox", {})) for d in prev]
        cur_c = [_centroid(d.get("bbox", {})) for d in cur]
        candidates: List[Tuple[float, int, int]] = []
        for i, (cx1, cy1) in enumerate(prev_c):
            if any(map(lambda v: v != v, (cx1, cy1))):
                continue
            for j, (cx2, cy2) in enumerate(cur_c):
                if any(map(lambda v: v != v, (cx2, cy2))):
                    continue
                dist = math.hypot(cx2 - cx1, cy2 - cy1)
                candidates.append((dist, i, j))
        candidates.sort(key=lambda x: x[0])
        for _, i, j in candidates:
            if i in used_prev or j in used_cur:
                continue
            pairs.append((i, j))
            used_prev.add(i)
            used_cur.add(j)
    return pairs


def iter_detections_per_frame(
    detections: Iterable[Dict[str, Any]]
) -> Iterator[Tuple[float, List[Dict[str, Any]]]]:
    """
    PUBLIC_INTERFACE
    Provide an iteration API to process detections per frame.

    Yields:
        (time_frame_in_secs, detections_for_frame)
        where detections_for_frame are detection dicts of the form returned by load_csv().

    Notes:
        - Frames are yielded in ascending time order.
    """
    by_frame: Dict[float, List[Dict[str, Any]]] = {}
    for det in detections:
        try:
            t = float(det.get("time_frame_in_secs"))
        except Exception:
            # Skip malformed
            continue
        by_frame.setdefault(t, []).append(det)

    for t in sorted(by_frame.keys()):
        yield t, by_frame[t]


# PUBLIC_INTERFACE
def main() -> None:
    """
    CLI entrypoint to parse CSV detections, filter, group, and optionally export JSON.

    CLI:
      --csv <path> (required)
      --video <path> (optional; not used by this script beyond acceptance)
      --min-confidence <float> (default 0.0)
      --labels <comma-separated> (optional, exact match)
      --grouping <frame|flat> (default 'frame')
      --output <path> (optional)
      --quiet (flag) suppresses informational prints and warnings

    Behavior:
      - Validates required headers.
      - Parses rows robustly and skips malformed with warnings (unless quiet).
      - Filters by confidence and labels if provided.
      - If grouping == 'frame', produce a dict keyed by time string with arrays of {label,bbox,confidence}.
      - If grouping == 'flat', produce a list of normalized detection dicts.
      - If --output is provided, write the normalized JSON to the file.

    Summary printed:
      - Total rows (valid detections after filters)
      - Frames found
      - Min/Max timestamps
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    csv_path = args.csv
    # Note: args.video is accepted for future frame extraction needs but not used here.
    quiet = bool(args.quiet)

    # Parse label filters
    labels = _normalize_label_list(args.labels)

    try:
        detections = load_csv(
            csv_path=csv_path,
            min_confidence=float(args["min_confidence"]) if isinstance(args, dict) and "min_confidence" in args else float(args.min_confidence),
            labels=labels,
            quiet=quiet,
        )
    except FileNotFoundError as e:
        _err(str(e))
        sys.exit(1)
    except ValueError as e:
        _err(str(e))
        sys.exit(2)
    except Exception as e:
        _err(f"Unexpected error during CSV load: {e}")
        sys.exit(1)

    # At this point detections are already filtered by min-confidence and labels inside load_csv.
    # If we wanted to apply additional filtering externally, we could call filter_detections again.

    # Group or keep flat
    if args.grouping == "frame":
        grouped = group_by_frame(detections)
        grouped_or_flat: Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]] = grouped
    else:
        grouped_or_flat = detections

    # Movement computation across consecutive frames
    movements: List[Dict[str, Any]] = []
    if bool(args.movement):
        # Build per-frame list sorted by timestamp
        frames = list(iter_detections_per_frame(detections))
        # Iterate adjacent frame pairs and associate detections
        for idx in range(len(frames) - 1):
            ts_prev, dets_prev = frames[idx]
            ts_cur, dets_cur = frames[idx + 1]
            # Respect label and confidence already filtered
            pairs = _associate_pairs(
                dets_prev, dets_cur, method=str(args.match_method), iou_threshold=float(args.iou_threshold)
            )
            # For each pair, compute centroid displacement
            for pi, ci in pairs:
                prev_det = dets_prev[pi]
                cur_det = dets_cur[ci]
                c1 = _centroid(prev_det.get("bbox", {}))
                c2 = _centroid(cur_det.get("bbox", {}))
                try:
                    import math
                    dx = float(c2[0]) - float(c1[0])
                    dy = float(c2[1]) - float(c1[1])
                    dist = math.hypot(dx, dy)
                except Exception:
                    continue

                # Assign a simple track id by combining frame index and pair index when no prior mapping
                track_id = f"t{idx}_{pi}->{ci}"
                # Emit movement event if distance exceeds threshold
                if dist > float(args.move_threshold):
                    event = {
                        "ts": float(ts_cur),
                        "id": track_id,
                        "dx": dx,
                        "dy": dy,
                        "dist": dist,
                        "bbox": {
                            "x1": cur_det.get("bbox", {}).get("x1"),
                            "y1": cur_det.get("bbox", {}).get("y1"),
                            "x2": cur_det.get("bbox", {}).get("x2"),
                            "y2": cur_det.get("bbox", {}).get("y2"),
                        },
                        "conf": float(cur_det.get("confidence", 0.0)),
                        "label": cur_det.get("label"),
                    }
                    movements.append(event)
                    if not quiet:
                        # Print exactly formatted BEAR_MOVED line
                        x1 = event["bbox"]["x1"]
                        y1 = event["bbox"]["y1"]
                        x2 = event["bbox"]["x2"]
                        y2 = event["bbox"]["y2"]
                        conf = event["conf"]
                        print(
                            f"BEAR_MOVED ts={event['ts']:.6f} id={track_id} "
                            f"dx={dx:.2f} dy={dy:.2f} dist={dist:.2f} "
                            f"bbox=({x1},{y1},{x2},{y2}) conf={conf:.2f}"
                        )

    # Write JSON if requested
    if args.output:
        _write_json(args.output, args.grouping, grouped_or_flat, quiet, movements=movements if bool(args.movement) else None)

    # Compute and print summary
    total_rows, frames_found, min_ts, max_ts = _summary(detections, quiet)

    print(f"Total rows: {total_rows}")
    print(f"Frames found: {frames_found}")
    if min_ts is not None and max_ts is not None:
        print(f"Min timestamp: {min_ts}")
        print(f"Max timestamp: {max_ts}")
    else:
        print("Min timestamp: N/A")
        print("Max timestamp: N/A")


if __name__ == "__main__":
    # Ensure module is import-safe
    main()
