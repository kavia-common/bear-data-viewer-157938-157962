#!/usr/bin/env python3
"""
detect_bear_motion.py

A CLI utility to determine whether a bear has MOVED across frames based on
centroid displacement relative to the bounding box size threshold per frame.

Inputs:
- CSV file path via CLI.
- CSV must contain at least a frame index and bounding box columns in one of the forms:
  1) frame, cx, cy, w, h
  2) frame, x, y, w, h
  3) frame, x1, y1, x2, y2

Computation steps:
1. Parse rows, tolerating missing/invalid rows with warnings.
2. For each valid row, compute the centroid:
   - If cx,cy given: centroid = (cx, cy)
   - Else if x,y given: centroid = (x + w/2, y + h/2)
   - Else if corners (x1,y1,x2,y2) given: centroid = ((x1+x2)/2, (y1+y2)/2)
3. For each frame in time order, compute displacement from previous valid centroid.
4. For each frame, compute bounding box size metric for that frame:
   - width,height from:
       - (w,h) directly
       - (x2-x1, y2-y1) for corners
   - size metric chosen from:
       - max: max(width, height)
       - min: min(width, height)
       - diag: sqrt(width^2 + height^2)
5. A frame is considered "moved" if displacement > threshold_scale * size_metric.
6. Finally print a summary to stdout:
   - Total frames processed (valid ones used)
   - Number of motion events (frames with moved=True; first frame has no displacement and is counted as False)
   - MOVED: True/False  (True if any frame exceeded threshold)

Optional:
- --threshold-scale (float, default 1.0) to scale the size metric before comparison.
- --size-metric (max|min|diag) choose bounding box size metric.
- --per-frame to print a CSV of per-frame displacement and moved flag to stdout.
  The per-frame CSV columns: frame, cx, cy, width, height, size_metric, displacement, moved
- The script prints human-readable summary by default.

Dependencies:
- Uses only the Python standard library.

Examples:
  Basic usage:
    python detect_bear_motion.py ./boxes.csv

  Use diagonal size metric and a 0.5 scale:
    python detect_bear_motion.py ./boxes.csv --size-metric diag --threshold-scale 0.5

  Output per-frame CSV:
    python detect_bear_motion.py ./boxes.csv --per-frame

  Combine:
    python detect_bear_motion.py ./boxes.csv --size-metric max --threshold-scale 1.25 --per-frame
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Any


def _warn(msg: str) -> None:
    """Print a warning to stderr."""
    print(f"[WARN] {msg}", file=sys.stderr)


def _err(msg: str) -> None:
    """Print an error to stderr."""
    print(f"[ERROR] {msg}", file=sys.stderr)


def _lower_strip(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _parse_float(val: Any) -> Optional[float]:
    """Attempt to parse a value into float. Return None if invalid."""
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _detect_column_indices(headers: List[str]) -> Dict[str, int]:
    """
    Detect usable column indices from header row.
    Supports any of the following sets:
      - frame, cx, cy, w, h
      - frame, x, y, w, h
      - frame, x1, y1, x2, y2

    Returns:
        dict mapping of detected keys to indices.
        Keys may include: frame,cx,cy,x,y,w,h,x1,y1,x2,y2
        If detection fails, returns empty dict.
    """
    idx: Dict[str, int] = {}
    if not headers:
        return idx

    hmap: Dict[str, int] = {}
    for i, h in enumerate(headers):
        key = _lower_strip(h)
        if key:
            hmap[key] = i

    # minimal requirement: frame + some bbox form
    if "frame" not in hmap:
        return {}

    # Prefer explicit centroid form first
    if all(k in hmap for k in ("cx", "cy", "w", "h")):
        idx["frame"] = hmap["frame"]
        idx["cx"] = hmap["cx"]
        idx["cy"] = hmap["cy"]
        idx["w"] = hmap["w"]
        idx["h"] = hmap["h"]
        return idx

    # Then try x,y,w,h
    if all(k in hmap for k in ("x", "y", "w", "h")):
        idx["frame"] = hmap["frame"]
        idx["x"] = hmap["x"]
        idx["y"] = hmap["y"]
        idx["w"] = hmap["w"]
        idx["h"] = hmap["h"]
        return idx

    # Then x1,y1,x2,y2
    if all(k in hmap for k in ("x1", "y1", "x2", "y2")):
        idx["frame"] = hmap["frame"]
        idx["x1"] = hmap["x1"]
        idx["y1"] = hmap["y1"]
        idx["x2"] = hmap["x2"]
        idx["y2"] = hmap["y2"]
        return idx

    return {}


def _compute_centroid_and_size(row: List[str], idx: Dict[str, int]) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Compute centroid (cx,cy) and (width,height) for a CSV row given detected indices.

    Returns:
        tuple: (frame_index, cx, cy, width, height)
        or None if the row is invalid/unusable.
    """
    # Parse frame index
    frame_val = row[idx["frame"]] if "frame" in idx and idx["frame"] < len(row) else None
    frame_f = _parse_float(frame_val)
    if frame_f is None:
        _warn(f"Skipping row: invalid frame index value={frame_val!r}")
        return None
    # keep as int if possible, else cast int of float
    try:
        frame_idx = int(frame_f)
    except Exception:
        frame_idx = int(round(frame_f))

    # Case: cx,cy,w,h
    if all(k in idx for k in ("cx", "cy", "w", "h")):
        cx = _parse_float(row[idx["cx"]]) if idx["cx"] < len(row) else None
        cy = _parse_float(row[idx["cy"]]) if idx["cy"] < len(row) else None
        w = _parse_float(row[idx["w"]]) if idx["w"] < len(row) else None
        h = _parse_float(row[idx["h"]]) if idx["h"] < len(row) else None
        if None in (cx, cy, w, h):
            _warn(f"Skipping row frame={frame_idx}: missing/invalid cx,cy,w,h")
            return None
        if w <= 0 or h <= 0:
            _warn(f"Skipping row frame={frame_idx}: non-positive width/height (w={w}, h={h})")
            return None
        return (frame_idx, float(cx), float(cy), float(w), float(h))

    # Case: x,y,w,h
    if all(k in idx for k in ("x", "y", "w", "h")):
        x = _parse_float(row[idx["x"]]) if idx["x"] < len(row) else None
        y = _parse_float(row[idx["y"]]) if idx["y"] < len(row) else None
        w = _parse_float(row[idx["w"]]) if idx["w"] < len(row) else None
        h = _parse_float(row[idx["h"]]) if idx["h"] < len(row) else None
        if None in (x, y, w, h):
            _warn(f"Skipping row frame={frame_idx}: missing/invalid x,y,w,h")
            return None
        if w <= 0 or h <= 0:
            _warn(f"Skipping row frame={frame_idx}: non-positive width/height (w={w}, h={h})")
            return None
        cx = x + w / 2.0
        cy = y + h / 2.0
        return (frame_idx, float(cx), float(cy), float(w), float(h))

    # Case: x1,y1,x2,y2
    if all(k in idx for k in ("x1", "y1", "x2", "y2")):
        x1 = _parse_float(row[idx["x1"]]) if idx["x1"] < len(row) else None
        y1 = _parse_float(row[idx["y1"]]) if idx["y1"] < len(row) else None
        x2 = _parse_float(row[idx["x2"]]) if idx["x2"] < len(row) else None
        y2 = _parse_float(row[idx["y2"]]) if idx["y2"] < len(row) else None
        if None in (x1, y1, x2, y2):
            _warn(f"Skipping row frame={frame_idx}: missing/invalid x1,y1,x2,y2")
            return None
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            _warn(f"Skipping row frame={frame_idx}: non-positive width/height from corners (w={w}, h={h})")
            return None
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (frame_idx, float(cx), float(cy), float(w), float(h))

    # Should not reach here if indices were validated
    _warn(f"Skipping row frame={frame_idx}: unsupported column configuration")
    return None


def _size_metric_value(width: float, height: float, metric: str) -> float:
    metric = (metric or "max").lower()
    if metric == "min":
        return float(min(width, height))
    if metric == "diag":
        return float(math.hypot(width, height))
    # default max
    return float(max(width, height))


def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return math.hypot(dx, dy)


def _read_csv_compute(records_path: str,
                      size_metric: str,
                      threshold_scale: float,
                      per_frame: bool) -> Tuple[int, int, bool, List[Dict[str, Any]]]:
    """
    Read CSV, compute per-frame displacement and moved flag.

    Returns:
        total_frames, num_motion_events, moved_any, per_frame_rows
        per_frame_rows includes dicts with:
            frame, cx, cy, width, height, size_metric, displacement, moved
    """
    if not os.path.exists(records_path) or not os.path.isfile(records_path):
        _err(f"CSV file not found: {records_path}")
        return 0, 0, False, []

    per_frame_rows: List[Dict[str, Any]] = []

    with open(records_path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            _err(f"CSV file is empty: {records_path}")
            return 0, 0, False, []

        idx = _detect_column_indices(headers)
        if not idx:
            _err("Failed to detect required columns. Expect one of:\n"
                 "  - frame,cx,cy,w,h\n"
                 "  - frame,x,y,w,h\n"
                 "  - frame,x1,y1,x2,y2")
            return 0, 0, False, []

        parsed: List[Tuple[int, float, float, float, float]] = []
        line_no = 1  # account for header
        for row in reader:
            line_no += 1
            if not row or all((c is None or str(c).strip() == "") for c in row):
                _warn(f"Skipping empty row at line {line_no}")
                continue
            try:
                res = _compute_centroid_and_size(row, idx)
                if res is not None:
                    parsed.append(res)
            except Exception as e:
                _warn(f"Skipping row at line {line_no}: parse error ({e})")
                continue

    if not parsed:
        _warn("No valid rows found after parsing. Nothing to process.")
        return 0, 0, False, []

    # Sort by frame index to ensure temporal order
    parsed.sort(key=lambda t: t[0])

    total_frames = len(parsed)
    motion_events = 0
    moved_any = False

    last_cx = None
    last_cy = None

    for i, (frame, cx, cy, w, h) in enumerate(parsed):
        size_val = _size_metric_value(w, h, size_metric)
        threshold = threshold_scale * size_val

        if last_cx is None or last_cy is None:
            disp = 0.0
            moved = False
        else:
            disp = _euclidean(cx, cy, last_cx, last_cy)
            moved = disp > threshold

        if moved:
            motion_events += 1
            moved_any = True

        per_frame_rows.append({
            "frame": frame,
            "cx": cx,
            "cy": cy,
            "width": w,
            "height": h,
            "size_metric": size_val,
            "displacement": disp,
            "moved": moved,
        })

        last_cx, last_cy = cx, cy

    return total_frames, motion_events, moved_any, per_frame_rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Determine bear motion based on centroid displacement vs bounding-box size threshold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV columns accepted (case-insensitive):
  (1) frame,cx,cy,w,h
  (2) frame,x,y,w,h
  (3) frame,x1,y1,x2,y2

Rules:
  - Centroid is computed per row.
  - Displacement is Euclidean distance between consecutive valid frame centroids.
  - A frame is considered 'moved' if displacement > threshold_scale * size_metric(width,height)
  - size_metric can be one of:
      * max  : max(width, height)         [default]
      * min  : min(width, height)
      * diag : sqrt(width^2 + height^2)

Examples:
  Basic:
    python detect_bear_motion.py ./boxes.csv

  Use diagonal metric with scale 0.75 and print per-frame details:
    python detect_bear_motion.py ./boxes.csv --size-metric diag --threshold-scale 0.75 --per-frame

  Use max metric with scale 1.5:
    python detect_bear_motion.py ./boxes.csv --size-metric max --threshold-scale 1.5
"""
    )
    parser.add_argument(
        "csv_path",
        help="Path to CSV file with frame and bounding box columns."
    )
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="Scale factor multiplied by the chosen size metric per frame (default: 1.0)."
    )
    parser.add_argument(
        "--size-metric",
        choices=["max", "min", "diag"],
        default="max",
        help="Bounding box size metric to compare against (default: max)."
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Print per-frame CSV (frame,cx,cy,width,height,size_metric,displacement,moved) to stdout."
    )
    return parser


# PUBLIC_INTERFACE
def main() -> None:
    """CLI entrypoint to detect motion from a CSV of bounding boxes.

    Parameters:
        csv_path (positional): Path to CSV with bounding box columns.
        --threshold-scale (float): Multiplier for size metric (default 1.0).
        --size-metric (str): One of max|min|diag (default max).
        --per-frame (flag): If set, prints per-frame CSV to stdout.

    Output:
        - If --per-frame is set, prints a CSV header and per-frame rows:
            frame,cx,cy,width,height,size_metric,displacement,moved
        - Always prints a summary:
            Total frames: <N>
            Motion events: <K>
            MOVED: True|False

    Behavior:
        - Skips malformed rows with warnings.
        - Sorts frames by the 'frame' value.
        - The first valid frame has no previous frame, so displacement is 0 and moved=False.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    total, motions, moved, rows = _read_csv_compute(
        records_path=args.csv_path,
        size_metric=args.size_metric,
        threshold_scale=float(args.threshold_scale),
        per_frame=bool(args.per_frame),
    )

    if args.per_frame:
        # Print per-frame CSV
        out = csv.writer(sys.stdout)
        out.writerow(["frame", "cx", "cy", "width", "height", "size_metric", "displacement", "moved"])
        for r in rows:
            out.writerow([
                r["frame"],
                f"{r['cx']:.6f}",
                f"{r['cy']:.6f}",
                f"{r['width']:.6f}",
                f"{r['height']:.6f}",
                f"{r['size_metric']:.6f}",
                f"{r['displacement']:.6f}",
                "True" if r["moved"] else "False",
            ])

    # Print summary (human-readable)
    print(f"Total frames: {total}")
    print(f"Motion events: {motions}")
    print(f"MOVED: {bool(moved)}")


if __name__ == "__main__":
    main()
