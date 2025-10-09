import os
import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# PUBLIC_INTERFACE
def load_detections() -> Tuple[List[Dict[str, Any]], datetime]:
    """Load detections into memory.

    Priority:
      1) If env DETECTIONS_CSV is set to a readable CSV path, load from CSV.
      2) Otherwise seed with embedded sample rows.

    CSV format must include headers: frame_time_seconds,label,x1,y1,x2,y2,confidence

    Returns:
        Tuple[List[dict], datetime]: A tuple of (detections list, last_updated time in UTC).

    Detection record shape:
      {
        "frame_time_seconds": float,
        "label": str,
        "x1": float, "y1": float, "x2": float, "y2": float,
        "confidence": float
      }
    """
    csv_path = os.getenv("DETECTIONS_CSV")
    if csv_path and os.path.exists(csv_path):
        detections = _load_from_csv(csv_path)
    else:
        detections = _load_from_embedded_sample()
    # Sort by time ascending for deterministic order; route may re-sort on demand
    detections.sort(key=lambda d: float(d.get("frame_time_seconds", 0.0)))
    return detections, datetime.now(timezone.utc)


def _load_from_csv(path: str) -> List[Dict[str, Any]]:
    """Load detections from CSV ensuring correct types."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame_time_seconds", "label", "x1", "y1", "x2", "y2", "confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
        for r in reader:
            try:
                # Coerce to required types: time->float, coords->float, conf->float, label->str
                frame_time = float(r["frame_time_seconds"])
                label = str(r["label"])
                x1 = float(r["x1"])
                y1 = float(r["y1"])
                x2 = float(r["x2"])
                y2 = float(r["y2"])
                conf = float(r["confidence"])
                rows.append(
                    {
                        "frame_time_seconds": frame_time,
                        "label": label,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": conf,
                    }
                )
            except Exception:
                # Skip bad rows; production could log these
                continue
    return rows


def _load_from_embedded_sample() -> List[Dict[str, Any]]:
    """Embedded seed dataset matching expected columns and types.

    Types:
      - frame_time_seconds: float (seconds)
      - label: str
      - x1, y1, x2, y2: float (from provided int coordinates)
      - confidence: float
    """
    # Static seed rows. Coordinates provided as ints are stored as floats for consistency.
    seed: List[Dict[str, Any]] = [
        {"frame_time_seconds": float(1), "label": "bear", "x1": float(10), "y1": float(20), "x2": float(110), "y2": float(220), "confidence": 0.92},
        {"frame_time_seconds": float(2), "label": "bear", "x1": float(15), "y1": float(25), "x2": float(120), "y2": float(230), "confidence": 0.88},
        {"frame_time_seconds": float(3), "label": "deer", "x1": float(50), "y1": float(60), "x2": float(180), "y2": float(260), "confidence": 0.81},
        {"frame_time_seconds": float(4), "label": "bear", "x1": float(12), "y1": float(22), "x2": float(115), "y2": float(225), "confidence": 0.61},
        {"frame_time_seconds": float(5), "label": "fox",  "x1": float(80), "y1": float(90), "x2": float(150), "y2": float(200), "confidence": 0.73},
    ]
    return seed


# PUBLIC_INTERFACE
def query_detections(
    detections: List[Dict[str, Any]],
    label: Optional[str] = None,
    min_confidence: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Filter detections list by label, confidence and time range.

    Args:
        detections: Source detections list.
        label: If provided, only rows with this label are returned (case-sensitive match).
        min_confidence: If provided, only rows with confidence >= min_confidence are returned.
        start_time: If provided, only rows with frame_time_seconds >= start_time are returned.
        end_time: If provided, only rows with frame_time_seconds <= end_time are returned.

    Returns:
        A filtered list of detections.
    """
    out: List[Dict[str, Any]] = []
    for d in detections:
        if label is not None and d.get("label") != label:
            continue
        if min_confidence is not None and float(d.get("confidence", 0.0)) < float(min_confidence):
            continue
        t = float(d.get("frame_time_seconds", 0.0))
        if start_time is not None and t < float(start_time):
            continue
        if end_time is not None and t > float(end_time):
            continue
        out.append(d)
    return out
