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


# PUBLIC_INTERFACE
SEED_DETECTIONS: List[Dict[str, Any]] = [
    # Note: The request specified ints for frame_time_seconds and bbox; we store as floats for consistency.
    {"frame_time_seconds": float(0), "label": "sports ball", "x1": float(714), "y1": float(764), "x2": float(820), "y2": float(840), "confidence": 0.3781},
    {"frame_time_seconds": float(0), "label": "bear", "x1": float(1290), "y1": float(430), "x2": float(1447), "y2": float(692), "confidence": 0.343},
    {"frame_time_seconds": float(10), "label": "bear", "x1": float(680), "y1": float(143), "x2": float(1212), "y2": float(1073), "confidence": 0.9395},
    {"frame_time_seconds": float(10), "label": "sports ball", "x1": float(1667), "y1": float(312), "x2": float(1728), "y2": float(350), "confidence": 0.3426},
    {"frame_time_seconds": float(10), "label": "sports ball", "x1": float(1537), "y1": float(265), "x2": float(1621), "y2": float(350), "confidence": 0.3212},
    {"frame_time_seconds": float(10), "label": "person", "x1": float(327), "y1": float(0), "x2": float(558), "y2": float(281), "confidence": 0.3185},
    {"frame_time_seconds": float(10), "label": "person", "x1": float(328), "y1": float(0), "x2": float(561), "y2": float(281), "confidence": 0.3073},
    {"frame_time_seconds": float(10), "label": "sports ball", "x1": float(1538), "y1": float(265), "x2": float(1621), "y2": float(350), "confidence": 0.2954},
    {"frame_time_seconds": float(10), "label": "person", "x1": float(329), "y1": float(0), "x2": float(564), "y2": float(282), "confidence": 0.2952},
    {"frame_time_seconds": float(10), "label": "person", "x1": float(328), "y1": float(1), "x2": float(556), "y2": float(281), "confidence": 0.2865},
    {"frame_time_seconds": float(10), "label": "person", "x1": float(329), "y1": float(0), "x2": float(568), "y2": float(282), "confidence": 0.2771},
    {"frame_time_seconds": float(10), "label": "person", "x1": float(329), "y1": float(1), "x2": float(566), "y2": float(281), "confidence": 0.2714},
    {"frame_time_seconds": float(10), "label": "person", "x1": float(329), "y1": float(0), "x2": float(573), "y2": float(282), "confidence": 0.2638},
    {"frame_time_seconds": float(20), "label": "baseball glove", "x1": float(1526), "y1": float(921), "x2": float(1693), "y2": float(1025), "confidence": 0.4365},
    {"frame_time_seconds": float(20), "label": "sports ball", "x1": float(1365), "y1": float(735), "x2": float(1416), "y2": float(776), "confidence": 0.2613},
    {"frame_time_seconds": float(30), "label": "bear", "x1": float(1005), "y1": float(278), "x2": float(1317), "y2": float(522), "confidence": 0.9163},
    {"frame_time_seconds": float(30), "label": "sports ball", "x1": float(752), "y1": float(561), "x2": float(861), "y2": float(638), "confidence": 0.4693},
    {"frame_time_seconds": float(40), "label": "bear", "x1": float(1154), "y1": float(248), "x2": float(1349), "y2": float(573), "confidence": 0.9312},
    {"frame_time_seconds": float(40), "label": "sports ball", "x1": float(372), "y1": float(449), "x2": float(429), "y2": float(495), "confidence": 0.7051},
    {"frame_time_seconds": float(40), "label": "sports ball", "x1": float(505), "y1": float(469), "x2": float(615), "y2": float(543), "confidence": 0.5613},
    {"frame_time_seconds": float(40), "label": "sports ball", "x1": float(454), "y1": float(514), "x2": float(546), "y2": float(596), "confidence": 0.2767},
    {"frame_time_seconds": float(60), "label": "bear", "x1": float(897), "y1": float(248), "x2": float(1146), "y2": float(534), "confidence": 0.8727},
    {"frame_time_seconds": float(60), "label": "sports ball", "x1": float(1341), "y1": float(403), "x2": float(1444), "y2": float(475), "confidence": 0.425},
    {"frame_time_seconds": float(60), "label": "sports ball", "x1": float(1199), "y1": float(382), "x2": float(1255), "y2": float(426), "confidence": 0.4082},
    {"frame_time_seconds": float(60), "label": "sports ball", "x1": float(1241), "y1": float(401), "x2": float(1287), "y2": float(449), "confidence": 0.3143},
    {"frame_time_seconds": float(70), "label": "bear", "x1": float(852), "y1": float(170), "x2": float(1562), "y2": float(833), "confidence": 0.9376},
    {"frame_time_seconds": float(70), "label": "sports ball", "x1": float(675), "y1": float(215), "x2": float(785), "y2": float(288), "confidence": 0.337},
    {"frame_time_seconds": float(70), "label": "sports ball", "x1": float(361), "y1": float(560), "x2": float(550), "y2": float(682), "confidence": 0.3103},
    {"frame_time_seconds": float(80), "label": "bear", "x1": float(1181), "y1": float(212), "x2": float(1310), "y2": float(461), "confidence": 0.4593},
    {"frame_time_seconds": float(80), "label": "sports ball", "x1": float(587), "y1": float(561), "x2": float(644), "y2": float(638), "confidence": 0.4182},
    {"frame_time_seconds": float(80), "label": "sports ball", "x1": float(550), "y1": float(591), "x2": float(595), "y2": float(637), "confidence": 0.3174},
    {"frame_time_seconds": float(80), "label": "potted plant", "x1": float(1552), "y1": float(8), "x2": float(1918), "y2": float(1042), "confidence": 0.3146},
    {"frame_time_seconds": float(80), "label": "sports ball", "x1": float(619), "y1": float(540), "x2": float(676), "y2": float(588), "confidence": 0.2531},
    {"frame_time_seconds": float(110), "label": "bear", "x1": float(1082), "y1": float(298), "x2": float(1294), "y2": float(704), "confidence": 0.9303},
    {"frame_time_seconds": float(110), "label": "sports ball", "x1": float(190), "y1": float(450), "x2": float(233), "y2": float(496), "confidence": 0.3198},
]

def _load_from_embedded_sample() -> List[Dict[str, Any]]:
    """Embedded seed dataset matching expected columns and types.

    Types:
      - frame_time_seconds: float (seconds)
      - label: str
      - x1, y1, x2, y2: float (from provided int coordinates)
      - confidence: float
    """
    # Use the provided SEED_DETECTIONS
    return list(SEED_DETECTIONS)


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
