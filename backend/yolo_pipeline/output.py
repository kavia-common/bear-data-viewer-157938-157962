"""Output handling operations for the pipeline."""

import csv
from pathlib import Path

def round2(x: float) -> float:
    """Round a float to 2 decimal places."""
    return float(f"{x:.2f}")

def round4(x: float) -> float:
    """Round a float to 4 decimal places."""
    return float(f"{x:.4f}")

def write_csv_header(out_path: Path):
    """Ensure CSV header exists; overwrite file to start fresh."""
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_time_seconds", "label", "x1", "y1", "x2", "y2", "confidence", 
            "cls_label", "cls_conf", "pose_status", "num_keypoints", "s3_image_link"
        ])

def append_detection(out_path: Path, time_s: float, label: str, x1: int, y1: int, x2: int, y2: int, 
                     conf: float, cls_label: str, cls_conf: float, 
                     pose_status: str, num_keypoints: int, s3_url: str):
    """Append one detection row to the CSV."""
    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            round2(time_s), label, int(x1), int(y1), int(x2), int(y2), round4(conf),
            cls_label or "", round4(cls_conf) if cls_conf else "", 
            pose_status or "", num_keypoints or 0, s3_url
        ])