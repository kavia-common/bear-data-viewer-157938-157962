#!/usr/bin/env python3
"""
video_yolo_pipeline_1.py

Purpose:
- Open a local video file "1.mp4" located in the same directory as this script.
- Sample one frame every 10 seconds across the video duration.
- Run YOLOv8m object detection (ultralytics) over sampled frames.
- Write detections to "identified_bear.dat" in CSV format with header:
  frame_time_seconds,label,x1,y1,x2,y2,confidence

Columns:
- frame_time_seconds: float (rounded to 2 decimals)
- label: class name predicted by YOLO model
- x1, y1, x2, y2: integer pixel coordinates (top-left and bottom-right)
- confidence: float (rounded to 4 decimals)

Behavior:
- If no objects are detected in a sampled frame, nothing is written for that frame.
- Handles missing video/model gracefully with informative prints and appropriate exit codes.
- Prints summary at the end: total_frames_sampled, total_detections_written, output_path.

Dependencies (already listed in backend/requirements.txt in this project):
- ultralytics (for YOLOv8)
- opencv-python (cv2)
- numpy (optional, not strictly needed here)
- torch, torchvision (pulled as dependencies of ultralytics; used by the model)

Quick setup notes:
- Ensure Python environment has dependencies installed (pip install -r backend/requirements.txt).
- On first run, ultralytics will download 'yolov8m.pt' if not present and internet is available.
- This script uses OpenCV's CAP_PROP_POS_MSEC to seek by timestamp; behavior can vary by codec/container.
  If seeking lands slightly off the exact timestamp, we still use the resulting frame.

Exit Codes:
- 0 on success (even if zero detections)
- 1 on missing dependencies or unexpected runtime error
- 2 if video file not found or cannot be opened
- 3 if model cannot be loaded

Usage:
- python video_yolo_pipeline_1.py
"""

import sys
import csv
from pathlib import Path
from typing import List, Tuple

# Optional imports with graceful error messages
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] OpenCV import failed: {e}", file=sys.stderr)
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] ultralytics import failed: {e}", file=sys.stderr)
    ULTRALYTICS_AVAILABLE = False


SAMPLE_INTERVAL_SECONDS = 10.0
MODEL_WEIGHTS = "yolov8m.pt"
OUTPUT_FILENAME = "identified_bear.dat"
INPUT_VIDEO = "1.mp4"


def _here() -> Path:
    """Return the directory of this script file."""
    return Path(__file__).resolve().parent


def _round2(x: float) -> float:
    """Round a float to 2 decimal places."""
    return float(f"{x:.2f}")


def _round4(x: float) -> float:
    """Round a float to 4 decimal places."""
    return float(f"{x:.4f}")


def _open_video(video_path: Path):
    """Open the video via cv2 with basic validation."""
    if not CV2_AVAILABLE:
        print("[ERROR] OpenCV (cv2) is required but not available.", file=sys.stderr)
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}", file=sys.stderr)
        return None
    return cap


def _get_video_meta(cap) -> Tuple[float, float]:
    """Get fps and duration (in seconds) using CAP_PROP_FPS and CAP_PROP_FRAME_COUNT.

    Returns:
        (fps, duration_sec)
    """
    fps = cap.get(cv2.CAP_PROP_FPS) if CV2_AVAILABLE else 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if CV2_AVAILABLE else 0.0
    duration_sec = (frame_count / fps) if (fps and fps > 0) else 0.0
    return float(fps or 0.0), float(duration_sec or 0.0)


def _compute_sample_times(duration_sec: float, interval_sec: float) -> List[float]:
    """Compute timestamps (seconds) to sample: 0, interval, 2*interval, ..., <= duration."""
    times: List[float] = []
    if duration_sec <= 0:
        return [0.0]
    t = 0.0
    # Guard against floating point accumulation
    while t <= duration_sec + 1e-6:
        times.append(round(t, 3))
        t += interval_sec
    # Ensure last time doesn't exceed duration
    if times and times[-1] > duration_sec:
        times[-1] = duration_sec
    return times


def _seek_and_read_frame(cap, time_sec: float):
    """Seek to a given time position and read a frame.

    Uses CAP_PROP_POS_MSEC for timestamp-based seeking to handle variable frame rates.

    Returns:
        (ok, frame)
    """
    if not CV2_AVAILABLE:
        return False, None
    # OpenCV expects milliseconds
    ok_seek = cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    if not ok_seek:
        # Still attempt to read; some backends set returns False but position might still move
        pass
    ok, frame = cap.read()
    return ok, frame


def _load_model():
    """Load YOLOv8m model weights."""
    if not ULTRALYTICS_AVAILABLE:
        print("[ERROR] ultralytics is required but not available.", file=sys.stderr)
        return None
    try:
        model = YOLO(MODEL_WEIGHTS)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model '{MODEL_WEIGHTS}': {e}", file=sys.stderr)
        print("If the system has no internet, please place yolov8m.pt alongside the script.", file=sys.stderr)
        return None


def _write_csv_header(out_path: Path):
    """Ensure CSV header exists; overwrite file to start fresh."""
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_time_seconds", "label", "x1", "y1", "x2", "y2", "confidence"])


def _append_detection(out_path: Path, time_s: float, label: str, x1: int, y1: int, x2: int, y2: int, conf: float):
    """Append one detection row to the CSV."""
    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([_round2(time_s), label, int(x1), int(y1), int(x2), int(y2), _round4(conf)])


def _process():
    """Main processing routine with summary printing and exit codes."""
    base_dir = _here()
    video_path = base_dir / INPUT_VIDEO
    out_path = base_dir / OUTPUT_FILENAME

    # Validate video path
    if not video_path.exists() or not video_path.is_file():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        return 2, 0, 0, out_path

    # Load model
    model = _load_model()
    if model is None:
        return 3, 0, 0, out_path

    # Open video
    cap = _open_video(video_path)
    if cap is None:
        return 2, 0, 0, out_path

    # Get metadata and compute samples
    fps, duration_sec = _get_video_meta(cap)
    sample_times = _compute_sample_times(duration_sec, SAMPLE_INTERVAL_SECONDS)

    total_frames_sampled = 0
    total_detections_written = 0

    # Prepare output CSV
    _write_csv_header(out_path)

    try:
        for t in sample_times:
            ok, frame = _seek_and_read_frame(cap, t)
            if not ok or frame is None:
                print(f"[WARN] Failed to read frame at ~{t:.2f}s; skipping.", file=sys.stderr)
                continue

            total_frames_sampled += 1

            # Run inference
            try:
                results = model(frame, verbose=False)
            except Exception as e:
                print(f"[WARN] Inference failed at ~{t:.2f}s: {e}", file=sys.stderr)
                continue

            # Extract and write detections
            try:
                if not results:
                    continue
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                names = getattr(r0, "names", {})
                if boxes is None:
                    continue

                confs = getattr(boxes, "conf", None)
                clses = getattr(boxes, "cls", None)
                xyxy = getattr(boxes, "xyxy", None)
                if confs is None or clses is None or xyxy is None:
                    continue

                # Convert to lists
                try:
                    conf_list = confs.squeeze(-1).tolist()
                except Exception:
                    conf_list = confs.tolist() if hasattr(confs, "tolist") else list(confs)

                cls_list = clses.squeeze(-1).tolist() if hasattr(clses, "squeeze") else (
                    clses.tolist() if hasattr(clses, "tolist") else list(clses)
                )
                xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)

                for i, raw_c in enumerate(conf_list):
                    try:
                        conf = float(raw_c)
                    except Exception:
                        conf = 0.0

                    cls_id = int(cls_list[i]) if i < len(cls_list) else -1
                    label = names.get(cls_id, str(cls_id))

                    if i < len(xyxy_list):
                        b = xyxy_list[i]
                        if isinstance(b, (list, tuple)) and len(b) >= 4:
                            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        else:
                            x1 = y1 = x2 = y2 = 0
                    else:
                        x1 = y1 = x2 = y2 = 0

                    _append_detection(out_path, t, label, x1, y1, x2, y2, conf)
                    total_detections_written += 1

            except Exception as e:
                print(f"[WARN] Failed to parse/write detections at ~{t:.2f}s: {e}", file=sys.stderr)
                continue

    finally:
        cap.release()

    print(f"[INFO] total_frames_sampled={total_frames_sampled}")
    print(f"[INFO] total_detections_written={total_detections_written}")
    print(f"[INFO] output_path={out_path}")

    return 0, total_frames_sampled, total_detections_written, out_path


def main():
    """CLI entrypoint with robust error handling."""
    if not CV2_AVAILABLE or not ULTRALYTICS_AVAILABLE:
        print("[ERROR] Missing required dependencies. Please ensure 'opencv-python' and 'ultralytics' are installed.", file=sys.stderr)
        sys.exit(1)

    try:
        code, _, _, _ = _process()
        sys.exit(code)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
