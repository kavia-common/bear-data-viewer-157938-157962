"""Video processing operations for the YOLO pipeline."""

import sys
import tempfile
import os
from pathlib import Path
from typing import Tuple, List, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[WARN] OpenCV (cv2) not available", file=sys.stderr)
    cv2 = None
    CV2_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("[WARN] requests not available", file=sys.stderr)
    requests = None
    REQUESTS_AVAILABLE = False

def open_video_from_source(url: str) -> Tuple[Optional[any], Optional[Path]]:
    """
    Try to open the video directly via URL with cv2.VideoCapture.
    If that fails, attempt to download to a secure temporary file and open from disk.
    Returns a tuple (cap, temp_path)
    """
    if not CV2_AVAILABLE:
        print("[ERROR] OpenCV (cv2) is required but not available.", file=sys.stderr)
        return None, None

    # First attempt: direct URL open
    print(f"[INFO] Attempting to open video via URL: {url}")
    cap = cv2.VideoCapture(url)
    if cap is not None and cap.isOpened():
        return cap, None

    # Fallback: download to temp file and open
    print("[WARN] Direct URL open failed. Falling back to streaming download...")
    if not REQUESTS_AVAILABLE:
        print("[ERROR] requests library is not available; cannot download the video.", file=sys.stderr)
        return None, None

    temp_file = None
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            fd, temp_path = tempfile.mkstemp(prefix="video_url_", suffix=".mp4")
            os.close(fd)
            temp_file = Path(temp_path)
            print(f"[INFO] Downloading video to temporary file: {temp_file}")
            with open(temp_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        cap = cv2.VideoCapture(str(temp_file))
        if cap is not None and cap.isOpened():
            print("[INFO] Opened video from downloaded temp file.")
            return cap, temp_file
        else:
            print("[ERROR] Failed to open video from downloaded file.", file=sys.stderr)
            if temp_file and temp_file.exists():
                temp_file.unlink()
            return None, None
    except Exception as e:
        print(f"[ERROR] Failed during URL fallback: {e}", file=sys.stderr)
        if temp_file and temp_file.exists():
            temp_file.unlink()
        return None, None

def get_video_meta(cap) -> Tuple[float, float]:
    """Get fps and duration (in seconds)."""
    fps = cap.get(cv2.CAP_PROP_FPS) if CV2_AVAILABLE else 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if CV2_AVAILABLE else 0.0
    duration_sec = (frame_count / fps) if (fps and fps > 0) else 0.0
    return float(fps or 0.0), float(duration_sec or 0.0)

def compute_sample_times(duration_sec: float, interval_sec: float) -> List[float]:
    """Compute timestamps (seconds) to sample."""
    times: List[float] = []
    if duration_sec <= 0:
        return [0.0]
    t = 0.0
    while t <= duration_sec + 1e-6:
        times.append(round(t, 3))
        t += interval_sec
    if times and times[-1] > duration_sec:
        times[-1] = duration_sec
    return times

def seek_and_read_frame(cap, time_sec: float):
    """Seek to a given time position and read a frame."""
    if not CV2_AVAILABLE:
        return False, None
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    return ok, frame

def save_frame_temp_jpg(frame, quality: int = 90) -> Optional[Path]:
    """Save an OpenCV frame (BGR) to a temporary JPEG and return its path."""
    try:
        fd, temp_path = tempfile.mkstemp(prefix="frame_", suffix=".jpg")
        os.close(fd)
        p = Path(temp_path)
        ok = cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            p.unlink()
            return None
        return p
    except Exception as e:
        print(f"[WARN] Failed to save frame to temp JPG: {e}", file=sys.stderr)
        return None