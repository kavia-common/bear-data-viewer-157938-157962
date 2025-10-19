#!/usr/bin/env python3
"""
video_yolo_pipeline_1.py

Purpose:
- Open a video from a hardcoded S3 HTTPS URL.
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
- requests (for URL fallback download; should be available, else add to requirements)

Quick setup notes:
- Ensure Python environment has dependencies installed (pip install -r backend/requirements.txt).
- On first run, ultralytics will download 'yolov8m.pt' if not present and internet is available.
- This script uses OpenCV's CAP_PROP_POS_MSEC to seek by timestamp; behavior can vary by codec/container.
  If seeking lands slightly off the exact timestamp, we still use the resulting frame.

Exit Codes:
- 0 on success (even if zero detections)
- 1 on missing dependencies or unexpected runtime error
- 2 if video URL not reachable or cannot be opened
- 3 if model cannot be loaded

Usage:
- python video_yolo_pipeline_1.py
"""

import sys
import csv
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import contextmanager


try:
    from dotenv import load_dotenv
    # Load .env file from the same directory as this script
    env_path = Path(__file__).resolve().parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"[INFO] Loaded environment variables from: {env_path}")
except ImportError:
    print("[WARN] python-dotenv not installed. Install with: pip install python-dotenv")
    print("[WARN] Falling back to system environment variables")

# Try to import boto3 for S3 uploads; guard if missing
try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
    BOTO3_AVAILABLE = True
except Exception as e:
    print(f"[WARN] boto3 import failed (S3 upload disabled): {e}", file=sys.stderr)
    boto3 = None  # type: ignore
    BOTO3_AVAILABLE = False

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

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception as e:
    print(f"[WARN] requests import failed (URL fallback disabled): {e}", file=sys.stderr)
    REQUESTS_AVAILABLE = False

try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except Exception as e:
    print(f"[WARN] mysql-connector-python import failed (DB disabled): {e}", file=sys.stderr)
    mysql_connector = None
    MySQLError = Exception
    MYSQL_AVAILABLE = False

def _get_db_env() -> Tuple[str, int, str, str, str]:
    """Load DB connection settings from environment."""
    host = os.environ.get("DB_HOST", "")
    port_raw = os.environ.get("DB_PORT", "3306")
    user = os.environ.get("DB_USER", "")
    password = os.environ.get("DB_PASSWORD", "")
    db_name = os.environ.get("DB_NAME", "")

    try:
        port = int(port_raw)
    except ValueError:
        print(f"[DB] Invalid DB_PORT value '{port_raw}', defaulting to 3306")
        port = 3306

    return host, port, user, password, db_name

@contextmanager
def _db_connection(database: Optional[str] = None):
    """Context manager that yields a MySQL/MariaDB connection."""
    if not MYSQL_AVAILABLE:
        raise RuntimeError("mysql-connector-python is required but not installed.")

    host, port, user, password, db_name_env = _get_db_env()
    db_to_use = database if database is not None else None

    conn = None
    try:
        conn = mysql_connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_to_use,
            autocommit=True,
        )
        yield conn
    except MySQLError as e:
        print(f"[DB] Connection error: {e}")
        raise
    finally:
        try:
            if conn is not None and conn.is_connected():
                conn.close()
        except Exception:
            pass

def setup_database_and_table() -> None:
    """Create the database and detections table if they don't exist."""
    if not MYSQL_AVAILABLE:
        print("[DB-SETUP] MySQL connector not available, skipping database setup.")
        return

    host, port, user, password, db_name = _get_db_env()
    if not db_name:
        print("[WARN] DB_NAME not set, skipping database setup.")
        return

    # Create database
    try:
        with _db_connection(database=None) as conn:
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            cursor.close()
    except Exception as e:
        print(f"[DB-SETUP] Failed creating database: {e}")
        return

    # Create table
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS detections (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            frame_time_seconds INT,
            label VARCHAR(255),
            x1 INT,
            y1 INT,
            x2 INT,
            y2 INT,
            obj_label_confidence FLOAT,
            pose VARCHAR(255),
            S3_img_link VARCHAR(1024),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    try:
        with _db_connection(database=db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.close()
    except Exception as e:
        print(f"[DB-SETUP] Failed creating table: {e}")

def insert_detection_to_db(time_s: float, label: str, x1: int, y1: int, x2: int, y2: int, 
                           conf: float, pose: str, s3_url: str) -> bool:
    """Insert a single detection record into the database."""
    if not MYSQL_AVAILABLE:
        return False

    host, port, user, password, db_name = _get_db_env()
    if not db_name:
        return False

    insert_sql = """
        INSERT INTO detections
            (frame_time_seconds, label, x1, y1, x2, y2, obj_label_confidence, pose, S3_img_link)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        with _db_connection(database=db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(insert_sql, (
                int(time_s), label, x1, y1, x2, y2, conf, pose, s3_url
            ))
            cursor.close()
            return True
    except Exception as e:
        print(f"[DB] Failed to insert detection: {e}")
        return False

from contextlib import contextmanager

SAMPLE_INTERVAL_SECONDS = 10.0
MODEL_WEIGHTS = "yolov8m.pt"
OUTPUT_FILENAME = "identified_bear.dat"

# Acceptance criteria: Use hardcoded S3 URL
HARDCODED_VIDEO_URL = "https://humanlabelimg-poc.s3.us-east-2.amazonaws.com/2.mp4"


def _here() -> Path:
    """Return the directory of this script file."""
    return Path(__file__).resolve().parent


def _round2(x: float) -> float:
    """Round a float to 2 decimal places."""
    return float(f"{x:.2f}")


def _round4(x: float) -> float:
    """Round a float to 4 decimal places."""
    return float(f"{x:.4f}")


def _open_video_from_source(url: str):
    """
    Try to open the video directly via URL with cv2.VideoCapture.
    If that fails, attempt to download to a secure temporary file and open from disk.
    Returns a tuple (cap, temp_path) where:
      - cap is the opened cv2.VideoCapture or None
      - temp_path is a Path to a temp file that should be deleted by the caller (or None)
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
            os.close(fd)  # Close file descriptor; we'll write using open()
            temp_file = Path(temp_path)
            print(f"[INFO] Downloading video to temporary file: {temp_file}")
            with open(temp_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
        # Try opening the downloaded file
        cap = cv2.VideoCapture(str(temp_file))
        if cap is not None and cap.isOpened():
            print("[INFO] Opened video from downloaded temp file.")
            return cap, temp_file
        else:
            print("[ERROR] Failed to open video from downloaded file.", file=sys.stderr)
            # Cleanup if open failed
            try:
                if temp_file and temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch video from URL: {e}", file=sys.stderr)
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        return None, None
    except Exception as e:
        print(f"[ERROR] Unexpected error during URL fallback: {e}", file=sys.stderr)
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        return None, None


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
        # Added 'pose' (blank for now) and 's3_image_link' for uploaded frame URL
        writer.writerow(["frame_time_seconds", "label", "x1", "y1", "x2", "y2", "confidence", "pose", "s3_image_link"])


def _append_detection(out_path: Path, time_s: float, label: str, x1: int, y1: int, x2: int, y2: int, conf: float, pose: str, s3_url: str):
    """Append one detection row to the CSV, including pose and s3_image_link."""
    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([_round2(time_s), label, int(x1), int(y1), int(x2), int(y2), _round4(conf), pose, s3_url])


def _get_boto3_s3_client():
    """Create a boto3 S3 client using environment variables, or return None if unavailable/misconfigured."""
    if not BOTO3_AVAILABLE:
        return None
    # Credentials/config via env
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL")  # Optional

    # If creds are missing, still try default chain; boto3 can pick up IAM or env
    try:
        session_kwargs = {}
        if aws_region:
            session_kwargs["region_name"] = aws_region
        session = boto3.session.Session(**session_kwargs)  # type: ignore

        client_kwargs = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        s3_client = session.client("s3", **client_kwargs)  # type: ignore
        return s3_client
    except Exception as e:
        print(f"[WARN] Failed to create boto3 S3 client: {e}", file=sys.stderr)
        return None


def _guess_region_from_bucket(bucket: str) -> Optional[str]:
    """Optionally query bucket region to build URL; non-fatal on failure."""
    client = _get_boto3_s3_client()
    if not client:
        return None
    try:
        resp = client.get_bucket_location(Bucket=bucket)  # type: ignore
        # For us-east-1, response can be None or 'us-east-1' depending on API
        loc = resp.get("LocationConstraint")
        return loc or "us-east-1"
    except Exception:
        return None


def _build_http_url(bucket: str, key: str, region_hint: Optional[str] = None, endpoint_url: Optional[str] = None) -> str:
    """
    Build an HTTP URL to the S3 object. If endpoint_url is provided (custom S3-compatible),
    use it; otherwise construct standard AWS S3 URL.
    """
    if endpoint_url:
        # Normalize endpoint - should be like https://s3.custom.local
        base = endpoint_url.rstrip("/")
        return f"{base}/{bucket}/{key}"
    # Standard AWS S3 URL format. Prefer virtual-hosted-style.
    region = region_hint or "us-east-1"
    if region == "us-east-1":
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def upload_to_s3(file_path: Path, bucket: str, key: str) -> str:
    """
    Upload file to S3 and return a public or presigned URL.
    - Makes object public-read if possible; otherwise generate a 7-day presigned URL.
    - Returns '' on any failure.
    """
    client = _get_boto3_s3_client()
    if not client:
        print("[WARN] boto3 not available or client not configured; skipping upload.", file=sys.stderr)
        return ""

    # Attempt upload with public-read ACL
    try:
        extra_args = {"ACL": "public-read"}
        client.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)  # type: ignore
    except ClientError as ce:
        # If AccessDenied for ACL or public blocks, attempt upload without ACL
        msg = str(ce)
        print(f"[WARN] S3 upload with public-read ACL failed: {msg}", file=sys.stderr)
        try:
            client.upload_file(str(file_path), bucket, key)  # type: ignore
        except Exception as e:
            print(f"[WARN] S3 upload failed: {e}", file=sys.stderr)
            return ""
    except (BotoCoreError, Exception) as e:
        print(f"[WARN] S3 upload failed: {e}", file=sys.stderr)
        return ""

    # Try to construct a public URL first
    try:
        endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL")
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or _guess_region_from_bucket(bucket)
        public_url = _build_http_url(bucket, key, region_hint=region, endpoint_url=endpoint_url)
        # We cannot verify public-read here; return constructed URL optimistically.
        return public_url
    except Exception as e:
        print(f"[WARN] Failed to build public URL: {e}", file=sys.stderr)

    # Fallback: presigned URL valid for 7 days (604800 seconds)
    try:
        presigned = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": "humanlabelimg-poc", "Key": key},
            ExpiresIn=604800,  # 7 days
        )  # type: ignore
        return presigned
    except Exception as e:
        print(f"[WARN] Failed to generate presigned URL: {e}", file=sys.stderr)
        return ""

def _save_frame_temp_jpg(frame, quality: int = 90) -> Optional[Path]:
    """Save an OpenCV frame (BGR) to a temporary JPEG and return its path."""
    try:
        fd, temp_path = tempfile.mkstemp(prefix="frame_", suffix=".jpg")
        os.close(fd)
        p = Path(temp_path)
        print("Temp Path",temp_path)
        # Encode as JPEG
        import cv2 as _cv2  # local import to ensure CV2_AVAILABLE not required here
        ok = _cv2.imwrite(str(p), frame, [_cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            try:
                p.unlink()
            except Exception:
                pass
            return None
        return p
    except Exception as e:
        print(f"[WARN] Failed to save frame to temp JPG: {e}", file=sys.stderr)
        return None


def _process():
    """Main processing routine with summary printing and exit codes."""
    base_dir = _here()
    out_path = base_dir / OUTPUT_FILENAME

    try:
        setup_database_and_table()
    except Exception as e:
        print(f"[WARN] Database setup failed: {e}. Continuing without DB.", file=sys.stderr)

    # Load model
    model = _load_model()
    if model is None:
        return 3, 0, 0, out_path

    # Open video from URL (with fallback)
    cap, temp_path = _open_video_from_source(HARDCODED_VIDEO_URL)
    if cap is None:
        print(f"[ERROR] Could not open video from URL: {HARDCODED_VIDEO_URL}", file=sys.stderr)
        return 2, 0, 0, out_path

    # Get metadata and compute samples
    fps, duration_sec = _get_video_meta(cap)
    if duration_sec <= 0:
        # If duration unavailable, we still attempt at least at t=0
        print("[WARN] Could not determine video duration. Sampling only the first frame.", file=sys.stderr)
        sample_times = [0.0]
    else:
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

                # Save the sampled frame once if we will record any detection, to upload to S3.
                uploaded_url_cache: Optional[str] = None

                for i, raw_c in enumerate(conf_list):
                    try:
                        conf = float(raw_c)
                    except Exception:
                        conf = 0.0

                    # Resolve class id and map to label via model names (if available)
                    cls_id = int(cls_list[i]) if i < len(cls_list) else -1
                    label = names.get(cls_id, None)

                    # Safeguard: if mapping unavailable or label not a string, skip
                    if not isinstance(label, str):
                        continue

                    # Normalize label for case-insensitive comparison and whitespace trimming
                    norm_label = label.strip().lower()
                    if norm_label != "bear":
                        # Only write rows labeled as 'bear'
                        continue

                    if i < len(xyxy_list):
                        b = xyxy_list[i]
                        if isinstance(b, (list, tuple)) and len(b) >= 4:
                            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        else:
                            x1 = y1 = x2 = y2 = 0
                    else:
                        x1 = y1 = x2 = y2 = 0

                    # Ensure we have attempted S3 upload once per sampled frame time if a detection is recorded
                    if uploaded_url_cache is None:
                        # Save the frame to a temp JPEG
                        tmp_img = _save_frame_temp_jpg(frame)
                        s3_url = ""
                        if tmp_img is not None:
                            try:
                                # Build S3 key based on timestamp in ms
                                timestamp_ms = int(round(t * 1000.0))
                                object_key = f"{timestamp_ms}.jpg"
                                s3_url = upload_to_s3(tmp_img, "humanlabelimg-poc", object_key)
                            except Exception as e:
                                print(f"[WARN] S3 upload attempt failed: {e}", file=sys.stderr)
                                s3_url = ""
                            finally:
                                try:
                                    tmp_img.unlink()
                                except Exception:
                                    pass
                        else:
                            print("[WARN] Skipping S3 upload due to temp image save failure.", file=sys.stderr)
                            s3_url = ""
                        uploaded_url_cache = s3_url  # cache result (even if empty) for subsequent detections in same frame

                    # Write the detection including pose (blank) and s3_image_link
                    # Write to CSV (existing)
                    _append_detection(out_path, t, label, x1, y1, x2, y2, conf, 
                                    pose="", s3_url=uploaded_url_cache or "")
                    
                    # ADD THIS: Write to database
                    insert_detection_to_db(t, label, x1, y1, x2, y2, conf, 
                                          pose="", s3_url=uploaded_url_cache or "")
                    
                    total_detections_written += 1

            except Exception as e:
                print(f"[WARN] Failed to parse/write detections at ~{t:.2f}s: {e}", file=sys.stderr)
                continue

    finally:
        try:
            cap.release()
        except Exception:
            pass
        # Cleanup temp file if we used fallback
        if temp_path and isinstance(temp_path, Path):
            try:
                if temp_path.exists():
                    temp_path.unlink()
                    print(f"[INFO] Cleaned up temporary file: {temp_path}")
            except Exception as e:
                print(f"[WARN] Failed to remove temp file {temp_path}: {e}", file=sys.stderr)

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
