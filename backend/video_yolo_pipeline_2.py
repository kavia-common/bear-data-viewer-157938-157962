#!/usr/bin/env python3
"""
Video YOLO Pipeline v2

PUBLIC_INTERFACE
Usage:
    python video_yolo_pipeline_2.py --input <path|s3://bucket/key> --model <path_or_name> --output-bucket <bucket> [--output-prefix <prefix>] [--csv <file>] [--min-conf 0.25]

Description:
- Loads environment variables from a .env file located in the same directory as this script.
- Accepts an input video path (local) or S3 URI s3://bucket/key (key may contain spaces).
- If S3 input, downloads to a temporary local file using boto3.
- Runs YOLO inference on frames sampled every ~4 seconds.
- Draws detections, saves annotated frames locally, uploads to S3, and constructs virtual-hosted style S3 URLs.
- Writes detections to CSV and optionally inserts into MySQL RDS (creating DB and table if needed).
- Graceful handling: skips DB operations if DB_HOST is missing; logs warnings and continues.

Environment Variables (loaded from backend/.env):
- DB_HOST (required for DB), DB_PORT (default 3306), DB_USER, DB_PASSWORD, DB_NAME (must be 'bear_stats')
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION (default 'us-east-2')

Table schema (bear_stats.detections):
- id BIGINT AUTO_INCREMENT PRIMARY KEY
- bear_id VARCHAR(64) NULL
- pose VARCHAR(64) NULL
- ts_seconds DOUBLE NOT NULL
- frame_index BIGINT NOT NULL
- s3_url TEXT NOT NULL
- confidence DOUBLE NULL
- class_id INT NULL
- class_name VARCHAR(64) NULL
- created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""
import argparse
import csv
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# Attempt to import required packages, provide helpful errors
try:
    from dotenv import dotenv_values
except Exception:
    print("Missing dependency 'python-dotenv'. Please install via: pip install python-dotenv", file=sys.stderr)
    raise

try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    print("Missing dependency 'boto3'. Please install via: pip install boto3", file=sys.stderr)
    raise

try:
    import cv2
except Exception:
    print("Missing dependency 'opencv-python'. Please install via: pip install opencv-python", file=sys.stderr)
    raise

# YOLO from ultralytics
try:
    from ultralytics import YOLO
except Exception:
    print("Missing dependency 'ultralytics'. Please install via: pip install ultralytics", file=sys.stderr)
    raise

# Prefer mysql-connector-python, fallback to pymysql
_db_driver = None
try:
    import mysql.connector  # type: ignore
    _db_driver = "mysql-connector"
except Exception:
    try:
        import pymysql  # type: ignore
        _db_driver = "pymysql"
    except Exception:
        _db_driver = None

VHS_S3_URL = "https://{bucket}.s3.{region}.amazonaws.com/{key}"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("video_yolo_pipeline_2")


@dataclass
class Config:
    db_host: Optional[str]
    db_port: int
    db_user: Optional[str]
    db_password: Optional[str]
    db_name: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_region: str


def load_env_from_local() -> Config:
    """
    Load environment variables from a .env file in the same directory as this script.
    """
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / ".env"
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}. Proceeding with current environment variables.")
        values = {}
    else:
        values = dotenv_values(str(env_path))

    def getv(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key, values.get(key, default)) if values else os.environ.get(key, default)

    cfg = Config(
        db_host=getv("DB_HOST", None),
        db_port=int(getv("DB_PORT", "3306")),
        db_user=getv("DB_USER", None),
        db_password=getv("DB_PASSWORD", None),
        db_name=getv("DB_NAME", None),
        aws_access_key_id=getv("AWS_ACCESS_KEY_ID", None),
        aws_secret_access_key=getv("AWS_SECRET_ACCESS_KEY", None),
        aws_region=getv("AWS_REGION", "us-east-2"),
    )
    # Validate DB_NAME if DB is configured
    if cfg.db_host and cfg.db_name and cfg.db_name != "bear_stats":
        logger.warning("DB_NAME must be 'bear_stats'. Overriding provided DB_NAME with 'bear_stats'.")
        cfg.db_name = "bear_stats"
    return cfg


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Parse s3://bucket/key with spaces allowed in key without URL encoding.
    """
    if not uri.lower().startswith("s3://"):
        raise ValueError("Not an S3 URI")
    stripped = uri[5:]  # remove s3://
    parts = stripped.split("/", 1)
    if len(parts) != 2 or not parts[0]:
        raise ValueError("Invalid S3 URI. Expected s3://bucket/key")
    bucket, key = parts[0], parts[1]
    return bucket, key


def make_s3_client(cfg: Config):
    """
    Create a boto3 S3 client using env or explicit credentials.
    """
    sess_kwargs = {}
    if cfg.aws_access_key_id and cfg.aws_secret_access_key:
        sess_kwargs["aws_access_key_id"] = cfg.aws_access_key_id
        sess_kwargs["aws_secret_access_key"] = cfg.aws_secret_access_key
    if cfg.aws_region:
        sess_kwargs["region_name"] = cfg.aws_region
    session = boto3.session.Session(**sess_kwargs) if sess_kwargs else boto3.session.Session(region_name=cfg.aws_region)
    return session.client("s3")


def download_s3_to_temp(s3_client, bucket: str, key: str) -> str:
    """
    Download an S3 object to a secure temporary file. Returns the local file path.
    """
    suffix = Path(key).suffix or ".mp4"
    fd, tmp_path = tempfile.mkstemp(prefix="input_video_", suffix=suffix)
    os.close(fd)
    try:
        logger.info(f"Downloading from s3://{bucket}/{key} to {tmp_path}")
        s3_client.download_file(bucket, key, tmp_path)
        return tmp_path
    except ClientError as e:
        logger.error(f"Failed to download S3 object: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading S3 object: {e}")
        raise


def upload_file_to_s3(s3_client, local_path: str, bucket: str, key: str, region: str) -> str:
    """
    Upload local file to S3 and return the constructed virtual-hosted style URL.
    """
    try:
        s3_client.upload_file(local_path, bucket, key)
        url = VHS_S3_URL.format(bucket=bucket, region=region, key=key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{key}")
        return url
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading to S3: {e}")
        raise


def init_db(cfg: Config):
    """
    Initialize DB connection and ensure database and table exist.
    Returns connection object or None if DB not configured.
    """
    if not cfg.db_host:
        logger.warning("DB_HOST not provided. Skipping DB operations.")
        return None

    if not _db_driver:
        logger.warning("No MySQL driver found (install mysql-connector-python or pymysql). Skipping DB operations.")
        return None

    # Connect without DB to create database if missing
    try:
        if _db_driver == "mysql-connector":
            server_conn = mysql.connector.connect(
                host=cfg.db_host,
                port=cfg.db_port,
                user=cfg.db_user,
                password=cfg.db_password,
            )
            server_conn.autocommit = True
            cur = server_conn.cursor()
            cur.execute("CREATE DATABASE IF NOT EXISTS bear_stats")
            cur.close()
            server_conn.close()

            conn = mysql.connector.connect(
                host=cfg.db_host,
                port=cfg.db_port,
                user=cfg.db_user,
                password=cfg.db_password,
                database="bear_stats",
            )
        else:
            # pymysql
            server_conn = pymysql.connect(
                host=cfg.db_host,
                port=cfg.db_port,
                user=cfg.db_user,
                password=cfg.db_password,
                autocommit=True,
            )
            with server_conn.cursor() as cur:
                cur.execute("CREATE DATABASE IF NOT EXISTS bear_stats")
            server_conn.close()

            conn = pymysql.connect(
                host=cfg.db_host,
                port=cfg.db_port,
                user=cfg.db_user,
                password=cfg.db_password,
                database="bear_stats",
                autocommit=True,
            )
        # Ensure table exists
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS detections (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            bear_id VARCHAR(64) NULL,
            pose VARCHAR(64) NULL,
            ts_seconds DOUBLE NOT NULL,
            frame_index BIGINT NOT NULL,
            s3_url TEXT NOT NULL,
            confidence DOUBLE NULL,
            class_id INT NULL,
            class_name VARCHAR(64) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
        if _db_driver == "mysql-connector":
            conn.commit()
        logger.info("Database and table ensured (bear_stats.detections).")
        return conn
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return None


def insert_detection_row(conn, row: dict):
    """
    Insert a detection row into the database.
    row keys: bear_id, pose, ts_seconds, frame_index, s3_url, confidence, class_id, class_name
    """
    if conn is None:
        return
    insert_sql = """
        INSERT INTO detections (bear_id, pose, ts_seconds, frame_index, s3_url, confidence, class_id, class_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    vals = (
        row.get("bear_id"),
        row.get("pose"),
        row.get("ts_seconds"),
        row.get("frame_index"),
        row.get("s3_url"),
        row.get("confidence"),
        row.get("class_id"),
        row.get("class_name"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(insert_sql, vals)
        if _db_driver == "mysql-connector":
            conn.commit()
        logger.info(f"DB insert: frame {row.get('frame_index')} ts {row.get('ts_seconds'):.2f}s -> OK")
    except Exception as e:
        logger.error(f"Failed to insert row into DB: {e}")


def ensure_csv_with_headers(csv_path: Path):
    """
    Ensure CSV file exists with headers.
    """
    if not csv_path.exists():
        headers = ["bear_id", "pose", "timestamp", "frame_index", "s3_url", "confidence", "class_id", "class_name"]
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        logger.info(f"Created CSV with headers at {csv_path}")


def append_csv_row(csv_path: Path, row: dict):
    """
    Append a detection row to the CSV.
    """
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row.get("bear_id"),
            row.get("pose"),
            row.get("ts_seconds"),
            row.get("frame_index"),
            row.get("s3_url"),
            row.get("confidence"),
            row.get("class_id"),
            row.get("class_name"),
        ])


def build_output_key(prefix: Optional[str], filename: str) -> str:
    if prefix:
        p = prefix.strip("/")
        return f"{p}/{filename}"
    return filename


def draw_annotations(frame, boxes, class_names, confidences, color=(0, 255, 0)):
    """
    Draw bounding boxes and labels on the frame.
    """
    for (x1, y1, x2, y2), cls_name, conf in zip(boxes, class_names, confidences):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(frame, label, (p1[0], max(p1[1] - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def get_frame_interval(cap) -> int:
    """
    Compute frame interval for ~4-second sampling.
    Fallback to 120 frames if FPS invalid (approx 30fps * 4s).
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        logger.warning("FPS not readable; falling back to fixed frame interval of 120.")
        return 120
    interval = max(int(round(fps * 4)), 1)
    return interval


def iter_sampled_frames(cap) -> Iterator[Tuple[int, float, any]]:
    """
    Yield tuples of (frame_index, timestamp_seconds, frame_image) for each sampled frame.
    """
    frame_interval = get_frame_interval(cap)
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_interval == 1 or (frame_idx % frame_interval == 0):
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if cap.get(cv2.CAP_PROP_POS_MSEC) else (frame_idx / max(cap.get(cv2.CAP_PROP_FPS), 1.0))
            yield frame_idx, ts, frame
        frame_idx += 1
        # Optionally log progress
        if total and frame_idx % max(int(total / 10), 1) == 0:
            logger.info(f"Progress: {frame_idx}/{total} frames")
    logger.info(f"Finished reading frames. Total processed frames: {frame_idx}")


def filter_bear_detections(result) -> Tuple[List[Tuple[float, float, float, float]], List[str], List[float], List[int]]:
    """
    From an ultralytics result, filter detections for class 'bear' if present in names; otherwise return all with class names.
    Returns lists: boxes[x1,y1,x2,y2], class_names, confidences, class_ids
    """
    boxes_xyxy = []
    class_names = []
    confidences = []
    class_ids = []

    names = result.names if hasattr(result, "names") else {}
    # result.boxes.xyxy, result.boxes.conf, result.boxes.cls
    try:
        if result.boxes is None:
            return boxes_xyxy, class_names, confidences, class_ids
        xyxy = result.boxes.xyxy.cpu().numpy().tolist()
        confs = result.boxes.conf.cpu().numpy().tolist()
        clss = result.boxes.cls.cpu().numpy().tolist()
    except Exception:
        return boxes_xyxy, class_names, confidences, class_ids

    # Determine 'bear' class id if exists in names
    bear_class_ids = []
    if isinstance(names, dict):
        for cid, cname in names.items():
            if str(cname).lower() == "bear":
                bear_class_ids.append(int(cid))

    for (b, conf, cidf) in zip(xyxy, confs, clss):
        cid = int(cidf)
        cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
        if bear_class_ids:
            if cid in bear_class_ids:
                boxes_xyxy.append(tuple(b))
                class_names.append(cname)
                confidences.append(float(conf))
                class_ids.append(cid)
        else:
            # If bear mapping unknown, include all; consumer can filter by class_name
            boxes_xyxy.append(tuple(b))
            class_names.append(cname)
            confidences.append(float(conf))
            class_ids.append(cid)
    return boxes_xyxy, class_names, confidences, class_ids


def run_pipeline(
    input_path_or_s3: str,
    model_path: str,
    output_bucket: str,
    output_prefix: Optional[str],
    csv_file: Path,
    min_conf: float,
):
    """
    Execute the end-to-end pipeline.
    """
    cfg = load_env_from_local()

    # Prepare S3 client
    s3_client = make_s3_client(cfg)

    # Prepare DB if configured
    conn = init_db(cfg)

    # Resolve input
    local_video_path = None
    tmp_to_cleanup = None
    if input_path_or_s3.lower().startswith("s3://"):
        bucket, key = parse_s3_uri(input_path_or_s3)
        local_video_path = download_s3_to_temp(s3_client, bucket, key)
        tmp_to_cleanup = local_video_path
    else:
        local_video_path = input_path_or_s3
        if not Path(local_video_path).exists():
            logger.error(f"Input file not found: {local_video_path}")
            return

    # Load model
    logger.info(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # Output directories
    out_dir = Path("./output_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    ensure_csv_with_headers(csv_file)

    # Capture video
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {local_video_path}")
        if tmp_to_cleanup:
            try:
                os.remove(tmp_to_cleanup)
            except Exception:
                pass
        return

    total_detections = 0
    uploaded_count = 0
    inserted_count = 0

    try:
        for frame_index, ts_seconds, frame in iter_sampled_frames(cap):
            # Run detection for this frame
            results = model.predict(source=frame, conf=min_conf, verbose=False)
            if not results:
                continue
            # We consider first result (ultralytics returns list)
            result = results[0]

            boxes, class_names, confidences, class_ids = filter_bear_detections(result)
            if not boxes:
                continue  # no bear detections

            # Draw and save annotated frame
            annotated = frame.copy()
            annotated = draw_annotations(annotated, boxes, class_names, confidences)
            # Filename with timestamp and frame index
            ts_str = f"{ts_seconds:.2f}".replace(".", "_")
            base_name = f"frame_{frame_index}_ts_{ts_str}.jpg"
            local_out_path = out_dir / base_name
            cv2.imwrite(str(local_out_path), annotated)

            # Upload to S3
            key = build_output_key(output_prefix, base_name)
            url = upload_file_to_s3(s3_client, str(local_out_path), output_bucket, key, cfg.aws_region)
            uploaded_count += 1

            # Insert one row per detection and CSV entry per detection
            for b, cname, conf, cid in zip(boxes, class_names, confidences, class_ids):
                # Determine bear_id, pose heuristically (unknown by default)
                row = {
                    "bear_id": "unknown",
                    "pose": "unknown",
                    "ts_seconds": float(ts_seconds),
                    "frame_index": int(frame_index),
                    "s3_url": url,
                    "confidence": float(conf),
                    "class_id": int(cid),
                    "class_name": str(cname),
                }
                append_csv_row(csv_file, row)
                if conn is not None:
                    insert_detection_row(conn, row)
                    inserted_count += 1
                total_detections += 1

            logger.info(f"Detections on frame {frame_index}: {len(boxes)} | Uploaded: {key}")

    finally:
        cap.release()
        if tmp_to_cleanup:
            try:
                os.remove(tmp_to_cleanup)
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    logger.info("=== Summary ===")
    logger.info(f"Total bear detections: {total_detections}")
    logger.info(f"Annotated frames uploaded: {uploaded_count}")
    logger.info(f"DB rows inserted: {inserted_count}")
    logger.info(f"CSV file: {csv_file}")


# PUBLIC_INTERFACE
def main():
    """CLI entrypoint for running the video YOLO pipeline with S3 and RDS integrations."""
    parser = argparse.ArgumentParser(description="Video YOLO Pipeline v2 with S3 and MySQL RDS integration.")
    parser.add_argument("--input", required=True, help="Input video path or s3://bucket/key (spaces allowed in key).")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name (default: yolov8n.pt).")
    parser.add_argument("--output-bucket", required=True, help="S3 bucket to upload annotated frames.")
    parser.add_argument("--output-prefix", default=None, help="Optional S3 key prefix for uploads.")
    parser.add_argument("--csv", default="detections.csv", help="Output CSV path (default: detections.csv).")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Minimum confidence threshold (default: 0.25).")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    try:
        run_pipeline(
            input_path_or_s3=args.input,
            model_path=args.model,
            output_bucket=args.output_bucket,
            output_prefix=args.output_prefix,
            csv_file=csv_path,
            min_conf=args.min_conf,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
