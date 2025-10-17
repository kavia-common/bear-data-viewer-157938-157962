"""
yolo_io_integration.py
Minimal, pluggable S3 and RDS (MySQL) helpers for your existing YOLO video pipeline.

This module:
- Loads configuration from a .env file in the same directory (if present) using python-dotenv, else falls back to process env
- Provides simple S3 helpers to:
  * parse s3:// URLs
  * download a video from s3:// or pass through a local path
  * upload any local file (image, csv, etc.) to S3 under a bucket/prefix, returning s3:// URL
- Provides minimal MySQL (RDS) helpers to:
  * connect to MySQL (server-level or database-specific)
  * ensure database exists
  * ensure detections table exists
  * insert a detection row and return the inserted id
- Keeps dependencies minimal: boto3 and pymysql. If env is missing, the functions raise informative ValueError exceptions.

PUBLIC INTERFACE SUMMARY
- get_config()
- parse_s3_url(url)
- download_video(source: str, dest_path: str) -> str
- upload_file(local_path: str, bucket: str, prefix: str) -> str
- get_mysql_connection(db: str | None = None)
- ensure_database_exists(db_name: str) -> None
- ensure_table_exists(db_name: str) -> None
- insert_detection(db_name: str, row: dict) -> int

Example integration in your existing pipeline (no changes required to pipeline structure):
------------------------------------------------------------------
from yolo_io_integration import (
    get_config,
    download_video,
    upload_file,
    ensure_database_exists,
    ensure_table_exists,
    insert_detection,
)

import os

cfg = get_config()
# Ensure DB and table (one-time setup at startup)
ensure_database_exists(cfg.get('DB_NAME', 'bear_stats'))
ensure_table_exists(cfg.get('DB_NAME', 'bear_stats'))

# Download input video (works with local path or s3:// URL)
local_video_path = download_video(
    source=input_video,  # could be "s3://my-bucket/path/to/video.mp4" or local path
    dest_path=os.path.join(temp_dir, 'video.mp4')
)

# ... your detection loop ...
# When saving a frame as an image to local disk (e.g., frame_file):
s3_url = upload_file(
    local_path=frame_file,
    bucket=cfg['S3_BUCKET'],
    prefix=cfg.get('S3_PREFIX', 'detections/')
)

insert_id = insert_detection(cfg.get('DB_NAME', 'bear_stats'), {
    'video_source': input_video,
    'frame_index': i,
    'timestamp_sec': ts,
    'class_label': label,
    'confidence': conf,
    'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
    's3_image_url': s3_url,
})
------------------------------------------------------------------

Required environment variables (see backend/.env.example):
- DB_HOST
- DB_PORT (default 3306)
- DB_USER
- DB_PASSWORD
- DB_NAME (default bear_stats)
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (default us-east-2)
- S3_BUCKET
- S3_PREFIX (optional)

Notes:
- Paths with spaces are handled transparently.
- This module does not modify your existing pipeline files; it only provides functions you can import and call.

"""

from __future__ import annotations

import os
import shutil
import logging
import mimetypes
from typing import Tuple, Optional

# Attempt to load dotenv from local .env; if missing, continue with process env
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

# Lazy imports for optional dependencies; we raise informative errors if missing when used
try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    boto3 = None
    BotoCoreError = Exception  # fallbacks for typing
    ClientError = Exception

try:
    import pymysql  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pymysql = None

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load_dotenv_if_available() -> None:
    """
    Load environment variables from a .env file located alongside this module, if python-dotenv is available.
    This is a no-op if python-dotenv is not installed or the file does not exist.
    """
    if load_dotenv is None:
        logger.debug("python-dotenv not installed; skipping .env load.")
        return
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.debug("Loaded environment from %s", env_path)
    else:
        logger.debug(".env not found at %s; relying on process environment.", env_path)


def _require_env(key: str, default: Optional[str] = None) -> str:
    """Helper to fetch an environment variable with optional default, raising ValueError if missing and default None."""
    val = os.getenv(key, default)
    if val is None or val == "":
        raise ValueError(f"Missing required environment variable: {key}")
    return val


# PUBLIC_INTERFACE
def get_config() -> dict:
    """Return configuration dict built from environment variables.

    PUBLIC_INTERFACE
    Returns a dictionary with the following keys (defaults noted):
    - DB_HOST
    - DB_PORT (int, default 3306)
    - DB_USER
    - DB_PASSWORD
    - DB_NAME (default 'bear_stats')
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION (default 'us-east-2')
    - S3_BUCKET
    - S3_PREFIX (optional, default '')

    Raises:
        ValueError: If any required configuration is missing.
    """
    _load_dotenv_if_available()

    cfg = {
        "DB_HOST": _require_env("DB_HOST") if os.getenv("DB_HOST") else None,
        "DB_PORT": int(os.getenv("DB_PORT", "3306")),
        "DB_USER": _require_env("DB_USER") if os.getenv("DB_USER") else None,
        "DB_PASSWORD": _require_env("DB_PASSWORD") if os.getenv("DB_PASSWORD") else None,
        "DB_NAME": os.getenv("DB_NAME", "bear_stats"),
        "AWS_ACCESS_KEY_ID": _require_env("AWS_ACCESS_KEY_ID") if os.getenv("AWS_ACCESS_KEY_ID") else None,
        "AWS_SECRET_ACCESS_KEY": _require_env("AWS_SECRET_ACCESS_KEY") if os.getenv("AWS_SECRET_ACCESS_KEY") else None,
        "AWS_REGION": os.getenv("AWS_REGION", "us-east-2"),
        "S3_BUCKET": _require_env("S3_BUCKET") if os.getenv("S3_BUCKET") else None,
        "S3_PREFIX": os.getenv("S3_PREFIX", ""),
    }

    # Validate required keys (those with None mean missing)
    missing = [k for k, v in cfg.items() if v is None and k in (
        "DB_HOST", "DB_USER", "DB_PASSWORD", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_BUCKET"
    )]
    if missing:
        raise ValueError(f"Missing required configuration keys in environment: {', '.join(missing)}")

    return cfg


# PUBLIC_INTERFACE
def parse_s3_url(url: str) -> Tuple[str, str]:
    """Parse an s3:// URL into (bucket, key).

    PUBLIC_INTERFACE
    Args:
        url: An S3 URL in the form s3://bucket/key with support for spaces (unescaped).

    Returns:
        A tuple (bucket, key).

    Raises:
        ValueError: If the URL is not a valid s3:// URL.
    """
    if not url.lower().startswith("s3://"):
        raise ValueError(f"Not an s3:// URL: {url}")
    # Strip scheme and split only on first slash to support keys with slashes and spaces
    rest = url[5:]
    parts = rest.split("/", 1)
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"Invalid s3 URL, expected s3://bucket/key: {url}")
    bucket, key = parts[0], parts[1]
    return bucket, key


def _infer_content_type(path: str) -> str:
    """Infer content type from file extension, defaulting to application/octet-stream."""
    ctype, _ = mimetypes.guess_type(path)
    return ctype or "application/octet-stream"


# PUBLIC_INTERFACE
def download_video(source: str, dest_path: str) -> str:
    """Download a video to dest_path if source is s3://, else copy or passthrough local file path.

    PUBLIC_INTERFACE
    Behavior:
    - If source starts with s3://, download using boto3.client('s3').download_file(bucket, key, dest_path)
    - Else if source is an existing local path, copy to dest_path if different, else return source
    - Ensures destination directory exists
    - Supports paths with spaces

    Args:
        source: Local path or s3:// URL for the source video.
        dest_path: Local destination file path to save the video.

    Returns:
        The local path to the downloaded/copied video.

    Raises:
        ValueError: If boto3 is not available or AWS env is missing when needed, or file not found for local path.
        Exception: For underlying S3 or filesystem errors.
    """
    _load_dotenv_if_available()
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    if source.lower().startswith("s3://"):
        if boto3 is None:
            raise ValueError("boto3 is required for S3 operations but is not installed.")
        # Validate AWS config
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-2")
        if not aws_access_key or not aws_secret:
            raise ValueError("Missing AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) for S3 download.")

        bucket, key = parse_s3_url(source)
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret,
                region_name=aws_region,
            )
            logger.info("Downloading from s3://%s/%s to %s", bucket, key, dest_path)
            s3.download_file(bucket, key, dest_path)
            return dest_path
        except (BotoCoreError, ClientError) as e:
            logger.exception("Failed to download from S3: %s", e)
            raise
    else:
        # Local file path
        if not os.path.exists(source):
            raise ValueError(f"Local source file not found: {source}")
        if os.path.abspath(source) == os.path.abspath(dest_path):
            logger.info("Source and destination are the same; returning %s", source)
            return source
        logger.info("Copying local file %s to %s", source, dest_path)
        shutil.copy2(source, dest_path)
        return dest_path


# PUBLIC_INTERFACE
def upload_file(local_path: str, bucket: str, prefix: str) -> str:
    """Upload a local file to S3 under bucket/prefix and return the s3:// URL.

    PUBLIC_INTERFACE
    Args:
        local_path: Path to the local file (supports spaces).
        bucket: S3 bucket name.
        prefix: Key prefix (e.g., "detections/"); can be empty or nested.

    Returns:
        The s3:// URL of the uploaded object.

    Raises:
        ValueError: If boto3 is not installed or AWS env missing or local file not found.
        Exception: For underlying S3 errors.
    """
    _load_dotenv_if_available()
    if boto3 is None:
        raise ValueError("boto3 is required for S3 operations but is not installed.")
    if not os.path.exists(local_path):
        raise ValueError(f"Local file not found: {local_path}")

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-2")
    if not aws_access_key or not aws_secret:
        raise ValueError("Missing AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) for S3 upload.")

    # Build S3 key
    fname = os.path.basename(local_path)
    # Normalize prefix: no leading slash, ensure trailing slash if non-empty
    norm_prefix = prefix.lstrip("/")
    if norm_prefix and not norm_prefix.endswith("/"):
        norm_prefix += "/"
    key = f"{norm_prefix}{fname}"

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region,
        )
        extra_args = {"ContentType": _infer_content_type(local_path)}
        logger.info("Uploading %s to s3://%s/%s", local_path, bucket, key)
        s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
        return f"s3://{bucket}/{key}"
    except (BotoCoreError, ClientError) as e:
        logger.exception("Failed to upload to S3: %s", e)
        raise


def _require_mysql_available():
    if pymysql is None:
        raise ValueError("pymysql is required for MySQL operations but is not installed.")


# PUBLIC_INTERFACE
def get_mysql_connection(db: Optional[str] = None):
    """Get a pymysql connection to the MySQL server or to the specified database.

    PUBLIC_INTERFACE
    Args:
        db: Optional database name. If None, connect to the server without selecting a database.

    Returns:
        A pymysql connection object.

    Raises:
        ValueError: If required DB environment variables are missing or pymysql is not installed.
        Exception: For underlying connection errors.
    """
    _load_dotenv_if_available()
    _require_mysql_available()

    host = _require_env("DB_HOST") if os.getenv("DB_HOST") else None
    user = _require_env("DB_USER") if os.getenv("DB_USER") else None
    password = _require_env("DB_PASSWORD") if os.getenv("DB_PASSWORD") else None
    port = int(os.getenv("DB_PORT", "3306"))

    if host is None or user is None or password is None:
        raise ValueError("Missing DB env (DB_HOST, DB_USER, DB_PASSWORD).")

    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        port=port,
        database=db if db else None,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        charset="utf8mb4",
    )
    return conn


# PUBLIC_INTERFACE
def ensure_database_exists(db_name: str) -> None:
    """Ensure that the specified database exists; create it if it does not.

    PUBLIC_INTERFACE
    Args:
        db_name: Name of the database to ensure exists.

    Raises:
        ValueError: If pymysql not installed or DB env missing.
        Exception: For underlying MySQL errors.
    """
    _load_dotenv_if_available()
    _require_mysql_available()
    if not db_name:
        raise ValueError("db_name must be a non-empty string.")

    conn = get_mysql_connection(db=None)
    try:
        with conn.cursor() as cur:
            logger.info("Ensuring database exists: %s", db_name)
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    finally:
        conn.close()


# PUBLIC_INTERFACE
def ensure_table_exists(db_name: str) -> None:
    """Ensure that the 'detections' table exists in db_name; create it if it does not.

    PUBLIC_INTERFACE
    Table schema:
        CREATE TABLE IF NOT EXISTS detections (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          video_source VARCHAR(512),
          frame_index INT,
          timestamp_sec DOUBLE,
          class_label VARCHAR(64),
          confidence FLOAT,
          bbox_x INT, bbox_y INT, bbox_w INT, bbox_h INT,
          s3_image_url VARCHAR(1024),
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    Args:
        db_name: Database name where the table should exist.

    Raises:
        ValueError: If pymysql not installed or DB env missing.
        Exception: For underlying MySQL errors.
    """
    _load_dotenv_if_available()
    _require_mysql_available()

    conn = get_mysql_connection(db=db_name)
    try:
        with conn.cursor() as cur:
            logger.info("Ensuring table 'detections' exists in database: %s", db_name)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                  id BIGINT AUTO_INCREMENT PRIMARY KEY,
                  video_source VARCHAR(512),
                  frame_index INT,
                  timestamp_sec DOUBLE,
                  class_label VARCHAR(64),
                  confidence FLOAT,
                  bbox_x INT, bbox_y INT, bbox_w INT, bbox_h INT,
                  s3_image_url VARCHAR(1024),
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
    finally:
        conn.close()


# PUBLIC_INTERFACE
def insert_detection(db_name: str, row: dict) -> int:
    """Insert a detection row into db_name.detections and return the inserted id.

    PUBLIC_INTERFACE
    Expected keys in row dict:
      - video_source (str)
      - frame_index (int)
      - timestamp_sec (float)
      - class_label (str)
      - confidence (float)
      - bbox_x (int), bbox_y (int), bbox_w (int), bbox_h (int)
      - s3_image_url (str)

    Args:
        db_name: Database to insert the row into.
        row: Dictionary of fields for the detections table.

    Returns:
        The auto-incremented primary key (id) of the inserted row.

    Raises:
        ValueError: If required keys are missing or pymysql not installed or DB env missing.
        Exception: For underlying MySQL errors.
    """
    _load_dotenv_if_available()
    _require_mysql_available()

    required = [
        "video_source", "frame_index", "timestamp_sec", "class_label", "confidence",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "s3_image_url"
    ]
    missing = [k for k in required if k not in row]
    if missing:
        raise ValueError(f"Missing required detection fields: {', '.join(missing)}")

    conn = get_mysql_connection(db=db_name)
    try:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO detections (
                    video_source, frame_index, timestamp_sec, class_label, confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h, s3_image_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                row["video_source"], row["frame_index"], row["timestamp_sec"], row["class_label"], row["confidence"],
                row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"], row["s3_image_url"]
            )
            cur.execute(sql, params)
            # Retrieve last row id
            cur.execute("SELECT LAST_INSERT_ID() AS id;")
            res = cur.fetchone()
            insert_id = int(res["id"]) if res and "id" in res else 0
            logger.debug("Inserted detection row id=%s", insert_id)
            return insert_id
    finally:
        conn.close()
