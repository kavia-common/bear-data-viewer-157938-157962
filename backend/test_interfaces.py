"""
A minimal helper for uploading and downloading sample files to/from S3.

Requirements:
- Hardcode bucket: "humanlabelimg-poc"
- Upload local file: "samplefileupload.txt" (same directory as this script)
- Download object: "samplefiledownload.txt" to current (same) directory
- Use AWS credentials from environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
- Region from AWS_REGION if present, default to 'us-east-1'
- No argument parser or additional logic
"""

import os
import pathlib

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

BUCKET_NAME = "humanlabelimg-poc"
LOCAL_FILE_NAME = "samplefileupload.txt"
DOWNLOAD_OBJECT_KEY = "samplefiledownload.txt"


# PUBLIC_INTERFACE
def upload_sample_file_to_s3() -> None:
    """
    Uploads the local file 'samplefileupload.txt' (expected in the same folder as this file)
    to the S3 bucket 'humanlabelimg-poc'.

    Credentials are read from environment variables:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
    Optionally, AWS_REGION (defaults to 'us-east-1' if not provided)

    Raises exceptions on failure to allow callers to detect issues, but prints concise
    status messages for quick visibility.
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not access_key or not secret_key:
        # Raise the standard boto exception used when credentials are missing
        raise NoCredentialsError()

    script_dir = pathlib.Path(__file__).resolve().parent
    local_path = script_dir / LOCAL_FILE_NAME

    if not local_path.exists():
        raise FileNotFoundError(f"Required file not found: {local_path}")

    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    s3 = session.client("s3")

    # Use the same filename as the S3 object key
    object_key = LOCAL_FILE_NAME

    try:
        s3.upload_file(str(local_path), BUCKET_NAME, object_key)
        print("Upload Success")
    except (BotoCoreError, ClientError) as e:
        # Re-raise for visibility while printing a helpful message
        print(f"S3 upload failed: {e}")
        raise


# PUBLIC_INTERFACE
def download_sample_file_from_s3() -> None:
    """
    Downloads the S3 object 'samplefiledownload.txt' from the bucket 'humanlabelimg-poc'
    into the current directory (same folder as this script).

    Credentials are read from environment variables:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
    Optionally, AWS_REGION (defaults to 'us-east-1' if not provided)

    Prints concise status messages and raises on fatal errors to surface issues.
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not access_key or not secret_key:
        raise NoCredentialsError()

    script_dir = pathlib.Path(__file__).resolve().parent
    local_dest = script_dir / DOWNLOAD_OBJECT_KEY

    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3 = session.client("s3")

    try:
        s3.download_file(BUCKET_NAME, DOWNLOAD_OBJECT_KEY, str(local_dest))
        print("Download Success")
    except (BotoCoreError, ClientError) as e:
        print(f"S3 download failed: {e}")
        raise


from typing import Optional, Tuple

# New imports for DB connectivity and env loading
from contextlib import contextmanager
from dotenv import load_dotenv

try:
    import mysql.connector as mysql_connector  # type: ignore
    from mysql.connector import Error as MySQLError  # type: ignore
except Exception:  # pragma: no cover - optional import resolution
    mysql_connector = None  # type: ignore
    MySQLError = Exception  # type: ignore


def _get_db_env() -> Tuple[str, int, str, str, str]:
    """
    Internal helper to load DB connection settings from environment.
    Uses python-dotenv to load .env if present.
    """
    load_dotenv()  # load from .env if available

    host = os.getenv("DB_HOST", "")
    port_raw = os.getenv("DB_PORT", "3306")
    user = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "")

    try:
        port = int(port_raw)
    except ValueError:
        print(f"[DB] Invalid DB_PORT value '{port_raw}', defaulting to 3306")
        port = 3306

    if not host or not user:
        print("[DB] Missing required DB env vars. Please set DB_HOST and DB_USER at minimum.")
    if not db_name:
        print("[DB] DB_NAME is empty; operations requiring a specific database may fail.")

    return host, port, user, password, db_name


@contextmanager
def _db_connection(database: Optional[str] = None):
    """
    Context manager that yields a MySQL/MariaDB connection, ensuring cleanup.
    - If `database` is provided, connects directly to it.
    - Otherwise, connects without specifying a DB (for CREATE DATABASE IF NOT EXISTS).
    """
    if mysql_connector is None:
        raise RuntimeError(
            "mysql-connector-python is required but not installed. "
            "Please add it to backend/requirements.txt and install dependencies."
        )

    host, port, user, password, db_name_env = _get_db_env()
    db_to_use = database if database is not None else None

    conn = None
    try:
        print(f"[DB] Connecting to MariaDB/MySQL at {host}:{port} (db={db_to_use or '-'}) ...")
        conn = mysql_connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_to_use,
            autocommit=True,
        )
        if conn.is_connected():
            print("[DB] Connection established.")
        yield conn
    except MySQLError as e:
        print(f"[DB] Connection error: {e}")
        raise
    finally:
        try:
            if conn is not None and conn.is_connected():
                conn.close()
                print("[DB] Connection closed.")
        except Exception:
            # Avoid masking original exceptions
            pass


# PUBLIC_INTERFACE
def setup_database_and_table() -> None:
    """
    Create the database if it doesn't exist, then create the detections table if it doesn't exist.
    This encapsulates the MariaDB connection and DB/table creation logic.

    Env vars required:
    - DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

    The detections table schema:
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      frame_time_seconds INT,
      label VARCHAR(255),
      x1 INT, y1 INT, x2 INT, y2 INT,
      obj_label_confidence FLOAT,
      pose VARCHAR(255),
      S3_img_link VARCHAR(1024),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    """
    host, port, user, password, db_name = _get_db_env()
    if not db_name:
        raise ValueError("DB_NAME must be set to create or access the database.")

    print("[DB-SETUP] Starting database and table setup ...")

    # 1) Connect without database and create DB if not exists
    try:
        with _db_connection(database=None) as conn:
            cursor = None
            try:
                cursor = conn.cursor()
                print(f"[DB-SETUP] Creating database if not exists: `{db_name}`")
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
                print("[DB-SETUP] Database ensured.")
            finally:
                if cursor is not None:
                    cursor.close()
    except Exception as e:
        print(f"[DB-SETUP] Failed creating/ensuring database: {e}")
        raise

    # 2) Reconnect to the newly ensured database and create table if not exists
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
            cursor = None
            try:
                cursor = conn.cursor()
                print("[DB-SETUP] Creating table `detections` if not exists ...")
                cursor.execute(create_table_sql)
                print("[DB-SETUP] Table ensured.")
            finally:
                if cursor is not None:
                    cursor.close()
    except Exception as e:
        print(f"[DB-SETUP] Failed creating/ensuring table: {e}")
        raise

    print("[DB-SETUP] Completed database and table setup.")


# PUBLIC_INTERFACE
def insert_and_read_sample_data() -> None:
    """
    Insert sample rows into the detections table and then select and print them.
    Uses parameterized INSERT statements and prints selected results clearly.
    """
    host, port, user, password, db_name = _get_db_env()
    if not db_name:
        raise ValueError("DB_NAME must be set to insert and read data.")

    print("[DB-DATA] Inserting sample rows into `detections` ...")

    insert_sql = """
        INSERT INTO detections
            (frame_time_seconds, label, x1, y1, x2, y2, obj_label_confidence, pose, S3_img_link)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    sample_rows = [
        (10, "bear", 12, 18, 200, 240, 0.92, "standing", "s3://bucket/path/img1.jpg"),
        (15, "bear", 30, 45, 220, 260, 0.88, "walking", "s3://bucket/path/img2.jpg"),
        (22, "bear", 50, 60, 240, 280, 0.95, "sitting", "s3://bucket/path/img3.jpg"),
    ]

    try:
        with _db_connection(database=db_name) as conn:
            cursor = None
            try:
                cursor = conn.cursor()
                for row in sample_rows:
                    cursor.execute(insert_sql, row)
                print(f"[DB-DATA] Inserted {len(sample_rows)} rows.")
            finally:
                if cursor is not None:
                    cursor.close()
    except Exception as e:
        print(f"[DB-DATA] Insert failed: {e}")
        raise

    print("[DB-DATA] Selecting rows from `detections` ...")
    select_sql = """
        SELECT id, frame_time_seconds, label, x1, y1, x2, y2, obj_label_confidence, pose, S3_img_link, created_at
        FROM detections
        ORDER BY id DESC
        LIMIT 10
    """
    try:
        with _db_connection(database=db_name) as conn:
            cursor = None
            try:
                cursor = conn.cursor()
                cursor.execute(select_sql)
                rows = cursor.fetchall()
                print(f"[DB-DATA] Retrieved {len(rows)} rows. Showing results:")
                for r in rows:
                    print(
                        f" - id={r[0]} time={r[1]}s label={r[2]} bbox=({r[3]},{r[4]},{r[5]},{r[6]}) "
                        f"conf={r[7]} pose={r[8]} s3={r[9]} created_at={r[10]}"
                    )
            finally:
                if cursor is not None:
                    cursor.close()
    except Exception as e:
        print(f"[DB-DATA] Select failed: {e}")
        raise


# PUBLIC_INTERFACE
def main() -> None:
    """
    Entrypoint to run DB setup and sample data operations.
    Also retains previous S3 demo behavior if needed, printing clear steps.
    """
    print("[MAIN] Starting operations ...")

    # Keep S3 sample calls optional and non-blocking if AWS creds are missing.
    try:
        print("[MAIN] S3: Attempting sample upload and download ...")
        upload_sample_file_to_s3()
        download_sample_file_from_s3()
        print("[MAIN] S3 operations completed.")
    except Exception as e:
        print(f"[MAIN] S3 operations skipped due to error: {e}")

    # DB operations
    print("[MAIN] DB: Setting up database and table ...")
    setup_database_and_table()
    print("[MAIN] DB: Inserting and reading sample data ...")
    insert_and_read_sample_data()

    print("[MAIN] All operations complete. Success.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[MAIN] Error: {exc}")
        raise
