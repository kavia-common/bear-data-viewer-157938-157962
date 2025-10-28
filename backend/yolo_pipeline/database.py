"""Database operations for the YOLO pipeline."""

import sys
from typing import Optional
from contextlib import contextmanager

try:
    import mysql.connector as mysql_connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    print("[WARN] mysql-connector-python not available", file=sys.stderr)
    mysql_connector = None
    MySQLError = Exception
    MYSQL_AVAILABLE = False

from .config import get_db_env

@contextmanager
def db_connection(database: Optional[str] = None):
    """Context manager that yields a MySQL/MariaDB connection."""
    if not MYSQL_AVAILABLE:
        raise RuntimeError("mysql-connector-python is required but not installed.")

    db_config = get_db_env()
    db_to_use = database if database is not None else db_config['db_name']

    conn = None
    try:
        conn = mysql_connector.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_to_use,
            autocommit=True,
        )
        yield conn
    except MySQLError as e:
        print(f"[DB] Connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def setup_database_and_table():
    """Create the database and animal_detections_v3 table if they don't exist."""
    if not MYSQL_AVAILABLE:
        print("[DB-SETUP] MySQL connector not available, skipping database setup.")
        return

    db_config = get_db_env()
    db_name = db_config['db_name']
    
    if not db_name:
        print("[WARN] DB_NAME not set, skipping database setup.")
        return

    # Create database
    try:
        with db_connection(None) as conn:
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            cursor.close()
            print(f"[DB-SETUP] Database '{db_name}' ready")
    except Exception as e:
        print(f"[DB-SETUP] Failed creating database: {e}")
        return

    create_table_sql = """
        CREATE TABLE IF NOT EXISTS animal_detections_v3 (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            frame_time_seconds DECIMAL(10,3) NOT NULL,
            in_video_time DECIMAL(10,3) DEFAULT NULL,
            detection_label VARCHAR(100) NOT NULL,
            bbox_x1 INT NOT NULL,
            bbox_y1 INT NOT NULL,
            bbox_x2 INT NOT NULL,
            bbox_y2 INT NOT NULL,
            detection_confidence DECIMAL(6,4) NOT NULL,
            classification_label VARCHAR(255) DEFAULT NULL,
            classification_confidence DECIMAL(6,4) DEFAULT NULL,
            pose_status VARCHAR(50) DEFAULT NULL,
            pose_keypoints_count INT DEFAULT 0,
            frame_s3_url VARCHAR(1024) DEFAULT NULL,
            video_source VARCHAR(512) DEFAULT NULL,
            ocr_date_text VARCHAR(512) DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_frame_time (frame_time_seconds),
            INDEX idx_in_video_time (in_video_time),
            INDEX idx_label (detection_label),
            INDEX idx_created (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    try:
        with db_connection(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES LIKE 'animal_detections_v3'")
            if cursor.fetchone() is None:
                cursor.execute(create_table_sql)
                print("[DB-SETUP] Table 'animal_detections_v3' created successfully")
            else:
                print("[DB-SETUP] Table 'animal_detections_v3' already exists")
            cursor.close()
    except Exception as e:
        print(f"[DB-SETUP] Failed creating table: {e}")

def insert_detection(time_s: float, label: str, x1: int, y1: int, x2: int, y2: int, 
                    conf: float, cls_label: str, cls_conf: float, 
                    pose_status: str, num_keypoints: int, s3_url: str,
                    video_source: str, in_video_time: Optional[float] = None,
                    ocr_date_text: Optional[str] = None) -> bool:
    """Insert a single detection record into the animal_detections_v3 table.
    
    Args:
        time_s: Frame time in seconds (from video timeline)
        label: Detection label (e.g., 'bear')
        x1, y1, x2, y2: Bounding box coordinates
        conf: Detection confidence
        cls_label: Classification label
        cls_conf: Classification confidence
        pose_status: Pose estimation status
        num_keypoints: Number of detected keypoints
        s3_url: S3 URL of the frame image
        video_source: Source video URL
        in_video_time: Optional time in video computed from frame_index/fps
        ocr_date_text: Optional OCR-extracted date text from frame
        
    Returns:
        bool: True if insertion was successful, False otherwise
    """
    if not MYSQL_AVAILABLE:
        return False

    db_config = get_db_env()
    db_name = db_config['db_name']
    
    if not db_name:
        return False

    insert_sql = """
        INSERT INTO animal_detections_v3
            (frame_time_seconds, in_video_time, detection_label, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
             detection_confidence, classification_label, classification_confidence, 
             pose_status, pose_keypoints_count, frame_s3_url, video_source, ocr_date_text)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        with db_connection(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(insert_sql, (
                float(time_s),
                float(in_video_time) if in_video_time is not None else None,
                label, 
                x1, y1, x2, y2, 
                float(conf), 
                cls_label if cls_label else None, 
                float(cls_conf) if cls_conf else None, 
                pose_status if pose_status else None, 
                int(num_keypoints) if num_keypoints else 0, 
                s3_url if s3_url else None,
                video_source,
                ocr_date_text if ocr_date_text else None
            ))
            cursor.close()
            return True
    except Exception as e:
        print(f"[DB] Failed to insert detection: {e}")
        return False