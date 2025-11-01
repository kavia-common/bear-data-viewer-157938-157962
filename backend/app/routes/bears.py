import os
import logging
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from flask import Blueprint, current_app, jsonify

# Create blueprint for bears routes
bears_bp = Blueprint("bears", __name__)
# Alias expected by app/__init__.py
blp = bears_bp

# Environment variable keys for DB
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Lazy import to avoid mandatory dependency at import time during tests without DB
def _get_mysql_connector():
    try:
        import mysql.connector  # type: ignore
        return mysql.connector
    except Exception as e:
        current_app.logger.warning(f"mysql-connector-python not available or failed to import: {e}")
        return None


@contextmanager
def db_connection():
    """
    Context manager for optional DB connection.
    Attempts to connect if DB env vars are present; yields connection or None on failure.
    Ensures connection is closed safely.
    """
    connector = _get_mysql_connector()
    conn = None
    if connector and DB_HOST and DB_USER and DB_NAME:
        try:
            conn = connector.connect(
                host=DB_HOST,
                port=int(DB_PORT) if DB_PORT else 3306,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
            )
            yield conn
        except Exception as e:
            # Log warning and yield None to fallback to empty result
            logging.getLogger(__name__).warning(f"DB connection failed: {e}")
            yield None
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
    else:
        if not connector:
            logging.getLogger(__name__).info("mysql-connector not available; serving empty bears list.")
        else:
            logging.getLogger(__name__).info("DB env vars missing; serving empty bears list.")
        yield None


def _serialize_timestamp(ts: Any) -> Optional[str]:
    """
    Convert various DB timestamp types to ISO 8601 string in UTC.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        # Ensure timezone-aware UTC
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        return ts.isoformat()
    # Fallback to string
    try:
        return str(ts)
    except Exception:
        return None


def _fetch_bears_from_db(conn) -> List[Dict[str, Any]]:
    """
    Fetch bear detection rows from the database and map to API schema.
    Adjust SQL and field names to your schema as needed.
    """
    results: List[Dict[str, Any]] = []
    try:
        cursor = conn.cursor(dictionary=True)
        # Example table/fields; adjust to real schema if available.
        # Fields expected to build response:
        #   id -> bearId
        #   pose -> pose
        #   created_at -> timestamp
        #   frame_index, fps -> in_video_time (frame_index / fps)
        #   ocr_date_text -> ocr_date_text
        query = """
            SELECT
                id,
                pose,
                created_at,
                frame_index,
                fps,
                ocr_date_text
            FROM bear_detections
            ORDER BY created_at DESC
            LIMIT 100
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            frame_index = row.get("frame_index")
            fps = row.get("fps")
            in_video_time = None
            try:
                if frame_index is not None and fps:
                    in_video_time = float(frame_index) / float(fps) if float(fps) != 0 else None
            except Exception:
                in_video_time = None

            results.append(
                {
                    "bearId": str(row.get("id")) if row.get("id") is not None else "",
                    "pose": row.get("pose") or "",
                    "timestamp": _serialize_timestamp(row.get("created_at")),
                    "in_video_time": in_video_time,
                    "ocr_date_text": row.get("ocr_date_text"),
                }
            )
        cursor.close()
    except Exception as e:
        current_app.logger.warning(f"DB query failed, returning empty list: {e}")
        results = []
    return results


# PUBLIC_INTERFACE
@bears_bp.get("/api/bears")
def get_bears():
    """
    Returns a list of Bear detection records from the database. Each record contains:
    - bearId: String ID of the bear (detection record ID)
    - pose: String describing the bear's pose status
    - timestamp: ISO 8601 UTC timestamp when the detection was recorded
    - in_video_time: Time in seconds within the video (computed from frame_index/fps)
    - ocr_date_text: OCR-extracted date text from the video frame

    Returns:
        list[dict]: A list of bear detection records suitable for JSON serialization.

    Note:
        Falls back to empty list if database is not available.
    """
    with db_connection() as conn:
        data: List[Dict[str, Any]] = []
        if conn:
            data = _fetch_bears_from_db(conn)

        # If DB is unavailable or returned no rows, provide 3 mock records
        if not data:
            now = datetime.now(timezone.utc)
            mock = [
                {
                    "bearId": "1",
                    "pose": "standing",
                    "timestamp": (now.replace(microsecond=0)).isoformat(),
                },
                {
                    "bearId": "2",
                    "pose": "walking",
                    "timestamp": (now.replace(microsecond=0) - timedelta(seconds=10)).isoformat(),
                },
                {
                    "bearId": "3",
                    "pose": "sitting",
                    "timestamp": (now.replace(microsecond=0) - timedelta(seconds=20)).isoformat(),
                },
            ]
            # Ensure sorted by timestamp descending (most recent first)
            data = sorted(mock, key=lambda x: x["timestamp"], reverse=True)

    return jsonify(data), 200


# PUBLIC_INTERFACE
@bears_bp.get("/api/bears/detectmotion")
def detect_motion_stub():
    """
    Stub endpoint to simulate bear motion detection.

    Returns:
        dict: Simple status payload for frontend integration testing.
    """
    return jsonify({"status": "ok"}), 200
