"""YOLO pipeline package for video processing and object detection."""

from .config import (
    SAMPLE_INTERVAL_SECONDS,
    DETECTION_WEIGHTS,
    CLASSIFICATION_WEIGHTS,
    POSE_WEIGHTS,
    OUTPUT_FILENAME,
    S3_BUCKET,
    S3_VIDEO_KEY,
    S3_FRAMES_PREFIX,
    S3_CSV_PREFIX,
    HARDCODED_VIDEO_URL,
    get_db_env,
    get_labels
)
from .database import setup_database_and_table, insert_detection
from .models import load_models, process_detection
from .output import write_csv_header, append_detection
from .storage import upload_to_s3
from .video import (
    open_video_from_source,
    get_video_meta,
    compute_sample_times,
    seek_and_read_frame,
    save_frame_temp_jpg
)
from .ocr import (
    extract_date_label_from_frame,
    extract_date_label_from_image_path,
    get_ocr_reader
)
