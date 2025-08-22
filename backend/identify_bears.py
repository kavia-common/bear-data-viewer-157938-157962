#!/usr/bin/env python3
"""
identify_bears.py

Continuously downloads a video (bears.mp4) from an S3 bucket, loads a YOLOv8
Lite model, and performs bear detection on video frames. For each detection with
confidence >= 0.3, records:
- bounding box coordinates (x1, y1, x2, y2)
- timestamp (UTC ISO-8601)
- hardcoded bearID ('011')
- detection probability
- label (e.g., 'bear')

Design principles:
- Modular architecture and separation of concerns
- Explicit model and data loading
- Clear logging and error handling
- Extensible structure with typed public interfaces and docstrings

No real S3 or YOLO credentials are included; placeholders and comments indicate where to configure them.
"""

import os
import sys
import time
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

# Optional dependencies (boto3 for S3, ultralytics for YOLOv8).
# These are intentionally optional to keep this script framework-like.
# If running in an environment without these libs, install them and/or
# add them to backend/requirements.txt as needed by your deployment.
try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover - optional dep
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dep
    YOLO = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dep
    cv2 = None  # type: ignore


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

DEFAULT_S3_BUCKET = os.environ.get("BEAR_S3_BUCKET", "identify_bear_bucket")
DEFAULT_S3_KEY = os.environ.get("BEAR_S3_KEY", "bears.mp4")
DEFAULT_DOWNLOAD_INTERVAL_SEC = float(os.environ.get("DOWNLOAD_INTERVAL_SEC", "30"))
DEFAULT_INFERENCE_SLEEP_SEC = float(os.environ.get("INFERENCE_LOOP_SLEEP_SEC", "2"))

# YOLOv8 lite model path or name. Use a "lite" variant if available. Placeholder:
# Examples that might work depending on your env:
# - "yolov8n.pt" (nano) or a lite quantized model checkpoint you provide.
# - "yolov8n-lite.pt" (placeholder if you have a lite-optimized model).
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")

# Probability threshold
PROB_THRESHOLD = float(os.environ.get("PROBABILITY_THRESHOLD", "0.3"))

# Hardcoded bear ID
HARDCODED_BEAR_ID = "011"


# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------

def _setup_logging() -> None:
    """
    Configure application logging with a standard formatter.
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout,
    )


logger = logging.getLogger("identify_bears")


# ------------------------------------------------------------------------------
# S3 Handling
# ------------------------------------------------------------------------------

def _get_s3_client() -> Optional[Any]:
    """
    Create and return a boto3 S3 client using environment variables or
    instance/role credentials when available.

    Expected env variables (optional):
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_DEFAULT_REGION or AWS_REGION

    Returns:
        boto3.client('s3') or None if boto3 is not installed.
    """
    if boto3 is None:
        logger.warning("boto3 is not installed. S3 download will be skipped.")
        return None

    # If you need to target a specific S3-compatible endpoint (e.g., MinIO),
    # set AWS_S3_ENDPOINT_URL in the environment.
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL")
    session = boto3.session.Session()
    s3_client = session.client("s3", endpoint_url=endpoint_url)
    return s3_client


# PUBLIC_INTERFACE
def download_video_from_s3(bucket: str, key: str, dest_path: str) -> bool:
    """
    Download a video object from S3 to the given destination path.

    Args:
        bucket: Name of the S3 bucket.
        key: Object key (path) within the bucket.
        dest_path: Local filesystem path where the object will be saved.

    Returns:
        True on success, False on failure.
    """
    s3 = _get_s3_client()
    if s3 is None:
        logger.error("S3 client unavailable. Cannot download from S3.")
        return False

    try:
        logger.info("Downloading s3://%s/%s to %s", bucket, key, dest_path)
        s3.download_file(bucket, key, dest_path)  # type: ignore
        return True
    except (BotoCoreError, ClientError) as e:  # type: ignore
        logger.exception("Failed to download from S3: %s", e)
        return False
    except Exception as e:
        logger.exception("Unexpected error during S3 download: %s", e)
        return False


# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------

# PUBLIC_INTERFACE
def load_yolo_model(model_path: str) -> Any:
    """
    Load a YOLOv8 model.

    Args:
        model_path: Path or model name to load (e.g., 'yolov8n.pt').

    Returns:
        The loaded model instance.

    Raises:
        RuntimeError: If ultralytics is not installed or the model cannot be loaded.
    """
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Please install it to run YOLOv8 inference."
        )
    logger.info("Loading YOLOv8 model from: %s", model_path)
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.exception("Failed to load YOLO model: %s", e)
        raise RuntimeError(f"Failed to load YOLO model: {e}") from e
    return model


# ------------------------------------------------------------------------------
# Video Handling
# ------------------------------------------------------------------------------

def _open_video_capture(video_path: str) -> Optional[Any]:
    """
    Open a video file for frame-by-frame reading.

    Args:
        video_path: Path to the video file.

    Returns:
        cv2.VideoCapture instance or None if OpenCV not installed or open fails.
    """
    if cv2 is None:
        logger.error("OpenCV (cv2) is not installed. Unable to process video.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        return None
    return cap


def _frame_generator(video_path: str) -> Generator[Tuple[Any, float], None, None]:
    """
    Yield frames and their timestamps (in seconds) from the video.

    Args:
        video_path: Path to the video file.

    Yields:
        Tuple[frame, timestamp_sec] for each frame.
    """
    cap = _open_video_capture(video_path)
    if cap is None:
        return

    fps = cap.get(cv2.CAP_PROP_FPS) if cv2 is not None else 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_sec = (frame_index / fps) if fps and fps > 0 else 0.0
            yield frame, timestamp_sec
            frame_index += 1
    finally:
        cap.release()


# ------------------------------------------------------------------------------
# Inference and Post-processing
# ------------------------------------------------------------------------------

def _predict(model: Any, frame: Any) -> Any:
    """
    Run model prediction on a single frame.

    Args:
        model: YOLO model instance.
        frame: Input frame (numpy array).

    Returns:
        Raw model result for this frame (library-specific object).
    """
    # For ultralytics YOLO, calling model(frame) returns results list.
    return model(frame)


def _utc_iso_now() -> str:
    """
    Returns current UTC time in ISO 8601 format.

    Returns:
        ISO 8601 UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


# PUBLIC_INTERFACE
def filter_and_format_detections(raw_results: Any, threshold: float) -> List[Dict[str, Any]]:
    """
    Filter model detections by a probability threshold and format them into
    standardized records.

    Args:
        raw_results: Model output for a single frame (ultralytics Results list).
        threshold: Minimum detection confidence to include.

    Returns:
        A list of dicts with keys:
        - bounding_box_coordinates: tuple[float, float, float, float] (x1, y1, x2, y2)
        - timestamp: str (UTC ISO-8601)
        - bearID: str ('011')
        - probability: float
        - label: str
    """
    formatted: List[Dict[str, Any]] = []
    now_iso = _utc_iso_now()

    try:
        # ultralytics returns a list-like Results; iterate accordingly.
        # Each element has .boxes (tensor) where each box has xyxy, conf, and cls.
        for result in raw_results:
            names = getattr(result, "names", {})  # id -> label mapping
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            # boxes.conf: Nx1 confidence
            # boxes.cls: Nx1 class id
            # boxes.xyxy: Nx4 bounding boxes
            confs = getattr(boxes, "conf", None)
            clses = getattr(boxes, "cls", None)
            xyxy = getattr(boxes, "xyxy", None)
            if confs is None or clses is None or xyxy is None:
                continue

            # Convert to CPU lists if tensors
            try:
                conf_list = confs.squeeze(-1).tolist()
            except Exception:
                conf_list = confs.tolist() if hasattr(confs, "tolist") else list(confs)

            cls_list = clses.squeeze(-1).tolist() if hasattr(clses, "squeeze") else (
                clses.tolist() if hasattr(clses, "tolist") else list(clses)
            )
            xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)

            for i, p in enumerate(conf_list):
                if p is None:
                    continue
                try:
                    prob = float(p)
                except Exception:
                    continue
                if prob < threshold:
                    continue

                # Resolve label
                cls_id = int(cls_list[i]) if i < len(cls_list) else -1
                label = names.get(cls_id, str(cls_id))

                # Bounding box
                if i < len(xyxy_list):
                    box = xyxy_list[i]
                    # Ensure a 4-element tuple
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    else:
                        x1 = y1 = x2 = y2 = 0.0
                else:
                    x1 = y1 = x2 = y2 = 0.0

                record = {
                    "bounding_box_coordinates": (x1, y1, x2, y2),
                    "timestamp": now_iso,
                    "bearID": HARDCODED_BEAR_ID,
                    "probability": prob,
                    "label": label,
                }
                formatted.append(record)

    except Exception as e:
        logger.exception("Error while formatting detections: %s", e)

    return formatted


# ------------------------------------------------------------------------------
# Results Handling
# ------------------------------------------------------------------------------

# PUBLIC_INTERFACE
def write_results(records: Iterable[Dict[str, Any]]) -> None:
    """
    Handle persisting or emitting results. This is a stub designed for extension.

    For now, it logs the detections. In a real deployment, you might:
      - Publish to a message queue (Kafka, RabbitMQ)
      - Insert into a database
      - Write to a file or object store
      - Emit metrics/events

    Args:
        records: Iterable of detection dicts returned by filter_and_format_detections.
    """
    count = 0
    for rec in records:
        logger.info(
            "Detection | bearID=%s label=%s prob=%.3f bbox=%s ts=%s",
            rec.get("bearID"),
            rec.get("label"),
            float(rec.get("probability", 0)),
            rec.get("bounding_box_coordinates"),
            rec.get("timestamp"),
        )
        count += 1
    if count == 0:
        logger.debug("No detections above threshold in this frame.")


# ------------------------------------------------------------------------------
# Orchestration Loop
# ------------------------------------------------------------------------------

def _ensure_latest_video(tmp_dir: str, bucket: str, key: str) -> Optional[str]:
    """
    Download the latest video to a deterministic path under tmp_dir.

    Args:
        tmp_dir: Temporary directory where the file will be placed.
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        The path to the downloaded video on success, or None on failure.
    """
    dest_path = os.path.join(tmp_dir, "bears.mp4")
    ok = download_video_from_s3(bucket, key, dest_path)
    return dest_path if ok else None


def _process_video(model: Any, video_path: str) -> None:
    """
    Process a single video file: read frames, run inference, filter, and write results.

    Args:
        model: Loaded YOLO model instance.
        video_path: Path to the video file to process.
    """
    for frame, _ts in _frame_generator(video_path):
        if frame is None:
            continue
        try:
            raw = _predict(model, frame)
            records = filter_and_format_detections(raw, PROB_THRESHOLD)
            write_results(records)
        except Exception as e:
            logger.exception("Inference error: %s", e)


# PUBLIC_INTERFACE
def run_loop(
    s3_bucket: str = DEFAULT_S3_BUCKET,
    s3_key: str = DEFAULT_S3_KEY,
    model_path: str = YOLO_MODEL_PATH,
    download_interval_sec: float = DEFAULT_DOWNLOAD_INTERVAL_SEC,
    idle_sleep_sec: float = DEFAULT_INFERENCE_SLEEP_SEC,
) -> None:
    """
    Main orchestration loop. Continuously:
      1. Downloads bears.mp4 from S3.
      2. Loads YOLOv8 Lite model (once).
      3. Processes the video frames, emitting detections.

    Args:
        s3_bucket: S3 bucket name to download from.
        s3_key: S3 object key for the video file (e.g., 'bears.mp4').
        model_path: YOLO model path or name. Use a Lite variant if available.
        download_interval_sec: Interval between refresh downloads of the video.
        idle_sleep_sec: Sleep between frames or on idle.

    Notes:
        - This loop is designed to run indefinitely until interrupted.
        - All secrets/credentials should be provided via environment variables or
          external configuration. This script does not hardcode credentials.
    """
    logger.info("Starting identify_bears loop. Bucket=%s Key=%s", s3_bucket, s3_key)

    try:
        model = load_yolo_model(model_path)
    except RuntimeError as e:
        logger.error("Model loading failed: %s", e)
        return

    with tempfile.TemporaryDirectory(prefix="bear-identify-") as tmp_dir:
        last_download_time = 0.0
        video_path: Optional[str] = None

        while True:
            now = time.time()
            if (now - last_download_time) >= download_interval_sec or not video_path:
                candidate = _ensure_latest_video(tmp_dir, s3_bucket, s3_key)
                if candidate:
                    video_path = candidate
                    last_download_time = now
                    logger.info("Video refreshed at %s", _utc_iso_now())
                else:
                    logger.warning("Video download failed; will retry after interval.")

            if video_path and os.path.exists(video_path):
                _process_video(model, video_path)
            else:
                logger.debug("No video available yet.")

            time.sleep(idle_sleep_sec)


# ------------------------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------------------------

def main() -> None:
    """
    CLI entrypoint to start the continuous bear identification loop.
    """
    _setup_logging()
    try:
        run_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")
    except Exception as e:
        logger.exception("Fatal error in identify_bears: %s", e)


if __name__ == "__main__":
    main()
