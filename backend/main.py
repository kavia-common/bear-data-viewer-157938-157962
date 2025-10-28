#!/usr/bin/env python3
"""
Main script for the YOLO video processing pipeline.

This script:
- Opens a video from a hardcoded S3 HTTPS URL
- Samples frames at regular intervals
- Runs object detection, classification, and pose estimation
- Saves results to CSV and database
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

from yolo_pipeline import (
    HARDCODED_VIDEO_URL, OUTPUT_FILENAME, S3_BUCKET, S3_FRAMES_PREFIX, S3_CSV_PREFIX,
    setup_database_and_table,
    load_models,
    open_video_from_source,
    get_video_meta,
    compute_sample_times,
    seek_and_read_frame,
    save_frame_temp_jpg,
    process_detection,
    write_csv_header,
    append_detection,
    upload_to_s3,
    insert_detection,
    extract_date_label_from_frame,
    get_ocr_reader
)

def process_video():
    """Main video processing routine with OCR date label extraction."""
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / OUTPUT_FILENAME

    # Setup database
    try:
        setup_database_and_table()
    except Exception as e:
        print(f"[WARN] Database setup failed: {e}. Continuing without DB.", file=sys.stderr)

    # Initialize OCR reader (lazy initialization)
    try:
        ocr_reader = get_ocr_reader()
        if ocr_reader:
            print("[INFO] OCR reader initialized for date label extraction")
        else:
            print("[WARN] OCR reader not available, date labels will not be extracted")
    except Exception as e:
        print(f"[WARN] Failed to initialize OCR: {e}", file=sys.stderr)

    # Load models
    detector, classifier, poser = load_models()
    if detector is None or classifier is None or poser is None:
        return 3, 0, 0, out_path

    # Open video
    cap, temp_path = open_video_from_source(HARDCODED_VIDEO_URL)
    if cap is None:
        print(f"[ERROR] Could not open video from URL: {HARDCODED_VIDEO_URL}", file=sys.stderr)
        return 2, 0, 0, out_path

    try:
        # Get metadata and compute samples
        fps, duration_sec = get_video_meta(cap)
        if duration_sec <= 0:
            print("[WARN] Could not determine video duration. Sampling only the first frame.", file=sys.stderr)
            sample_times = [0.0]
        else:
            sample_times = compute_sample_times(duration_sec, 10.0)  # 10-second intervals

        total_frames_sampled = 0
        total_detections_written = 0

        # Prepare output CSV
        write_csv_header(out_path)

        for idx, t in enumerate(sample_times):
            ok, frame = seek_and_read_frame(cap, t)
            if not ok or frame is None:
                print(f"[WARN] Failed to read frame at ~{t:.2f}s; skipping.", file=sys.stderr)
                continue

            total_frames_sampled += 1

            # Extract OCR date label from frame
            # Compute frame_index from time and fps for in_video_time calculation
            # in_video_time will be: frame_index / fps
            # Since we're sampling at specific times, we can compute approximate frame_index
            frame_index = int(round(t * fps)) if fps > 0 else None
            ocr_date_text, in_video_time = extract_date_label_from_frame(
                frame, 
                frame_index=frame_index, 
                fps=fps
            )
            
            if ocr_date_text:
                print(f"[INFO] Frame at {t:.2f}s: OCR extracted '{ocr_date_text}', in_video_time={in_video_time}")
            else:
                print(f"[DEBUG] Frame at {t:.2f}s: No OCR text extracted")

            # Run detection and processing
            try:
                results = detector(frame, verbose=False)
                detections = process_detection(results, classifier, poser, frame)

                if not detections:
                    continue

                # Upload frame to S3 (only if we have detections)
                s3_url = ""
                tmp_img = save_frame_temp_jpg(frame)
                if tmp_img is not None:
                    try:
                        timestamp_ms = int(round(t * 1000.0))
                        object_key = f"{S3_FRAMES_PREFIX}frame_{timestamp_ms}.jpg"
                        s3_url = upload_to_s3(tmp_img, S3_BUCKET, object_key)
                    finally:
                        tmp_img.unlink()

                # Write detections
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cls_label, cls_conf = det['classification']
                    pose_status, num_keypoints = det['pose']

                    # Write to CSV
                    append_detection(
                        out_path, t, det['label'], x1, y1, x2, y2, det['confidence'],
                        cls_label, cls_conf, pose_status, num_keypoints, s3_url
                    )

                    # Write to database with in_video_time and ocr_date_text
                    insert_detection(
                        t, det['label'], x1, y1, x2, y2, det['confidence'],
                        cls_label, cls_conf, pose_status, num_keypoints, s3_url,
                        HARDCODED_VIDEO_URL,
                        in_video_time=in_video_time,
                        ocr_date_text=ocr_date_text
                    )

                    total_detections_written += 1
                    print(f"[INFO] Detected {det['label']} at ~{t:.2f}s with confidence {det['confidence']:.4f}")

            except Exception as e:
                print(f"[WARN] Failed to process frame at ~{t:.2f}s: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue

    finally:
        cap.release()
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                print(f"[INFO] Cleaned up temporary file: {temp_path}")
            except Exception as e:
                print(f"[WARN] Failed to remove temp file {temp_path}: {e}", file=sys.stderr)

    print(f"[INFO] total_frames_sampled={total_frames_sampled}")
    print(f"[INFO] total_detections_written={total_detections_written}")
    print(f"[INFO] output_path={out_path}")

    # Upload CSV to S3
    try:
        csv_s3_key = f"{S3_CSV_PREFIX}{OUTPUT_FILENAME}"
        csv_url = upload_to_s3(out_path, S3_BUCKET, csv_s3_key)
        if csv_url:
            print(f"[INFO] CSV uploaded to S3: {csv_url}")
        else:
            print("[WARN] CSV upload to S3 failed")
    except Exception as e:
        print(f"[WARN] Failed to upload CSV to S3: {e}", file=sys.stderr)

    return 0, total_frames_sampled, total_detections_written, out_path

def main():
    """CLI entrypoint with robust error handling."""
    # Load environment variables
    env_path = Path(__file__).resolve().parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"[INFO] Loaded environment variables from: {env_path}")

    try:
        code, _, _, _ = process_video()
        sys.exit(code)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()