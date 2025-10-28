"""Configuration settings for the YOLO pipeline."""

import os

# Sampling Configuration
#SAMPLE_INTERVAL_SECONDS = 10.0
SAMPLE_INTERVAL_SECONDS = 2.0

# Model Weights
DETECTION_WEIGHTS = "yolov8m.pt"
CLASSIFICATION_WEIGHTS = "yolov8m-cls.pt"
POSE_WEIGHTS = "yolov8m-pose.pt"

# Output Configuration
OUTPUT_FILENAME = "detections_Video_2_A02_output_000.csv"

# S3 Configuration
S3_BUCKET = "humanlabelimg-poc"
#S3_VIDEO_KEY = "2.mp4"
#S3_VIDEO_KEY = "output_anteater_clip.mp4"
#S3_VIDEO_KEY = "output_anteater_human_clip.mp4"
S3_VIDEO_KEY = "Video_2_A02/Video_2_A02_output_000.mp4"
S3_FRAMES_PREFIX = "processed_images_from_video/op_giant_ant_eater/Video_2_A02/Video_2_A02_output_000/Video_2_A02_output_000"
S3_CSV_PREFIX = "output_csvs/Video_2_A02/Video_2_A02_output_000"

# Video URL
#HARDCODED_VIDEO_URL = f"https://{S3_BUCKET}.s3.us-east-2.amazonaws.com/{S3_VIDEO_KEY}"
HARDCODED_VIDEO_URL = f"https://{S3_BUCKET}.s3.us-east-2.amazonaws.com/giant_ant_eater/video_for_training/totrain/{S3_VIDEO_KEY}"

def get_db_env():
    """Load DB connection settings from environment."""
    return {
        'host': os.environ.get("DB_HOST", ""),
        'port': int(os.environ.get("DB_PORT", "3306")),
        'user': os.environ.get("DB_USER", ""),
        'password': os.environ.get("DB_PASSWORD", ""),
        'db_name': os.environ.get("DB_NAME", "")
    }

def get_labels():
    # Surveillance Zone Polygons
    # Using BOX-A, BOX-B, BOX-C, BOX-D, BOX-E, BOX-F naming convention

    # BOX-A (Left area)
    BOX_A = [
        {"x": 138.64, "y": 315.36},
        {"x": 408.75, "y": 224.53},
        {"x": 653.76, "y": 514.95},
        {"x": 299.99, "y": 695.42}
    ]

    # BOX-B (Upper center area)
    BOX_B = [
        {"x": 780.45, "y": 118.15},
        {"x": 452.97, "y": 347.63},
        {"x": 653.76, "y": 518.54},
        {"x": 1024.26, "y": 332.09}
    ]

    # BOX-C (Center-left area)
    BOX_C = [
        {"x": 653.76, "y": 514.95},
        {"x": 871.28, "y": 702.59},
        {"x": 412.33, "y": 940.43},
        {"x": 298.79, "y": 690.64}
    ]

    # BOX-D (Middle right area)
    BOX_D = [
        {"x": 657.34, "y": 516.15},
        {"x": 1051.75, "y": 304.60},
        {"x": 1288.39, "y": 452.80},
        {"x": 870.08, "y": 694.23}
    ]

    # BOX-E (Bottom right area)
    BOX_E = [
        {"x": 868.89, "y": 703.79},
        {"x": 234.25, "y": 1004.97},
        {"x": 812.72, "y": 1005.99},  
        {"x": 1238.20, "y": 1004.97}
    ]

    # BOX-F (Large right area)
    BOX_F = [
        {"x": 866.50, "y": 699.01},
        {"x": 1354.13, "y": 371.53},
        {"x": 1724.63, "y": 622.52},
        {"x": 1234.61, "y": 1004.97}
    ]

    # DATE (Timestamp area - top-left rectangle)
    DATE = [
        {"x": 29.88, "y": 29.71},
        {"x": 523.48, "y": 29.71},
        {"x": 523.48, "y": 113.37},
        {"x": 29.88, "y": 113.37}
    ]

    # Combined zones dictionary (for easy iteration)
    zones = {
        "BOX-A": BOX_A,
        "BOX-B": BOX_B,
        "BOX-C": BOX_C,
        "BOX-D": BOX_D,
        "BOX-E": BOX_E,
        "BOX-F": BOX_F,
        "DATE": DATE
    }
    
    return zones
