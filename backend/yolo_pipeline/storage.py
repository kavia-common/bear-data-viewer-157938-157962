"""S3 storage operations for the YOLO pipeline."""

import os
import sys
from pathlib import Path

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    print("[WARN] boto3 not available", file=sys.stderr)
    boto3 = None
    BOTO3_AVAILABLE = False

def get_s3_client():
    """Create a boto3 S3 client using environment variables."""
    if not BOTO3_AVAILABLE:
        return None
    
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")

    try:
        session_kwargs = {}
        if aws_region:
            session_kwargs["region_name"] = aws_region
        session = boto3.session.Session(**session_kwargs)

        client_kwargs = {}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        return session.client("s3", **client_kwargs)
    except Exception as e:
        print(f"[WARN] Failed to create boto3 S3 client: {e}", file=sys.stderr)
        return None

def upload_to_s3(file_path: Path, bucket: str, key: str) -> str:
    """Upload file to S3 and return a public URL."""
    client = get_s3_client()
    if not client:
        return ""

    try:
        client.upload_file(str(file_path), bucket, key)
    except Exception as e:
        print(f"[WARN] S3 upload failed: {e}", file=sys.stderr)
        return ""

    # Build public URL
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
    if region == "us-east-1":
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"