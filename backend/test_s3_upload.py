#!/usr/bin/env python3
"""
Simple S3 upload test script.

This script uploads a local image file named 'bala_test.png' (expected to be in the same directory)
to a hardcoded S3 bucket and key using boto3 with hardcoded credentials and region.

Usage:
  python test_s3_upload.py

Notes:
  - This is a test utility that uses hardcoded values for credentials and S3 details.
  - It prints informative logs before and after the upload.
  - It returns/prints the final S3 object URL (constructed from bucket, region, and key).
  - Basic exceptions are handled and printed to stdout.
"""

import os
import sys
import traceback
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# =========================
# Hardcoded configuration
# =========================
# Replace these placeholder values with actual test credentials and details.
AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_ACCESS_KEY"
AWS_REGION = "us-east-1"  # e.g., "us-east-1"
S3_BUCKET_NAME = "your-bucket-name"
S3_OBJECT_KEY = "test/bala_test.png"  # Destination key in the bucket

# Local file expected to be present in the same folder as this script
LOCAL_FILENAME = "bala_test.png"


def _construct_s3_url(bucket: str, region: str, key: str) -> str:
    """
    Construct a public-style S3 URL based on bucket, region, and object key.
    Note: This does not guarantee the object is publicly accessible. It's a convenience URL.

    For us-east-1, the URL typically omits the region in the hostname.
    For other regions, the region is included.
    """
    if region == "us-east-1":
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


# PUBLIC_INTERFACE
def upload_test_image() -> Optional[str]:
    """Upload the local bala_test.png to the configured S3 bucket and return the constructed S3 URL.

    Returns:
        Optional[str]: Constructed S3 URL on success, or None on failure.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, LOCAL_FILENAME)

    print("[INFO] Starting S3 upload test")
    print(f"[INFO] Local file expected at: {local_path}")
    print(f"[INFO] Target bucket: {S3_BUCKET_NAME}")
    print(f"[INFO] Target key: {S3_OBJECT_KEY}")
    print(f"[INFO] Region: {AWS_REGION}")

    if not os.path.isfile(local_path):
        print(f"[ERROR] Local file not found: {local_path}")
        return None

    try:
        print("[INFO] Creating S3 client...")
        s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        print("[INFO] Uploading file to S3...")
        s3_client.upload_file(local_path, S3_BUCKET_NAME, S3_OBJECT_KEY)
        url = _construct_s3_url(S3_BUCKET_NAME, AWS_REGION, S3_OBJECT_KEY)
        print("[SUCCESS] Upload complete.")
        print(f"[INFO] Object URL (may require appropriate ACL/permissions): {url}")
        return url
    except (BotoCoreError, ClientError) as e:
        print("[ERROR] boto3/Botocore error during upload.")
        print(str(e))
        # Optionally print more details if available
        exc_info = getattr(e, "response", None)
        if exc_info:
            print(f"[ERROR] Service response: {exc_info}")
        return None
    except Exception as e:
        print("[ERROR] Unexpected error during upload.")
        print(str(e))
        traceback.print_exc()
        return None


def main() -> None:
    """Entry point for the script which triggers the upload and prints the resulting URL."""
    result = upload_test_image()
    if result:
        print(f"[RESULT] Uploaded file URL: {result}")
        sys.exit(0)
    else:
        print("[RESULT] Upload failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
