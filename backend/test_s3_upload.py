#!/usr/bin/env python3
"""
Simple S3 upload test script.

This script uploads a local image file named 'bala_test.png' (expected to be in the same directory)
to a specified S3 bucket and key using boto3 with credentials sourced from environment variables.

Usage:
  - As a script: python test_s3_upload.py
  - As a pytest test: will be skipped if AWS credentials are not present in env.

Environment variables used:
  - AWS_ACCESS_KEY_ID (required for running)
  - AWS_SECRET_ACCESS_KEY (required for running)
  - AWS_DEFAULT_REGION (optional; defaults to 'us-east-1' if not set)
  - AWS_SESSION_TOKEN (optional; used if present)
  - TEST_S3_BUCKET_NAME (optional; if set, used as bucket name)
  - TEST_S3_OBJECT_KEY (optional; if set, used as object key)

Notes:
  - No hardcoded secrets. Credentials are read from environment variables.
  - Skips the pytest-based run if required credentials are missing.
  - Prints informative logs and returns the final S3 object URL (constructed from bucket, region, and key).
"""

import os
import sys
import traceback
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

try:
    import pytest  # optional dependency in test context
except Exception:  # pragma: no cover - script mode may not have pytest
    pytest = None  # type: ignore

# =========================
# Configuration (from env)
# =========================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")  # optional

# Allow overriding bucket/key via env for flexibility in CI
S3_BUCKET_NAME = os.getenv("TEST_S3_BUCKET_NAME", "your-bucket-name")
S3_OBJECT_KEY = os.getenv("TEST_S3_OBJECT_KEY", "test/bala_test.png")

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

    Behavior:
        - Reads AWS credentials from environment variables.
        - If credentials are missing and running under pytest, the test is skipped with a clear message.
        - If run as a standalone script and credentials are missing, prints an error and returns None.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, LOCAL_FILENAME)

    missing = []
    if not AWS_ACCESS_KEY_ID:
        missing.append("AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_ACCESS_KEY:
        missing.append("AWS_SECRET_ACCESS_KEY")

    if missing:
        msg = f"Missing required AWS environment variables: {', '.join(missing)}"
        if pytest is not None:
            pytest.skip(msg)
            return None  # pragma: no cover (pytest.skip raises, but keep for safety)
        else:
            print(f"[SKIP] {msg}")
            return None

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
        # Build kwargs for boto3 client based on env
        client_kwargs = {
            "service_name": "s3",
            "region_name": AWS_REGION,
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
        }
        if AWS_SESSION_TOKEN:
            client_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

        s3_client = boto3.client(**client_kwargs)

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
        # Non-fatal if skipped due to missing env vars; return code 0 with message.
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            print("[RESULT] Skipped upload: missing AWS credentials in environment.")
            sys.exit(0)
        print("[RESULT] Upload failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
