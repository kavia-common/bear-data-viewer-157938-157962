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


if __name__ == "__main__":
    try:
        # Upload first, then download
        upload_sample_file_to_s3()
        download_sample_file_from_s3()
        print("Success")
    except Exception as exc:
        # Let existing function's behavior surface; also print a simple error note here.
        print(f"Error: {exc}")
        raise
