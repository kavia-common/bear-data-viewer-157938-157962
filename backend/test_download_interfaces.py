import os
import sys
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError


# PUBLIC_INTERFACE
def download_sample_file_from_s3(
    bucket_name: str = "humanlabelimg-poc",
    object_key: str = "samplefiledownload.txt",
    local_filename: Optional[str] = None,
) -> None:
    """Download a sample file from S3 using credentials from environment variables.

    Reads AWS credentials from environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION (optional, defaults to 'us-east-1')

    Parameters:
    - bucket_name: S3 bucket name to download from. Defaults to 'humanlabelimg-poc'.
    - object_key: S3 object key to download. Defaults to 'samplefiledownload.txt'.
    - local_filename: Local destination filename. Defaults to same as object_key.

    Behavior:
    - Prints a success message on completion.
    - Prints a helpful error message on failure.
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")
    dest = local_filename or object_key

    if not access_key or not secret_key:
        print("Error: Missing AWS credentials. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set in the environment.")
        return

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        s3_client.download_file(bucket_name, object_key, dest)
        print(f"Downloaded '{object_key}' from bucket '{bucket_name}' to local file '{dest}'.")
    except NoCredentialsError:
        print("Error: AWS credentials not found by boto3. Check environment variables.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        msg = e.response.get("Error", {}).get("Message", str(e))
        print(f"S3 ClientError [{code}]: {msg}")
    except BotoCoreError as e:
        print(f"AWS SDK error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)


if __name__ == "__main__":
    # Minimal entry point: no argument parsing, just attempt the download.
    download_sample_file_from_s3()
