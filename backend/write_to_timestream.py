#!/usr/bin/env python3
"""
write_to_timestream.py

Utility module and CLI for writing bear activity observations to AWS Timestream
and previewing insert payloads from a CSV file.

Environment:
- AWS credentials are resolved by boto3 standard chain (env, shared config, IAM role).
- Region: via AWS_REGION env var, or defaults to "us-east-1".

Constants:
- DB_NAME = "metropark_bearsDB"
- TABLE_NAME = "monitor_bears_tbl"

Features:
- get_timestream_client(): Returns a cached boto3 Timestream Write client.
- insert_bear_activity(...): Build and write a single MULTI-measure record.
- print_inserts_from_csv(csv_path): Read CSV and print the JSON payload that
  would be sent to Timestream (dry-run preview, no writes).

CSV columns (case-insensitive):
  bear_id, timestamp, activity, object_label, bounding_box, timeInSeconds

Time handling:
- TimeUnit: NANOSECONDS
- If timestamp is numeric:
    * If >= 1e12 => treat as milliseconds
    * Else => treat as seconds
  Then convert to nanoseconds.
- Else try parse ISO8601; on failure, use current time.
"""

import os
import sys
import csv
import json
import time
import logging
from typing import Any, Dict, Optional, Tuple

# Optional dependency: dateutil for ISO8601 parsing; fallback if not available
try:
    from dateutil import parser as dateutil_parser  # type: ignore
    _DATEUTIL_AVAILABLE = True
except Exception:
    _DATEUTIL_AVAILABLE = False

try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore

# Constants for Timestream
DB_NAME = "metropark_bearsDB"
TABLE_NAME = "monitor_bears_tbl"

# Cached client
_TS_WRITE_CLIENT: Optional[Any] = None

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("write_to_timestream")


def _get_region() -> str:
    """Resolve AWS region from environment with default."""
    return os.environ.get("AWS_REGION", "us-east-1")


def _now_nanos() -> str:
    """Return current time in epoch nanoseconds as string."""
    return str(int(time.time() * 1_000_000_000))


def _parse_timestamp_to_nanos(ts: str) -> str:
    """
    Parse various timestamp formats into epoch nanoseconds (string).

    Logic:
    - If ts can be parsed as integer:
        - If >= 1e12, treat as milliseconds.
        - Else, treat as seconds.
    - Else if ISO8601 (via dateutil if available, else fromisoformat):
        convert to epoch seconds then to nanos.
    - On failure, use current time.

    Returns:
        str: epoch nanoseconds
    """
    if ts is None or str(ts).strip() == "":
        return _now_nanos()

    s = str(ts).strip()

    # Try numeric fast-path
    try:
        ival = int(float(s))
        # Heuristic: millis if very large
        if ival >= 1_000_000_000_000:  # 1e12
            nanos = ival * 1_000_000
        else:
            nanos = ival * 1_000_000_000
        return str(nanos)
    except Exception:
        pass

    # Try ISO8601
    try:
        if _DATEUTIL_AVAILABLE:
            dt = dateutil_parser.isoparse(s)  # type: ignore
        else:
            # Simple fallback using Python's fromisoformat (may not support all variants)
            from datetime import datetime
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        epoch = dt.timestamp()  # seconds as float
        nanos = int(epoch * 1_000_000_000)
        return str(nanos)
    except Exception:
        logger.warning("Failed to parse timestamp '%s'; using current time.", s)
        return _now_nanos()


def _parse_time_in_seconds(value: str) -> Tuple[str, str]:
    """
    Parse timeInSeconds into (type, value_string).

    Attempts float parsing; if the float is integral, uses BIGINT, else DOUBLE.
    If parsing fails, uses DOUBLE with 0.0.

    Returns:
        (measure_type, value_str)
        where measure_type in {"DOUBLE", "BIGINT"}
    """
    try:
        f = float(value)
        if f.is_integer():
            return "BIGINT", str(int(f))
        return "DOUBLE", str(f)
    except Exception:
        logger.debug("timeInSeconds parse failed for '%s'; defaulting to DOUBLE=0.0", value)
        return "DOUBLE", "0.0"


def _normalize_headers(headers: list) -> Dict[str, int]:
    """
    Build a case-insensitive header index mapping for CSV.

    Returns:
        dict: lower_header -> index
    """
    idx: Dict[str, int] = {}
    for i, h in enumerate(headers):
        if h is None:
            continue
        idx[h.strip().lower()] = i
    return idx


# PUBLIC_INTERFACE
def get_timestream_client() -> Any:
    """
    Return a cached AWS Timestream Write client.

    Relies on boto3 credentials resolution and AWS_REGION environment variable.

    Returns:
        boto3.client('timestream-write')

    Raises:
        RuntimeError: if boto3 is not available.
    """
    global _TS_WRITE_CLIENT
    if _TS_WRITE_CLIENT is not None:
        return _TS_WRITE_CLIENT
    if boto3 is None:
        raise RuntimeError("boto3 is not available. Please install boto3 to use Timestream features.")
    region = _get_region()
    _TS_WRITE_CLIENT = boto3.client("timestream-write", region_name=region)
    return _TS_WRITE_CLIENT


def _build_record(
    bear_id: str,
    timestamp: str,
    activity: str,
    object_label: str,
    bounding_box: str,
    timeInSeconds: str,
) -> Dict[str, Any]:
    """
    Construct a Timestream Write record with MULTI measure and needed dimensions.
    """
    time_nanos = _parse_timestamp_to_nanos(timestamp)
    mv_type, mv_time_val = _parse_time_in_seconds(timeInSeconds)

    record: Dict[str, Any] = {
        "Dimensions": [
            {"Name": "bear_id", "Value": str(bear_id)},
            {"Name": "activity", "Value": str(activity)},
            {"Name": "object_label", "Value": str(object_label)},
        ],
        "MeasureName": "bear_observation",
        "MeasureValueType": "MULTI",
        "MeasureValues": [
            {"Name": "bounding_box", "Type": "VARCHAR", "Value": str(bounding_box)},
            {"Name": "timeInSeconds", "Type": mv_type, "Value": mv_time_val},
        ],
        "Time": time_nanos,
        "TimeUnit": "NANOSECONDS",
    }
    return record


# PUBLIC_INTERFACE
def insert_bear_activity(
    bear_id: str,
    timestamp: str,
    activity: str,
    object_label: str,
    bounding_box: str,
    timeInSeconds: str,
) -> Dict[str, Any]:
    """
    Insert one bear observation record into AWS Timestream.

    Args:
        bear_id: Bear identifier (dimension).
        timestamp: Observation time (numeric epoch seconds/millis or ISO8601).
        activity: Bear activity (dimension).
        object_label: Detected object label (dimension).
        bounding_box: Bounding box string (included as VARCHAR measure).
        timeInSeconds: Time in seconds (parsed as DOUBLE or BIGINT measure).

    Returns:
        dict: Response from Timestream Write or error information.

    Notes:
        - Uses DatabaseName=metropark_bearsDB, TableName=monitor_bears_tbl.
        - TimeUnit is NANOSECONDS.
    """
    record = _build_record(bear_id, timestamp, activity, object_label, bounding_box, timeInSeconds)
    client = get_timestream_client()

    common_attributes = {
        # Common attributes can be used to set shared dimensions or timestamps if needed.
        # Here we leave it empty; records are explicit.
    }

    try:
        resp = client.write_records(
            DatabaseName=DB_NAME,
            TableName=TABLE_NAME,
            Records=[record],
            CommonAttributes=common_attributes,  # type: ignore
        )
        logger.info("Successfully wrote record to Timestream. Status: %s", resp.get("ResponseMetadata", {}).get("HTTPStatusCode"))
        return {"ok": True, "response": resp}
    except (BotoCoreError, ClientError) as e:  # type: ignore
        logger.exception("AWS Timestream write_records failed: %s", e)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.exception("Unexpected error writing to Timestream: %s", e)
        return {"ok": False, "error": str(e)}


# PUBLIC_INTERFACE
def print_inserts_from_csv(csv_path: str) -> None:
    """
    Read a CSV and print the JSON payloads that would be sent to Timestream.

    This function is a safe dry-run preview and does not perform any writes.

    CSV headers are case-insensitive and should include:
      bear_id, timestamp, activity, object_label, bounding_box, timeInSeconds

    Skips rows missing required fields (bear_id, timestamp) with a warning.
    """
    if not os.path.exists(csv_path) or not os.path.isfile(csv_path):
        logger.error("CSV file not found: %s", csv_path)
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            logger.error("CSV file is empty: %s", csv_path)
            return

        hidx = _normalize_headers(headers)

        def _get(row: list, key: str) -> str:
            i = hidx.get(key, -1)
            return row[i].strip() if 0 <= i < len(row) and row[i] is not None else ""

        row_num = 1  # header read; start counting data rows from 2
        for row in reader:
            row_num += 1
            bear_id = _get(row, "bear_id")
            timestamp = _get(row, "timestamp")
            activity = _get(row, "activity")
            object_label = _get(row, "object_label")
            bounding_box = _get(row, "bounding_box")
            time_in_seconds = _get(row, "timeinseconds") or _get(row, "time_in_seconds")

            if not bear_id or not timestamp:
                logger.warning("Skipping row %d: missing required 'bear_id' or 'timestamp'", row_num)
                continue

            record = _build_record(bear_id, timestamp, activity, object_label, bounding_box, time_in_seconds)
            payload = {
                "DatabaseName": DB_NAME,
                "TableName": TABLE_NAME,
                "Records": [record],
            }
            print(json.dumps(payload, indent=2))


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="AWS Timestream writer utility for bear observations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Preview inserts from CSV (no writes):
    python write_to_timestream.py --csv ./observations.csv

  Insert a single record:
    python write_to_timestream.py --insert \\
      --bear_id B001 \\
      --timestamp "2024-10-01T12:00:00Z" \\
      --activity "walking" \\
      --object_label "bear" \\
      --bounding_box "10,20,100,200" \\
      --timeInSeconds "3.5"
        """,
    )

    parser.add_argument("--csv", help="Path to CSV file to preview insert payloads (no writes).")

    insert = parser.add_argument_group("insert")
    insert.add_argument("--insert", action="store_true", help="Insert a single record to Timestream.")
    insert.add_argument("--bear_id", type=str, help="Bear ID (dimension).")
    insert.add_argument("--timestamp", type=str, help="Timestamp (epoch seconds/millis or ISO8601).")
    insert.add_argument("--activity", type=str, default="", help="Activity (dimension).")
    insert.add_argument("--object_label", type=str, default="", help="Object label (dimension).")
    insert.add_argument("--bounding_box", type=str, default="", help="Bounding box string.")
    insert.add_argument("--timeInSeconds", type=str, default="0", help="Time in seconds value.")

    return parser


def main():
    """
    CLI entrypoint:
    - --csv <path>: Print payloads for each CSV row (no writes).
    - --insert with named args: Perform a single write and print response.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.csv:
        print_inserts_from_csv(args.csv)
        return

    if args.insert:
        missing = []
        if not args.bear_id:
            missing.append("--bear_id")
        if not args.timestamp:
            missing.append("--timestamp")
        if missing:
            logger.error("Missing required arguments for insert: %s", ", ".join(missing))
            sys.exit(2)

        result = insert_bear_activity(
            bear_id=args.bear_id,
            timestamp=args.timestamp,
            activity=args.activity or "",
            object_label=args.object_label or "",
            bounding_box=args.bounding_box or "",
            timeInSeconds=args.timeInSeconds or "0",
        )
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("ok") else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
