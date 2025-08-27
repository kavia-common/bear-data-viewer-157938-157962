"""
bear_data_service.py

Shared business logic module that can be used by both Flask app and Lambda handler.
This module contains the core bear data logic extracted from the original Flask implementation.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List


# PUBLIC_INTERFACE
def get_bear_records() -> List[Dict[str, Any]]:
    """
    Core business logic for generating bear data records.
    This function contains the actual bear data logic that was in the Flask route.
    
    Returns:
        list[dict]: A list of bear records with bearId, pose, and timestamp fields.
    """
    now = datetime.now(timezone.utc)
    
    # Core bear data logic - this is the same logic from the original Flask app
    data = [
        {"bearId": "B001", "pose": "Sitting", "timestamp": (now - timedelta(seconds=5))},
        {"bearId": "B002", "pose": "Standing", "timestamp": (now - timedelta(seconds=15))},
        {"bearId": "B003", "pose": "Walking", "timestamp": (now - timedelta(seconds=25))},
    ]
    
    return data


# PUBLIC_INTERFACE  
def format_bear_records_for_api(records: List[Dict[str, Any]], format_timestamps: bool = True) -> List[Dict[str, Any]]:
    """
    Format bear records for API response.
    
    Args:
        records: List of bear records from get_bear_records()
        format_timestamps: If True, convert datetime objects to ISO 8601 strings
        
    Returns:
        list[dict]: Formatted bear records ready for JSON serialization
    """
    if not format_timestamps:
        return records
    
    formatted_records = []
    for record in records:
        formatted_record = record.copy()
        if 'timestamp' in formatted_record and hasattr(formatted_record['timestamp'], 'isoformat'):
            formatted_record['timestamp'] = formatted_record['timestamp'].isoformat()
        formatted_records.append(formatted_record)
    
    return formatted_records


# PUBLIC_INTERFACE
def get_bear_data_for_flask() -> List[Dict[str, Any]]:
    """
    Get bear data formatted for Flask app (with datetime objects).
    Flask-Smorest/Marshmallow will handle the datetime serialization.
    
    Returns:
        list[dict]: Bear records with datetime objects for Flask/Marshmallow
    """
    return get_bear_records()


# PUBLIC_INTERFACE
def get_bear_data_for_lambda() -> List[Dict[str, Any]]:
    """
    Get bear data formatted for Lambda response (with ISO 8601 strings).
    Lambda needs pre-serialized data for JSON response.
    
    Returns:
        list[dict]: Bear records with ISO 8601 timestamp strings for Lambda
    """
    records = get_bear_records()
    return format_bear_records_for_api(records, format_timestamps=True)
