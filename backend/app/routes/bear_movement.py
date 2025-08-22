from flask_smorest import Blueprint
from flask.views import MethodView
from marshmallow import Schema, fields
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
import os

# Import the database model
try:
    from identify_bear_movement import BearMovementDetection, Config
    MOVEMENT_DETECTION_AVAILABLE = True
except ImportError:
    MOVEMENT_DETECTION_AVAILABLE = False
    BearMovementDetection = None

blp = Blueprint(
    "BearMovement",
    "bear_movement",
    url_prefix="/api",
    description="Endpoints for bear movement detection data."
)


class BearMovementSchema(Schema):
    """Schema representing a Bear Movement Detection record returned by the API."""
    id = fields.Integer(required=True, description="Unique detection ID")
    frame_number = fields.Integer(required=True, description="Video frame number")
    frame_timestamp = fields.DateTime(
        required=True,
        format="iso",
        description="ISO 8601 timestamp (UTC) when frame was processed"
    )
    video_timestamp_sec = fields.Float(required=True, description="Timestamp within the video in seconds")
    bear_id = fields.String(required=True, description="Bear identifier")
    bear_x1 = fields.Float(allow_none=True, description="Bear bounding box x1 coordinate")
    bear_y1 = fields.Float(allow_none=True, description="Bear bounding box y1 coordinate")
    bear_x2 = fields.Float(allow_none=True, description="Bear bounding box x2 coordinate")
    bear_y2 = fields.Float(allow_none=True, description="Bear bounding box y2 coordinate")
    bear_confidence = fields.Float(allow_none=True, description="Bear detection confidence score")
    movement_state = fields.String(required=True, description="Movement state: moving, resting, or unknown")
    movement_confidence = fields.Float(required=True, description="Movement analysis confidence score")
    distance_moved = fields.Float(allow_none=True, description="Distance moved since previous frame in pixels")
    static_objects = fields.Raw(allow_none=True, description="JSON array of static object detections")
    processing_timestamp = fields.DateTime(
        required=True,
        format="iso",
        description="ISO 8601 timestamp (UTC) when detection was processed"
    )


class BearMovementSummarySchema(Schema):
    """Schema for movement summary statistics."""
    total_detections = fields.Integer(required=True, description="Total number of detections")
    moving_count = fields.Integer(required=True, description="Number of moving state detections")
    resting_count = fields.Integer(required=True, description="Number of resting state detections")
    unknown_count = fields.Integer(required=True, description="Number of unknown state detections")
    avg_confidence = fields.Float(required=True, description="Average movement confidence")
    latest_timestamp = fields.DateTime(
        allow_none=True,
        format="iso",
        description="Timestamp of most recent detection"
    )


def _get_db_session():
    """Create database session for movement detection data."""
    if not MOVEMENT_DETECTION_AVAILABLE:
        return None
    
    try:
        # Use the same database URL as the movement detection system
        database_url = os.environ.get("DATABASE_URL") or Config.DATABASE_URL
        engine = create_engine(database_url, future=True)
        SessionLocal = sessionmaker(bind=engine, future=True)
        return SessionLocal()
    except Exception:
        return None


@blp.route("/bear-movements")
class BearMovementList(MethodView):
    """Provide access to bear movement detection data."""

    # PUBLIC_INTERFACE
    def get(self):
        """
        Returns a list of recent bear movement detection records.
        
        Query Parameters:
        - limit: Number of records to return (default: 50, max: 500)
        - hours: Number of hours back to query (default: 24)
        - movement_state: Filter by movement state (moving, resting, unknown)
        
        Returns:
            list[dict]: A list of bear movement detection records.
        """
        if not MOVEMENT_DETECTION_AVAILABLE:
            return {
                "error": "Bear movement detection system not available",
                "message": "The movement detection module is not installed or configured"
            }, 503
        
        session = _get_db_session()
        if not session:
            return {
                "error": "Database connection failed",
                "message": "Unable to connect to movement detection database"
            }, 503
        
        try:
            # Parse query parameters
            from flask import request
            limit = min(int(request.args.get('limit', 50)), 500)
            hours = int(request.args.get('hours', 24))
            movement_state = request.args.get('movement_state')
            
            # Calculate time threshold
            time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Build query
            query = session.query(BearMovementDetection).filter(
                BearMovementDetection.frame_timestamp >= time_threshold
            )
            
            if movement_state:
                query = query.filter(BearMovementDetection.movement_state == movement_state)
            
            # Order by timestamp (most recent first) and limit
            detections = query.order_by(desc(BearMovementDetection.frame_timestamp)).limit(limit).all()
            
            # Convert to dict format
            results = []
            for detection in detections:
                result = {
                    "id": detection.id,
                    "frame_number": detection.frame_number,
                    "frame_timestamp": detection.frame_timestamp,
                    "video_timestamp_sec": detection.video_timestamp_sec,
                    "bear_id": detection.bear_id,
                    "bear_x1": detection.bear_x1,
                    "bear_y1": detection.bear_y1,
                    "bear_x2": detection.bear_x2,
                    "bear_y2": detection.bear_y2,
                    "bear_confidence": detection.bear_confidence,
                    "movement_state": detection.movement_state,
                    "movement_confidence": detection.movement_confidence,
                    "distance_moved": detection.distance_moved,
                    "static_objects": detection.static_objects,
                    "processing_timestamp": detection.processing_timestamp,
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            return {
                "error": "Query failed",
                "message": str(e)
            }, 500
        finally:
            session.close()

    # Document response schema via flask-smorest
    get = blp.response(
        200,
        BearMovementSchema(many=True),
        description="List of bear movement detection records",
    )(get)


@blp.route("/bear-movements/summary")
class BearMovementSummary(MethodView):
    """Provide summary statistics for bear movement data."""

    # PUBLIC_INTERFACE
    def get(self):
        """
        Returns summary statistics for bear movement detection data.
        
        Query Parameters:
        - hours: Number of hours back to analyze (default: 24)
        
        Returns:
            dict: Summary statistics including counts and averages.
        """
        if not MOVEMENT_DETECTION_AVAILABLE:
            return {
                "error": "Bear movement detection system not available",
                "message": "The movement detection module is not installed or configured"
            }, 503
        
        session = _get_db_session()
        if not session:
            return {
                "error": "Database connection failed",
                "message": "Unable to connect to movement detection database"
            }, 503
        
        try:
            from flask import request
            from sqlalchemy import func
            
            hours = int(request.args.get('hours', 24))
            time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Get summary statistics
            query = session.query(BearMovementDetection).filter(
                BearMovementDetection.frame_timestamp >= time_threshold
            )
            
            total_detections = query.count()
            moving_count = query.filter(BearMovementDetection.movement_state == 'moving').count()
            resting_count = query.filter(BearMovementDetection.movement_state == 'resting').count()
            unknown_count = query.filter(BearMovementDetection.movement_state == 'unknown').count()
            
            # Calculate average confidence
            avg_confidence_result = session.query(
                func.avg(BearMovementDetection.movement_confidence)
            ).filter(
                BearMovementDetection.frame_timestamp >= time_threshold
            ).scalar()
            
            avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0
            
            # Get latest timestamp
            latest_detection = query.order_by(desc(BearMovementDetection.frame_timestamp)).first()
            latest_timestamp = latest_detection.frame_timestamp if latest_detection else None
            
            return {
                "total_detections": total_detections,
                "moving_count": moving_count,
                "resting_count": resting_count,
                "unknown_count": unknown_count,
                "avg_confidence": round(avg_confidence, 3),
                "latest_timestamp": latest_timestamp,
            }
            
        except Exception as e:
            return {
                "error": "Summary query failed",
                "message": str(e)
            }, 500
        finally:
            session.close()

    # Document response schema via flask-smorest
    get = blp.response(
        200,
        BearMovementSummarySchema,
        description="Bear movement detection summary statistics",
    )(get)
