from flask_smorest import Blueprint
from flask.views import MethodView
from marshmallow import Schema, fields
from datetime import datetime, timezone, timedelta
import sys
import os

# Import database connection utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
try:
    from yolo_pipeline.database import db_connection
    from yolo_pipeline.config import get_db_env
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("[WARN] Database connection not available in bears endpoint", file=sys.stderr)

blp = Blueprint(
    "Bears",
    "bears",
    url_prefix="/api",
    description="Endpoints for bear data."
)


class BearSchema(Schema):
    """Schema representing a Bear data record returned by the API."""
    bearId = fields.String(required=True, description="Unique ID of the bear (detection ID)")
    pose = fields.String(required=True, description="Pose of the bear (e.g., Sitting, Standing, Walking)")
    timestamp = fields.DateTime(
        required=True,
        format="iso",
        description="ISO 8601 timestamp (UTC) when the pose was recorded"
    )
    in_video_time = fields.Float(
        allow_none=True,
        description="Time in seconds within the video timeline (computed from frame_index/fps)"
    )
    ocr_date_text = fields.String(
        allow_none=True,
        description="OCR-extracted date text from the frame"
    )


@blp.route("/bears")
class BearList(MethodView):
    """Provide read-only access to Bear detection data from animal_detections_v3 table."""

    # PUBLIC_INTERFACE
    def get(self):
        """
        Returns a list of Bear detection records from the database. Each record contains:
        - bearId: String ID of the bear (detection record ID)
        - pose: String describing the bear's pose status
        - timestamp: ISO 8601 UTC timestamp when the detection was recorded
        - in_video_time: Time in seconds within the video (computed from frame_index/fps)
        - ocr_date_text: OCR-extracted date text from the video frame

        Returns:
            list[dict]: A list of bear detection records suitable for JSON serialization.
            
        Note:
            Falls back to mock data if database is not available.
        """
        # Try to fetch from database
        if DB_AVAILABLE:
            try:
                db_config = get_db_env()
                db_name = db_config.get('db_name')
                
                if db_name:
                    with db_connection(db_name) as conn:
                        cursor = conn.cursor(dictionary=True)
                        
                        # Query animal_detections_v3 table for bear detections
                        query = """
                            SELECT 
                                id,
                                detection_label,
                                pose_status,
                                created_at,
                                in_video_time,
                                ocr_date_text,
                                frame_time_seconds
                            FROM animal_detections_v3
                            WHERE LOWER(detection_label) LIKE '%bear%'
                            ORDER BY created_at DESC
                            LIMIT 100
                        """
                        
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        cursor.close()
                        
                        if rows:
                            data = []
                            for row in rows:
                                data.append({
                                    "bearId": f"B{row['id']:04d}",
                                    "pose": row['pose_status'] or "Unknown",
                                    "timestamp": row['created_at'],
                                    "in_video_time": float(row['in_video_time']) if row['in_video_time'] is not None else None,
                                    "ocr_date_text": row['ocr_date_text']
                                })
                            
                            print(f"[INFO] Returning {len(data)} bear records from database", file=sys.stderr)
                            return data
                        else:
                            print("[INFO] No bear detections found in database, returning mock data", file=sys.stderr)
                            
            except Exception as e:
                print(f"[ERROR] Failed to fetch bear data from database: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
        
        # Fallback to mock data
        now = datetime.now(timezone.utc)
        data = [
            {
                "bearId": "B001", 
                "pose": "Sitting", 
                "timestamp": (now - timedelta(seconds=5)),
                "in_video_time": 10.5,
                "ocr_date_text": "2025-10-28 17:30:34"
            },
            {
                "bearId": "B002", 
                "pose": "Standing", 
                "timestamp": (now - timedelta(seconds=15)),
                "in_video_time": 20.0,
                "ocr_date_text": "2025-10-28 17:30:44"
            },
            {
                "bearId": "B003", 
                "pose": "Walking", 
                "timestamp": (now - timedelta(seconds=25)),
                "in_video_time": 30.5,
                "ocr_date_text": "2025-10-28 17:30:54"
            },
        ]
        return data

    # Document response schema via flask-smorest
    get = blp.response(
        200,
        BearSchema(many=True),
        description="List of Bear detection data with OCR date labels and video timing",
    )(get)


class OCRResultSchema(Schema):
    """Schema for OCR extraction result."""
    extracted_text = fields.String(allow_none=True, description="OCR-extracted date text")
    in_video_time = fields.Float(allow_none=True, description="In-video time if applicable")
    success = fields.Boolean(required=True, description="Whether OCR extraction was successful")
    message = fields.String(description="Status message")


@blp.route("/extract-date-label")
class ExtractDateLabel(MethodView):
    """Endpoint to extract date label from an uploaded image using OCR."""

    # PUBLIC_INTERFACE
    def post(self):
        """
        Extract date label text from an uploaded image file using OCR.
        
        Expects a multipart/form-data POST request with an 'image' file field.
        The image should contain a date label overlay in the DATE region as defined in config.py.
        
        Returns:
            dict: OCR extraction result containing:
                - extracted_text: The OCR-extracted date text (or None)
                - in_video_time: None (not applicable for single images)
                - success: Boolean indicating if extraction succeeded
                - message: Status message
                
        Example:
            curl -X POST -F "image=@frame.jpg" http://localhost:5000/api/extract-date-label
        """
        from flask import request
        import tempfile
        
        try:
            # Import OCR function
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
            from yolo_pipeline.ocr import extract_date_label_from_image_path
            
            # Check if image file is present
            if 'image' not in request.files:
                return {
                    "success": False,
                    "message": "No image file provided",
                    "extracted_text": None,
                    "in_video_time": None
                }, 400
            
            file = request.files['image']
            if file.filename == '':
                return {
                    "success": False,
                    "message": "Empty filename",
                    "extracted_text": None,
                    "in_video_time": None
                }, 400
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # Extract date label
                extracted_text, in_video_time = extract_date_label_from_image_path(tmp_path)
                
                if extracted_text:
                    return {
                        "success": True,
                        "message": "Date label extracted successfully",
                        "extracted_text": extracted_text,
                        "in_video_time": in_video_time
                    }, 200
                else:
                    return {
                        "success": False,
                        "message": "No text could be extracted from the DATE region",
                        "extracted_text": None,
                        "in_video_time": None
                    }, 200
                    
            finally:
                # Clean up temp file
                import os as os_module
                if os_module.path.exists(tmp_path):
                    os_module.unlink(tmp_path)
                    
        except ImportError as e:
            return {
                "success": False,
                "message": f"OCR functionality not available: {str(e)}. Install easyocr: pip install easyocr",
                "extracted_text": None,
                "in_video_time": None
            }, 503
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "message": f"Error processing image: {str(e)}",
                "extracted_text": None,
                "in_video_time": None
            }, 500

    # Document response schema
    post = blp.response(
        200,
        OCRResultSchema(),
        description="OCR extraction result"
    )(post)
