from flask_smorest import Blueprint
from flask.views import MethodView
from marshmallow import Schema, fields, ValidationError
from flask import request, make_response, jsonify
import logging
from typing import Any, Dict, Optional

from app.data_loader import load_detections, query_detections

logger = logging.getLogger("bears")

blp = Blueprint(
    "Bears",
    "bears",
    url_prefix="/api",
    description="Endpoints for bear and detection data."
)


class DetectionSchema(Schema):
    """Schema for a single detection record."""
    frame_time_seconds = fields.Float(required=True, description="Frame timestamp in seconds since start of video")
    label = fields.String(required=True, description="Detected label/class")
    x1 = fields.Float(required=True, description="Bounding box x1")
    y1 = fields.Float(required=True, description="Bounding box y1")
    x2 = fields.Float(required=True, description="Bounding box x2")
    y2 = fields.Float(required=True, description="Bounding box y2")
    confidence = fields.Float(required=True, description="Detection confidence [0,1]")


class DetectionsResponseSchema(Schema):
    """Schema for detections response with metadata."""
    detections = fields.List(fields.Nested(DetectionSchema), required=True, description="Filtered detections")
    count = fields.Integer(required=True, description="Number of detections returned")
    last_updated = fields.DateTime(required=True, format="iso", description="Server-side last update timestamp (UTC)")


def _parse_float(name: str, value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        raise ValidationError(f"Invalid value for {name}. Must be a number.")


def _build_response(payload: Dict[str, Any], status: int = 200):
    resp = make_response(jsonify(payload), status)
    try:
        logger.info(
            "RESP %s | Status=%s, CORS={Origin:%s, Methods:%s, Headers:%s, Expose:%s, Credentials:%s}",
            request.path,
            status,
            resp.headers.get("Access-Control-Allow-Origin"),
            resp.headers.get("Access-Control-Allow-Methods"),
            resp.headers.get("Access-Control-Allow-Headers"),
            resp.headers.get("Access-Control-Expose-Headers"),
            resp.headers.get("Access-Control-Allow-Credentials"),
        )
    except Exception:
        pass
    return resp


# Load dataset once at import time (in-memory store)
_ALL_DETECTIONS, _LAST_UPDATED = load_detections()


@blp.route("/detections", methods=["GET", "OPTIONS"])
class DetectionsList(MethodView):
    """Return detection records with optional filters.

    Query parameters:
      - label: string, filter by label (exact match)
      - min_confidence: float, filter where confidence >= value
      - start_time: float, include detections with frame_time_seconds >= start_time
      - end_time: float, include detections with frame_time_seconds <= end_time

    Returns JSON:
      {
        "detections": [ ... ],
        "count": <int>,
        "last_updated": "<ISO-8601 UTC>"
      }
    """

    # PUBLIC_INTERFACE
    def get(self):
        """Get all detections, optionally filtered by label, min_confidence, and time range."""
        try:
            label = request.args.get("label")
            min_confidence = _parse_float("min_confidence", request.args.get("min_confidence"))
            start_time = _parse_float("start_time", request.args.get("start_time"))
            end_time = _parse_float("end_time", request.args.get("end_time"))
        except ValidationError as ve:
            return _build_response({"error": str(ve)}, 400)

        filtered = query_detections(
            _ALL_DETECTIONS,
            label=label,
            min_confidence=min_confidence,
            start_time=start_time,
            end_time=end_time,
        )
        # default to descending by frame_time_seconds for convenience
        filtered.sort(key=lambda d: float(d.get("frame_time_seconds", 0.0)), reverse=True)

        payload = {
            "detections": filtered,
            "count": len(filtered),
            "last_updated": _LAST_UPDATED,
        }
        return _build_response(payload, 200)

    get = blp.response(
        200,
        DetectionsResponseSchema(),
        description="List of detections with optional filters and metadata",
    )(get)


@blp.route("/bears", methods=["GET", "OPTIONS"])
class BearList(MethodView):
    """Backward-compatible alias that returns only detections where label == 'bear'.

    CORS: Preflight (OPTIONS) is handled by flask-cors configuration at the app level.
    """

    # PUBLIC_INTERFACE
    def get(self):
        """Return bear-only detections, preserving legacy /api/bears route."""
        # Parse optional filters but force label='bear' if not provided to maintain legacy behavior.
        try:
            label = request.args.get("label") or "bear"
            min_confidence = _parse_float("min_confidence", request.args.get("min_confidence"))
            start_time = _parse_float("start_time", request.args.get("start_time"))
            end_time = _parse_float("end_time", request.args.get("end_time"))
        except ValidationError as ve:
            return _build_response({"error": str(ve)}, 400)

        filtered = query_detections(
            _ALL_DETECTIONS,
            label=label,
            min_confidence=min_confidence,
            start_time=start_time,
            end_time=end_time,
        )
        filtered.sort(key=lambda d: float(d.get("frame_time_seconds", 0.0)), reverse=True)

        payload = {
            "detections": filtered,
            "count": len(filtered),
            "last_updated": _LAST_UPDATED,
        }
        # Using same response shape as /api/detections to unify contract moving forward.
        return _build_response(payload, 200)

    get = blp.response(
        200,
        DetectionsResponseSchema(),
        description="Bear-only detections (legacy alias) with optional filters and metadata",
    )(get)
