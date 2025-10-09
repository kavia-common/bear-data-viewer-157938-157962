from flask_smorest import Blueprint
from flask.views import MethodView
from marshmallow import Schema, fields
from flask import jsonify, make_response
from typing import Any, Dict, List
from app.data_loader import SEED_DETECTIONS

blp = Blueprint(
    "Results",
    "results",
    url_prefix="/api",
    description="Canonical results endpoint returning the embedded manual dataset."
)


class ResultSchema(Schema):
    """Schema for a single result record."""
    frame_time_seconds = fields.Float(required=True, description="Frame timestamp in seconds since start of video")
    label = fields.String(required=True, description="Detected label/class")
    x1 = fields.Float(required=True, description="Bounding box x1")
    y1 = fields.Float(required=True, description="Bounding box y1")
    x2 = fields.Float(required=True, description="Bounding box x2")
    y2 = fields.Float(required=True, description="Bounding box y2")
    confidence = fields.Float(required=True, description="Detection confidence [0,1]")


class ResultsResponseSchema(Schema):
    """Schema for results response."""
    results = fields.List(fields.Nested(ResultSchema), required=True, description="List of manual results")


def _build_response(payload: Dict[str, Any], status: int = 200):
    """Build JSON response while letting CORS middleware/app config add headers."""
    resp = make_response(jsonify(payload), status)
    return resp


@blp.route("/results", methods=["GET", "OPTIONS"])
class ResultsList(MethodView):
    """Return the canonical manual dataset under { "results": [...] }.

    This endpoint serves the embedded data and does not rely on CSV or legacy mocks.
    CORS is managed at the app level and by middleware for /api/* paths.
    """

    # PUBLIC_INTERFACE
    def get(self):
        """Get the in-memory manual results dataset. Always uses embedded data."""
        # Copy list to avoid external mutation
        rows: List[Dict[str, Any]] = list(SEED_DETECTIONS)
        # Optionally sort deterministically by time asc for consistent UI rendering
        rows.sort(key=lambda d: float(d.get("frame_time_seconds", 0.0)))
        payload = {"results": rows}
        return _build_response(payload, 200)

    # Document the response shape via flask-smorest
    get = blp.response(
        200,
        ResultsResponseSchema(),
        description="Canonical in-memory results dataset."
    )(get)
