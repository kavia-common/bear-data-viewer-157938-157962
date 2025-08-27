from flask_smorest import Blueprint
from flask.views import MethodView
from marshmallow import Schema, fields
from bear_data_service import get_bear_data_for_flask

blp = Blueprint(
    "Bears",
    "bears",
    url_prefix="/api",
    description="Endpoints for bear data."
)


class BearSchema(Schema):
    """Schema representing a Bear data record returned by the API."""
    bearId = fields.String(required=True, description="Unique ID of the bear")
    pose = fields.String(required=True, description="Pose of the bear (e.g., Sitting, Standing, Walking)")
    timestamp = fields.DateTime(
        required=True,
        format="iso",
        description="ISO 8601 timestamp (UTC) when the pose was recorded"
    )


@blp.route("/bears")
class BearList(MethodView):
    """Provide read-only access to mock Bear data."""

    # PUBLIC_INTERFACE
    def get(self):
        """
        Returns a list of mock Bear records. Each record contains:
        - bearId: String ID of the bear
        - pose: String describing the bear's pose
        - timestamp: ISO 8601 UTC timestamp when the pose was recorded

        Returns:
            list[dict]: A list of bear records suitable for JSON serialization.
        """
        # Use shared service module for business logic
        return get_bear_data_for_flask()

    # Document response schema via flask-smorest
    get = blp.response(
        200,
        BearSchema(many=True),
        description="List of mock Bear data",
    )(get)
