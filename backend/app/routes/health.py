from flask_smorest import Blueprint
from flask.views import MethodView

# PUBLIC_INTERFACE
blp = Blueprint(
    "Health",
    "health",
    url_prefix="/",
    description="Health check route providing readiness/liveness status",
)


@blp.route("/")
class HealthCheck(MethodView):
    """
    Health check endpoint.

    Returns:
        JSON: {"status": "ok"} with HTTP 200 to indicate service is healthy.
    """
    # PUBLIC_INTERFACE
    def get(self):
        """
        Simple health check to verify the API is running.

        Returns:
            dict: {"status": "ok"}
        """
        return {"status": "ok"}
