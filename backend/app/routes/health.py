from flask_smorest import Blueprint
from flask.views import MethodView

# Use a clean, correctly spelled name and tag for API docs consistency.
blp = Blueprint("Health", "health", url_prefix="/", description="Health check route")


@blp.route("/")
class HealthCheck(MethodView):
    # PUBLIC_INTERFACE
    def get(self):
        """
        Health check endpoint.

        Returns:
            dict: A JSON object indicating service health.
        """
        return {"message": "Healthy"}
