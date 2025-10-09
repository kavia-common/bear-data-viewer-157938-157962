from flask_smorest import Blueprint
from flask.views import MethodView
from app.data_loader import load_detections

blp = Blueprint("Dataset Health", "dataset_health", url_prefix="/api", description="Dataset health check")

@blp.route("/dataset/health", methods=["GET"])
class DatasetHealth(MethodView):
    # PUBLIC_INTERFACE
    def get(self):
        """Health check for the in-memory detections dataset.

        Returns:
            {"status": "ok", "count": <int>}
        """
        detections, _ = load_detections()
        return {"status": "ok", "count": len(detections)}
