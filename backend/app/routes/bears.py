from flask_smorest import Blueprint
from flask.views import MethodView
from marshmallow import Schema, fields
from flask import jsonify

blp = Blueprint(
    "Bears",
    "bears",
    url_prefix="/api",
    description="Endpoints for bear data."
)

class BearSchema(Schema):
    frame_time_seconds = fields.Float(required=True)
    label = fields.String(required=True)
    x1 = fields.Float(required=True)
    y1 = fields.Float(required=True)
    x2 = fields.Float(required=True)
    y2 = fields.Float(required=True)
    confidence = fields.Float(required=True)

class BearsResponseSchema(Schema):
    bears = fields.List(fields.Nested(BearSchema), required=True)

# PUBLIC_INTERFACE
@blp.route("/bears", methods=["GET"])
class Bears(MethodView):
    """Return original mock bear dataset with the old response shape.

    Response:
    {
      "bears": [
        { "frame_time_seconds": 10.0, "label": "bear", "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "confidence": 0.9 }
      ]
    }
    """
    def get(self):
        # Old mock data shape
        data = {
            "bears": [
                {"frame_time_seconds": 10.0, "label": "bear", "x1": 680.0, "y1": 143.0, "x2": 1212.0, "y2": 1073.0, "confidence": 0.9395},
                {"frame_time_seconds": 30.0, "label": "bear", "x1": 1005.0, "y1": 278.0, "x2": 1317.0, "y2": 522.0, "confidence": 0.9163},
                {"frame_time_seconds": 40.0, "label": "bear", "x1": 1154.0, "y1": 248.0, "x2": 1349.0, "y2": 573.0, "confidence": 0.9312},
                {"frame_time_seconds": 60.0, "label": "bear", "x1": 897.0, "y1": 248.0, "x2": 1146.0, "y2": 534.0, "confidence": 0.8727},
                {"frame_time_seconds": 70.0, "label": "bear", "x1": 852.0, "y1": 170.0, "x2": 1562.0, "y2": 833.0, "confidence": 0.9376},
                {"frame_time_seconds": 80.0, "label": "bear", "x1": 1181.0, "y1": 212.0, "x2": 1310.0, "y2": 461.0, "confidence": 0.4593},
                {"frame_time_seconds": 110.0, "label": "bear", "x1": 1082.0, "y1": 298.0, "x2": 1294.0, "y2": 704.0, "confidence": 0.9303},
            ]
        }
        return jsonify(data)

    get = blp.response(
        200,
        BearsResponseSchema(),
        description="Original bears dataset"
    )(get)
