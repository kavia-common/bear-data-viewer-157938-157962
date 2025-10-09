# bear-data-viewer-157938-157962

Backend API updates

- New in-memory detections dataset served at /api/detections with optional CSV loader.
- Legacy /api/bears is preserved and returns detections where label == "bear" by default to avoid breaking the frontend.
- CORS remains configured via BACKEND_CORS_ORIGINS.

Environment variables

- DETECTIONS_CSV: Optional absolute/relative path to a CSV file with headers:
  frame_time_seconds,label,x1,y1,x2,y2,confidence
  If provided, the backend loads detections from the CSV at startup; otherwise it uses an embedded sample.

Endpoints

- GET /api/detections
  Query params:
    - label: string (exact match)
    - min_confidence: float
    - start_time: float (frame_time_seconds >= start_time)
    - end_time: float (frame_time_seconds <= end_time)
  Response:
  {
    "detections": [
      {
        "frame_time_seconds": 1.0,
        "label": "bear",
        "x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 220.0,
        "confidence": 0.92
      }
    ],
    "count": 1,
    "last_updated": "2025-01-01T00:00:00+00:00"
  }

- GET /api/bears
  Same schema as /api/detections but defaults to label="bear" if not specified.
  Optional query params (label, min_confidence, start_time, end_time) are accepted.

- GET /api/dataset/health
  Returns {"status":"ok","count": <int>} to verify dataset availability.

OpenAPI

- The project uses flask-smorest; visit /docs for Swagger UI.