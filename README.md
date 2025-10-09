# bear-data-viewer-157938-157962

Backend API updates

- NEW canonical endpoint: /api/results serving embedded manual dataset as { "results": [ ... ] }. No CSV required.
- /api/detections remains available (optionally supports CSV via DETECTIONS_CSV) for backward compatibility.
- Legacy /api/bears is preserved and returns detections where label == "bear".
- CORS remains configured via BACKEND_CORS_ORIGINS.

Environment variables

- DETECTIONS_CSV: Optional absolute/relative path to a CSV file with headers:
  frame_time_seconds,label,x1,y1,x2,y2,confidence
  If provided, /api/detections and /api/bears load from CSV at startup; otherwise they use the embedded sample.
  Note: /api/results always returns the embedded dataset and does NOT depend on CSV.

Endpoints

- GET /api/results
  Response:
  {
    "results": [
      {
        "frame_time_seconds": 1.0,
        "label": "bear",
        "x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 220.0,
        "confidence": 0.92
      }
    ]
  }

- GET /api/detections
  Query params:
    - label: string (exact match)
    - min_confidence: float
    - start_time: float (frame_time_seconds >= start_time)
    - end_time: float (frame_time_seconds <= end_time)
  Response:
  {
    "detections": [ ... ],
    "count": <int>,
    "last_updated": "<ISO-8601 UTC>"
  }

- GET /api/bears
  Same schema as /api/detections but defaults to label="bear" if not specified.

- GET /api/dataset/health
  Returns {"status":"ok","count": <int>} to verify dataset availability.

OpenAPI

- The project uses flask-smorest; visit /docs for Swagger UI.