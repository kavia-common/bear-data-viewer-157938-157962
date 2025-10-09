# bear-data-viewer-157938-157962

Backend API (restored to original behavior)

- Endpoint: GET /api/bears returns a mock list of bear detections under:
  { "bears": [ { frame_time_seconds, label, x1, y1, x2, y2, confidence }, ... ] }

- Health: GET / returns {"message": "Healthy"}

CORS
- Configured with flask-cors for /api/* (GET/OPTIONS).

OpenAPI
- Visit /docs for Swagger UI.
