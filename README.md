# bear-data-viewer-157938-157962

Backend API (restored to original behavior)

- Endpoint: GET /api/bears returns a mock list of bear detections under:
  { "bears": [ { frame_time_seconds, label, x1, y1, x2, y2, confidence }, ... ] }

- Health: GET / returns {"message": "Healthy"}

CORS
- Configured with flask-cors for /api/* (GET/OPTIONS).
- In this dev setup, all origins are allowed. In production, restrict to your frontend origin(s).

Ports and protocol
- Default dev: backend on http://localhost:3001 and frontend on http://localhost:3000.
- If your frontend runs over HTTPS, ensure the backend is also reachable via HTTPS to avoid mixed-content errors.

OpenAPI
- Visit /docs for Swagger UI.
