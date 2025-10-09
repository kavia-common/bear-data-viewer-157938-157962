# Backend - Flask API

How to run locally:
- Install dependencies:
  pip install -r requirements.txt
- Start server on 0.0.0.0:3001:
  python run.py

Health check:
- GET / returns {"status":"ok"}

API routes:
- GET /api/bears returns mock bear data

OpenAPI docs:
- Visit /docs to load Swagger UI

CORS:
- Set CORS_ALLOWED_ORIGINS env var as a comma-separated list to allow specific front-end origins.
- Defaults to specific preview origins if not set.

Environment variables:
- CORS_ALLOWED_ORIGINS: Comma-separated allowed origins for CORS

Notes:
- This app binds to port 3001. Ensure this port is exposed by the container.
