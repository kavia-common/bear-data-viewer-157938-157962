# bear-data-viewer-157938-157962

Backend CORS configuration:
- The Flask backend reads CORS settings from environment variables.
- See backend/.env.example for:
  - CORS_ALLOW_ORIGINS (e.g., http://localhost:3000)
  - CORS_ALLOW_CREDENTIALS (true/false)
  - CORS_ALLOW_HEADERS (comma-separated)
  - CORS_ALLOW_METHODS (comma-separated)
- Copy the example to .env and adjust as needed:
  - cd backend && cp .env.example .env
- Defaults already allow http://localhost:3000 and scope CORS to /api/*.