# bear-data-viewer-157938-157962

Backend CORS configuration:
- The Flask backend reads CORS settings from environment variables.
- See backend/.env.example for:
  - CORS_ALLOW_ORIGINS (e.g., http://localhost:3000, https://vscode-internal-34388-beta.beta01.cloud.kavia.ai:3000, https://vscode-internal-34388-beta.beta01.cloud.kavia.ai:4000)
  - CORS_ALLOWED_ORIGINS (legacy; supported for backward compatibility if CORS_ALLOW_ORIGINS is not set)
  - CORS_ALLOW_CREDENTIALS (true/false)
  - CORS_ALLOW_HEADERS (comma-separated)
  - CORS_ALLOW_METHODS (comma-separated)
- Copy the example to .env and adjust as needed:
  - cd backend && cp .env.example .env
- Defaults already allow http://localhost:3000 and common preview origins, and scope CORS to /api/*.
- The application logs the resolved CORS configuration at startup for easier diagnostics.