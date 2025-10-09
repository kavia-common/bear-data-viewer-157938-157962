from typing import List
from flask import Flask, request

# PUBLIC_INTERFACE
def apply_api_cors_headers(app: Flask, allowed_origins: List[str], supports_credentials: bool = False) -> None:
    """Attach an after_request hook to ensure CORS headers exist for /api/* responses.

    This is a safety net for custom response paths. flask-cors should normally handle CORS,
    but this guarantees presence of key headers for verification.

    Args:
        app: Flask application instance.
        allowed_origins: List of allowed origins configured for /api/*.
        supports_credentials: Whether to include Access-Control-Allow-Credentials.
    """
    @app.after_request
    def _ensure_cors_headers(resp):
        try:
            path = request.path or ""
            if not path.startswith("/api/"):
                return resp

            # If flask-cors has already added headers, keep them; otherwise add minimal ones.
            if "Access-Control-Allow-Origin" not in resp.headers:
                origin = request.headers.get("Origin")
                # Mirror back the origin only if in the allowed list; otherwise do not add.
                if origin and origin in allowed_origins:
                    resp.headers["Access-Control-Allow-Origin"] = origin
                # If no origin or not allowed, do not inject a permissive header.

            if "Access-Control-Allow-Methods" not in resp.headers:
                resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"

            if "Access-Control-Allow-Headers" not in resp.headers:
                resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

            if "Access-Control-Expose-Headers" not in resp.headers:
                resp.headers["Access-Control-Expose-Headers"] = "Content-Type"

            if supports_credentials and "Access-Control-Allow-Credentials" not in resp.headers:
                resp.headers["Access-Control-Allow-Credentials"] = "true"
        except Exception:
            # Do not let header logic break the response.
            return resp
        return resp
