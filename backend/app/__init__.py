import os
import logging
from flask import Flask, request, make_response
from flask_cors import CORS
from .routes.health import blp as health_blp
from .routes.bears import blp as bears_blp
from flask_smorest import Api

# Load .env if present (non-fatal if python-dotenv not installed or file missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # It's okay if python-dotenv isn't installed; environments like Docker/CI inject vars.
    pass

app = Flask(__name__)
app.url_map.strict_slashes = False

# Basic logging for startup diagnostics
logger = logging.getLogger("cors-config")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

# Configure CORS using environment variables with sensible defaults.
# Backwards compatibility: also support legacy CORS_ALLOWED_ORIGINS (note the 'ED').
# Precedence: CORS_ALLOW_ORIGINS takes priority, then CORS_ALLOWED_ORIGINS, else defaults.
allowed_origins_env = os.getenv("CORS_ALLOW_ORIGINS") or os.getenv("CORS_ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Defaults include localhost and example preview origins.
    # Include current running container preview domain ports 3000 and 4000 as requested.
    allowed_origins = [
        # Local development frontend
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Backend direct access during dev tools
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        # Cloud preview origins (retained for current environment)
        "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:3000",
        "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:4000",
        "https://vscode-internal-15672-beta.beta01.cloud.kavia.ai:4000",
        # Add current running workspace preview mentioned in the task
        "https://vscode-internal-34388-beta.beta01.cloud.kavia.ai:3000",
        "https://vscode-internal-34388-beta.beta01.cloud.kavia.ai:4000",
    ]

supports_credentials = (os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true")
allow_headers = [h.strip() for h in os.getenv("CORS_ALLOW_HEADERS", "Content-Type,Authorization").split(",") if h.strip()]
methods = [m.strip().upper() for m in os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,PATCH,DELETE,OPTIONS").split(",") if m.strip()]

# Log resolved CORS config at startup for debugging
logger.info("CORS resolved origins: %s", allowed_origins)
logger.info("CORS supports_credentials=%s", supports_credentials)
logger.info("CORS allow_headers=%s", allow_headers)
logger.info("CORS methods=%s", methods)

# Apply CORS only to API routes and explicitly allow headers/methods.
# flask-cors will automatically handle OPTIONS preflight responses.
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins,
            "methods": methods,
            "allow_headers": allow_headers,
            "expose_headers": ["Content-Type"],
            "supports_credentials": supports_credentials,
            "max_age": 600,
        }
    },
    vary_header=True,  # ensure Vary: Origin behavior
    automatic_options=True,  # ensure preflight handling
)

# Fallback after_request to ensure CORS headers are present for API routes
# This addresses cases where middleware is bypassed or headers are stripped by a proxy.
@app.after_request
def ensure_cors_headers(resp):
    """
    Ensure CORS headers are present on /api/* responses for allowed origins.
    This is a safe fallback in case a proxy strips headers or an early return bypasses CORS.
    """
    try:
        origin = request.headers.get("Origin")
        path = request.path or ""
        # Only apply to API routes and when an Origin header is sent.
        if origin and path.startswith("/api"):
            if origin in allowed_origins:
                # If flask-cors already set the header, leave it.
                if not resp.headers.get("Access-Control-Allow-Origin"):
                    resp.headers["Access-Control-Allow-Origin"] = origin
                    resp.headers.add("Vary", "Origin")
                    # Mirror core settings used above
                    resp.headers.setdefault("Access-Control-Allow-Methods", ", ".join(methods))
                    resp.headers.setdefault("Access-Control-Allow-Headers", ", ".join(allow_headers))
                    resp.headers.setdefault("Access-Control-Expose-Headers", "Content-Type")
                    resp.headers.setdefault("Access-Control-Max-Age", "600")
                    if supports_credentials:
                        resp.headers.setdefault("Access-Control-Allow-Credentials", "true")
            # If origin not allowed, do not add CORS headers.
        return resp
    except Exception:
        # In case of unexpected errors in the hook, return the response unmodified.
        return resp

# OpenAPI/Swagger configuration
app.config["API_TITLE"] = "My Flask API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config['OPENAPI_URL_PREFIX'] = '/docs'
app.config["OPENAPI_SWAGGER_UI_PATH"] = ""
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

api = Api(app)
api.register_blueprint(health_blp)
api.register_blueprint(bears_blp)
