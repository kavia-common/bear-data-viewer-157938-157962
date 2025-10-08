import os
from flask import Flask, request, make_response
from flask_cors import CORS
from .routes.health import blp as health_blp
from .routes.bears import blp as bears_blp
from flask_smorest import Api


app = Flask(__name__)
app.url_map.strict_slashes = False

# Configure CORS to allow specified frontend origins.
# CORS_ALLOWED_ORIGINS (comma-separated) can override defaults.
allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Defaults include localhost and the requested cloud preview origin.
    allowed_origins = [
        # Local development
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Backend direct access during dev tools
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        # Existing cloud preview origins (retain)
        "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:3000",
        "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:4000",
        # Newly required frontend preview origin
        "https://vscode-internal-15672-beta.beta01.cloud.kavia.ai:4000",
    ]

# Apply CORS only to API routes and explicitly allow common headers/methods.
# flask-cors will automatically handle OPTIONS preflight responses.
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type"],
            "supports_credentials": False,
            "max_age": 600,
        }
    },
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
                    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
                    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
                    resp.headers.setdefault("Access-Control-Expose-Headers", "Content-Type")
                    resp.headers.setdefault("Access-Control-Max-Age", "600")
                    # We don't enable credentials unless required; keep consistent
                    # If you need credentials, set supports_credentials True above and here:
                    # resp.headers["Access-Control-Allow-Credentials"] = "true"
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
