import os
from flask import Flask
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
    # Send CORS headers on all responses for matched resources (including GET)
    # and appropriately handle preflight OPTIONS.
)

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
