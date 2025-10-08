import os
from flask import Flask
from flask_cors import CORS
from .routes.health import blp as health_blp
from .routes.bears import blp as bears_blp
from flask_smorest import Api


app = Flask(__name__)
app.url_map.strict_slashes = False

# Configure CORS to allow only the specified frontend origin by default.
# You can set CORS_ALLOWED_ORIGINS in the environment as a comma-separated list of origins to override.
allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Include local development origins and deployed preview origins.
    allowed_origins = [
        # Local development
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # In some dev setups, backend may be hit directly from a tool on same host/port
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        # Cloud preview URLs (retain existing)
        "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:3000",
        "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:4000",
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
