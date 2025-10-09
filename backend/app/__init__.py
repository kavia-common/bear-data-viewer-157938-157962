import os
from flask import Flask
from flask_cors import CORS
from .routes.health import blp as health_blp
from .routes.bears import blp as bears_blp
from .routes.health_dataset import blp as dataset_health_blp
from flask_smorest import Api
from .middleware import apply_api_cors_headers  # optional safety middleware


app = Flask(__name__)
app.url_map.strict_slashes = False

# Configure CORS using BACKEND_CORS_ORIGINS (comma-separated). If unset, default to the frontend origin.
# Note: Do not hardcode secrets. Origins are not secret.
origins_env = os.getenv("BACKEND_CORS_ORIGINS")
if origins_env:
    allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "https://vscode-internal-20401-qa.qa01.cloud.kavia.ai:3000",
    ]

# Apply CORS only to /api/* and include standard methods/headers to satisfy preflight.
# Expose Content-Type for clients that need to read it.
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins,
            "methods": ["GET", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type"],
            "supports_credentials": False,  # enable True only if cookies/credentials are needed
            "max_age": 600,
        }
    },
)

# Optional middleware to ensure CORS headers are present for /api/* responses even if a custom response path is used.
apply_api_cors_headers(app, allowed_origins, supports_credentials=False)

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
api.register_blueprint(dataset_health_blp)
