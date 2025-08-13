import os
from flask import Flask
from flask_cors import CORS
from .routes.health import blp as health_blp
from .routes.bears import blp as bears_blp
from flask_smorest import Api


app = Flask(__name__)
app.url_map.strict_slashes = False

# Configure CORS to allow frontend origins from env or sensible defaults.
# You can set CORS_ALLOWED_ORIGINS in the environment as a comma-separated list of origins.
allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Defaults include common local dev ports and the platform preview URL.
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://vscode-internal-16867-beta.beta01.cloud.kavia.ai:4000",
    ]

# Apply CORS to the whole app.
# supports_credentials is False by default; adjust if you later need cookies/credentials across origins.
CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
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
