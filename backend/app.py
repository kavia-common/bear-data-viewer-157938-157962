import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env if present (non-fatal if missing)
load_dotenv()

def create_app() -> Flask:
    """
    PUBLIC_INTERFACE
    create_app()

    This is the Flask application factory. It sets up the app instance,
    configures CORS for React running at http://localhost:3000, and
    registers API blueprints/routes.
    """
    app = Flask(__name__)

    # Configure logging
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    # Configure CORS for /api/* endpoints to allow frontend origin
    # Credentials are disabled as requested.
    CORS(
        app,
        resources={r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:3001"]}},
        supports_credentials=False,
    )

    # Register routes
    try:
        from app.routes.health import health_bp
        app.register_blueprint(health_bp)
    except Exception as e:
        app.logger.warning(f"Health blueprint registration failed: {e}")

    # Bears routes
    try:
        from app.routes.bears import bears_bp
        app.register_blueprint(bears_bp)
    except Exception as e:
        app.logger.warning(f"Bears blueprint registration failed: {e}")

    @app.get("/")
    def root():
        """Simple root to indicate service is running."""
        return {"status": "running"}

    return app


# Only run development server if executed directly.
# The preview system manages the port (typically 3001) externally, so avoid hardcoding or overriding it.
if __name__ == "__main__":
    app = create_app()
    # Respect existing environment; default to 3001 but allow the platform to control.
    port = int(os.getenv("PORT", "3001"))
    # host set to 0.0.0.0 for container environments
    app.run(host="0.0.0.0", port=port)
