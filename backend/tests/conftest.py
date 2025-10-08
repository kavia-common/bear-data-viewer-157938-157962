import os
import sys
import pytest

# Ensure the backend root is on sys.path so `from app import app` works when running pytest from this folder
BACKEND_ROOT = os.path.dirname(os.path.dirname(__file__))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app import app as flask_app


@pytest.fixture(scope="session")
def app():
    """
    Provides the Flask application instance for tests.
    """
    return flask_app


@pytest.fixture()
def client(app):
    """
    Provides a Flask test client with an application context.
    """
    app.testing = True
    with app.app_context():
        with app.test_client() as client:
            yield client
