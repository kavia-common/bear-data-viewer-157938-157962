import pytest
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
