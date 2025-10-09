from app import app

# PUBLIC_INTERFACE
def create_app():
    """
    Create and return the Flask application instance.

    Returns:
        Flask: The configured Flask app.
    """
    return app


if __name__ == "__main__":
    """
    Development entrypoint.

    Binds the Flask server to 0.0.0.0 on port 3001 so it is reachable
    from the container network and matches the expected exposed port.
    """
    # Bind to all interfaces and port 3001 to avoid default port 5000 conflicts.
    app.run(host="0.0.0.0", port=3001)
