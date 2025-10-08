from flask import Response

ORIGIN = "http://localhost:3000"


def test_cors_headers_present_on_get_bears(client):
    """
    Simulate a browser GET with Origin header to ensure CORS headers are returned.
    """
    resp: Response = client.get("/api/bears", headers={"Origin": ORIGIN})
    assert resp.status_code == 200
    # Access-Control-Allow-Origin should echo back the allowed origin
    assert resp.headers.get("Access-Control-Allow-Origin") == ORIGIN
    # Vary header should include Origin for proper caching behavior
    vary = resp.headers.get("Vary", "")
    assert "Origin" in vary


def test_preflight_options_returns_cors_headers(client):
    """
    Simulate a CORS preflight OPTIONS request with Access-Control-Request-Method and headers.
    """
    headers = {
        "Origin": ORIGIN,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Content-Type, Authorization",
    }
    resp: Response = client.options("/api/bears", headers=headers)
    # flask-cors typically returns 200 for preflight
    assert resp.status_code in (200, 204)
    assert resp.headers.get("Access-Control-Allow-Origin") == ORIGIN
    # Ensure requested headers are allowed
    allow_headers = resp.headers.get("Access-Control-Allow-Headers", "")
    assert "Content-Type" in allow_headers
    assert "Authorization" in allow_headers
    # Ensure method allowed
    allow_methods = resp.headers.get("Access-Control-Allow-Methods", "")
    assert "GET" in allow_methods
