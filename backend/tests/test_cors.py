from flask import Response

ORIGIN_LOCAL = "http://localhost:3000"
ORIGIN_CLOUD = "https://vscode-internal-15672-beta.beta01.cloud.kavia.ai:4000"
ORIGIN_CURRENT_PREVIEW_3000 = "https://vscode-internal-34388-beta.beta01.cloud.kavia.ai:3000"


def test_cors_headers_present_on_get_bears_localhost(client):
    """
    Simulate a browser GET with Origin header for localhost to ensure CORS headers are returned.
    """
    resp: Response = client.get("/api/bears", headers={"Origin": ORIGIN_LOCAL})
    assert resp.status_code == 200
    # Access-Control-Allow-Origin should echo back the allowed origin
    assert resp.headers.get("Access-Control-Allow-Origin") == ORIGIN_LOCAL
    # Vary header should include Origin for proper caching behavior
    vary = resp.headers.get("Vary", "")
    assert "Origin" in vary


def test_preflight_options_returns_cors_headers_localhost(client):
    """
    Simulate a CORS preflight OPTIONS request with Access-Control-Request-Method and headers for localhost.
    """
    headers = {
        "Origin": ORIGIN_LOCAL,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Content-Type, Authorization",
    }
    resp: Response = client.options("/api/bears", headers=headers)
    # flask-cors typically returns 200 for preflight
    assert resp.status_code in (200, 204)
    assert resp.headers.get("Access-Control-Allow-Origin") == ORIGIN_LOCAL
    # Ensure requested headers are allowed
    allow_headers = resp.headers.get("Access-Control-Allow-Headers", "")
    assert "Content-Type" in allow_headers
    assert "Authorization" in allow_headers
    # Ensure method allowed
    allow_methods = resp.headers.get("Access-Control-Allow-Methods", "")
    assert "GET" in allow_methods


def test_cors_headers_present_on_get_bears_cloud_origin(client):
    """
    Simulate a browser GET with Origin header for the cloud preview origin to ensure CORS headers are returned.
    """
    resp: Response = client.get("/api/bears", headers={"Origin": ORIGIN_CLOUD})
    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == ORIGIN_CLOUD
    vary = resp.headers.get("Vary", "")
    assert "Origin" in vary


def test_preflight_options_returns_cors_headers_cloud_origin(client):
    """
    Simulate a CORS preflight OPTIONS request with Access-Control-Request-Method and headers for the cloud origin.
    """
    headers = {
        "Origin": ORIGIN_CLOUD,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Content-Type, Authorization",
    }
    resp: Response = client.options("/api/bears", headers=headers)
    assert resp.status_code in (200, 204)
    assert resp.headers.get("Access-Control-Allow-Origin") == ORIGIN_CLOUD
    allow_headers = resp.headers.get("Access-Control-Allow-Headers", "")
    assert "Content-Type" in allow_headers
    assert "Authorization" in allow_headers
    allow_methods = resp.headers.get("Access-Control-Allow-Methods", "")
    assert "GET" in allow_methods


def test_cors_headers_present_on_get_bears_current_preview(client):
    """
    Ensure Access-Control-Allow-Origin is returned for the currently running preview :3000 origin.
    """
    resp: Response = client.get("/api/bears", headers={"Origin": ORIGIN_CURRENT_PREVIEW_3000})
    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == ORIGIN_CURRENT_PREVIEW_3000
