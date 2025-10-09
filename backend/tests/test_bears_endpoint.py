def test_bears_returns_200(client):
    """Ensure the /api/bears endpoint returns HTTP 200 OK."""
    resp = client.get("/api/bears")
    assert resp.status_code == 200, "Expected 200 OK from /api/bears"

def test_bears_trailing_slash_returns_200(client):
    """Ensure the /api/bears/ (with trailing slash) also returns 200 due to strict_slashes=False."""
    resp = client.get("/api/bears/")
    assert resp.status_code == 200, "Expected 200 OK from /api/bears/"

def test_bears_response_shape(client):
    """Validate that the response is JSON object with bears array."""
    resp = client.get("/api/bears")
    assert resp.status_code == 200
    assert "application/json" in resp.content_type

    data = resp.get_json()
    assert isinstance(data, dict)
    assert "bears" in data and isinstance(data["bears"], list)

    # All entries should be labeled 'bear'
    for det in data["bears"]:
        assert det.get("label") == "bear"
