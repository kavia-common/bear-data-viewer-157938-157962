def test_bears_returns_200(client):
    """
    Ensure the /api/bears endpoint returns HTTP 200 OK.
    """
    resp = client.get("/api/bears")
    assert resp.status_code == 200, "Expected 200 OK from /api/bears"


def test_bears_trailing_slash_returns_200(client):
    """
    Ensure the /api/bears/ (with trailing slash) also returns 200 due to strict_slashes=False.
    """
    resp = client.get("/api/bears/")
    assert resp.status_code == 200, "Expected 200 OK from /api/bears/"


def test_bears_response_shape(client):
    """
    Validate that the response is JSON object with detections array, count, and last_updated.
    """
    resp = client.get("/api/bears")
    assert resp.status_code == 200
    assert "application/json" in resp.content_type

    data = resp.get_json()
    assert isinstance(data, dict)
    assert "detections" in data and isinstance(data["detections"], list)
    assert "count" in data and isinstance(data["count"], int)
    assert "last_updated" in data and isinstance(data["last_updated"], str)

    # If sample data, detections should be bear-only by default
    for det in data["detections"]:
        assert det.get("label") == "bear"


def test_detections_endpoint_filters_and_metadata(client):
    """
    Validate /api/detections supports filters and returns metadata.
    """
    # Fetch all
    resp = client.get("/api/detections")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert isinstance(payload, dict)
    all_count = payload.get("count", -1)
    assert all_count >= 0

    # Min confidence filter (should be <= total)
    resp2 = client.get("/api/detections?min_confidence=0.9")
    assert resp2.status_code == 200
    payload2 = resp2.get_json()
    assert payload2["count"] <= all_count

    # Time range filter
    resp3 = client.get("/api/detections?start_time=0.0&end_time=3.0")
    assert resp3.status_code == 200
    payload3 = resp3.get_json()
    for det in payload3["detections"]:
        t = float(det["frame_time_seconds"])
        assert 0.0 <= t <= 3.0


def test_dataset_health(client):
    """
    Verify dataset health endpoint returns ok and count >= 0.
    """
    resp = client.get("/api/dataset/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "ok"
    assert isinstance(payload["count"], int)
