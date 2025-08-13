from datetime import datetime, timedelta, timezone


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


def test_bears_response_is_json_array_with_expected_length_and_type(client):
    """
    Validate that the response is JSON, is a list, and contains expected number of items.
    """
    resp = client.get("/api/bears")
    assert resp.status_code == 200
    assert "application/json" in resp.content_type

    data = resp.get_json()
    assert isinstance(data, list), "Response JSON must be a list"
    # The current implementation returns exactly 3 mock records
    assert len(data) == 3, "Expected 3 mock bear records"


def test_bears_item_structure_and_types(client):
    """
    Validate each item has bearId (str), pose (str), timestamp (str) fields.
    """
    resp = client.get("/api/bears")
    data = resp.get_json()
    required_keys = {"bearId", "pose", "timestamp"}

    for i, item in enumerate(data):
        assert isinstance(item, dict), f"Item {i} must be a dict"
        assert required_keys.issubset(item.keys()), f"Item {i} must contain keys {required_keys}"
        assert isinstance(item["bearId"], str), f"Item {i} bearId must be a string"
        assert isinstance(item["pose"], str), f"Item {i} pose must be a string"
        assert isinstance(item["timestamp"], str), f"Item {i} timestamp must be a string"


def test_bears_timestamp_iso_utc_and_reasonable_recency(client):
    """
    Validate timestamps are ISO 8601, timezone-aware UTC, and reasonably recent (within 5 minutes).
    """
    resp = client.get("/api/bears")
    data = resp.get_json()

    now = datetime.now(timezone.utc)
    lower_bound = now - timedelta(minutes=5)

    for i, item in enumerate(data):
        ts_str = item["timestamp"]
        # Parse as ISO 8601 (supports +00:00)
        parsed = datetime.fromisoformat(ts_str)
        assert parsed.tzinfo is not None, f"Item {i} timestamp must be timezone-aware"
        assert parsed.utcoffset() == timedelta(0), f"Item {i} timestamp must be UTC"
        # The mock uses current time minus some seconds; allow generous bounds
        assert lower_bound <= parsed <= now, f"Item {i} timestamp must be within the last 5 minutes"


def test_bears_sorted_descending_by_timestamp(client):
    """
    Validate the items are returned sorted with most recent first (descending by timestamp).
    Based on the implementation: now-5s, now-15s, now-25s.
    """
    resp = client.get("/api/bears")
    data = resp.get_json()

    parsed_times = [datetime.fromisoformat(item["timestamp"]) for item in data]
    # Ensure non-increasing order: t[i] >= t[i+1]
    for i in range(len(parsed_times) - 1):
        assert parsed_times[i] >= parsed_times[i + 1], "Timestamps must be in descending order (most recent first)"


def test_bears_method_not_allowed_for_post(client):
    """
    Verify that POST is not allowed on /api/bears (read-only endpoint).
    """
    resp = client.post("/api/bears", json={})
    assert resp.status_code in (405, 404), "Expected 405 Method Not Allowed (or 404) for POST on /api/bears"
