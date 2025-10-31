import pytest

# These tests assume that a module detect_anteater_pacing.py will be implemented
# with the following public interfaces:
# - compute_centroid(bbox) -> (cx, cy)
# - classify_zone(centroid, zones) -> zone_id (e.g., "A","B",...)
# - detect_pacing(events, zones, threshold_ms=5000) -> list of state changes dicts
#
# State change dicts are expected to follow:
#   {'type': 'START'|'CONTINUE'|'STOP', 'anchor': 'A'..'F', 't': <ms timestamp>}
#
# Zones are polygons defined as lists of (x, y) tuples with keys 'id' and 'polygon'.


# PUBLIC_INTERFACE
def point_in_polygon(point, polygon):
    """Simple ray casting algorithm to determine if a point is inside a polygon."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Check if the edge crosses the horizontal line at y
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


@pytest.fixture
def zones_rect_grid():
    """
    Define six rectangular zones A..F in a simple grid for deterministic classification.
    Coordinate system: x to the right, y downwards (typical image space).
    Each zone is a rectangle polygon: (x1,y1) -> (x2,y1) -> (x2,y2) -> (x1,y2).
    """
    # Create 3x2 grid zones: A,B,C top row; D,E,F bottom row
    # Each cell 100x100
    def rect(x, y, w, h):
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    return [
        {"id": "A", "polygon": rect(0, 0, 100, 100)},
        {"id": "B", "polygon": rect(100, 0, 100, 100)},
        {"id": "C", "polygon": rect(200, 0, 100, 100)},
        {"id": "D", "polygon": rect(0, 100, 100, 100)},
        {"id": "E", "polygon": rect(100, 100, 100, 100)},
        {"id": "F", "polygon": rect(200, 100, 100, 100)},
    ]


@pytest.fixture
def classify_zone_impl(zones_rect_grid):
    """
    A helper function that mimics classify_zone behavior based on centroid inclusion.
    This will be used to validate that our tests construct events in the intended zones.
    """
    def _classify(centroid):
        for z in zones_rect_grid:
            if point_in_polygon(centroid, z["polygon"]):
                return z["id"]
        return None
    return _classify


@pytest.fixture
def centroid():
    # PUBLIC_INTERFACE
    def _compute_centroid(bbox):
        """
        Compute the centroid of a bbox {x1,x2,y1,y2}.
        """
        x1 = bbox["x1"]
        x2 = bbox["x2"]
        y1 = bbox["y1"]
        y2 = bbox["y2"]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return _compute_centroid


def bbox_from_point(x, y):
    # Create a tiny bbox around a point so centroid equals approximately the point
    return {"x1": x - 1, "x2": x + 1, "y1": y - 1, "y2": y + 1}


def make_event(t_ms, x, y):
    return {"timestamp_ms": t_ms, "bbox": bbox_from_point(x, y)}


@pytest.fixture
def A_point():
    # Inside A: e.g., center of A is (50,50)
    return (50, 50)


@pytest.fixture
def B_point():
    return (150, 50)


@pytest.fixture
def C_point():
    return (250, 50)


@pytest.fixture
def D_point():
    return (50, 150)


@pytest.fixture
def E_point():
    return (150, 150)


@pytest.fixture
def F_point():
    return (250, 150)


def expect_state(t, type_, anchor):
    return {"type": type_, "anchor": anchor, "t": t}


def require_module():
    """
    Attempt to import the target module and required functions.
    The module may not yet exist; tests should still compile and mark missing implementation.
    """
    import importlib
    try:
        mod = importlib.import_module("backend.detect_anteater_pacing")
    except Exception as e:
        pytest.skip(f"detect_anteater_pacing module not implemented yet: {e}")
        return None

    required = ["detect_pacing", "classify_zone", "compute_centroid"]
    for name in required:
        if not hasattr(mod, name):
            pytest.skip(f"Missing required function: {name} in detect_anteater_pacing")
    return mod


def test_no_pacing_detected(zones_rect_grid, A_point, B_point):
    mod = require_module()
    if mod is None:
        return

    # Short path A->B without returning to A within threshold
    events = [
        make_event(0, *A_point),
        make_event(1000, *B_point),
        make_event(12000, *B_point),  # Long after, still no A return
    ]
    results = mod.detect_pacing(events, zones_rect_grid, threshold_ms=5000)
    assert isinstance(results, list)
    assert results == []


def test_single_pacing_start_A_B_A_within_5s(zones_rect_grid, A_point, B_point):
    mod = require_module()
    if mod is None:
        return

    # A at t0, B at t1, A at t2 within 5s -> START at t2 with anchor A
    t0, t1, t2 = 0, 2000, 4500
    events = [
        make_event(t0, *A_point),
        make_event(t1, *B_point),
        make_event(t2, *A_point),
    ]
    results = mod.detect_pacing(events, zones_rect_grid, threshold_ms=5000)
    assert results == [expect_state(t2, "START", "A")]


def test_continue_with_repeated_A_star_A_within_5s(zones_rect_grid, A_point, B_point, C_point):
    mod = require_module()
    if mod is None:
        return

    # Repeated cycles: A->B->A (start), then A->C->A (continue), both within 5s
    t0, t1, t2 = 0, 1000, 3000          # First cycle returns at t2 -> START
    t3, t4, t5 = 3100, 3300, 6000       # Second cycle returns at t5 (3s from t3) -> CONTINUE

    events = [
        make_event(t0, *A_point),
        make_event(t1, *B_point),
        make_event(t2, *A_point),
        make_event(t3, *A_point),
        make_event(t4, *C_point),
        make_event(t5, *A_point),
    ]
    results = mod.detect_pacing(events, zones_rect_grid, threshold_ms=5000)
    assert results == [
        expect_state(t2, "START", "A"),
        expect_state(t5, "CONTINUE", "A"),
    ]


def test_stop_when_anchor_return_exceeds_5s(zones_rect_grid, A_point, B_point, C_point):
    mod = require_module()
    if mod is None:
        return

    # Start: A->B->A within 5s at t2
    # Then slow return: A->C->A taking >5s should STOP at the late A return time
    t0, t1, t2 = 0, 1000, 4000            # return in 4s -> START
    t3, t4, t5 = 8000, 9000, 14001        # 6.001s after t3 -> STOP

    events = [
        make_event(t0, *A_point),
        make_event(t1, *B_point),
        make_event(t2, *A_point),
        make_event(t3, *A_point),
        make_event(t4, *C_point),
        make_event(t5, *A_point),  # exceeds 5s from t3
    ]
    results = mod.detect_pacing(events, zones_rect_grid, threshold_ms=5000)
    assert results == [
        expect_state(t2, "START", "A"),
        expect_state(t5, "STOP", "A"),
    ]


def test_exactly_5s_boundary_is_inclusive(zones_rect_grid, A_point, B_point):
    mod = require_module()
    if mod is None:
        return

    # A->B->A where A return occurs exactly 5000ms after leaving A — should be included
    t0, t1, t2 = 0, 1000, 5000  # left A at t0, return at t2=5000 -> <= 5000 inclusive
    events = [
        make_event(t0, *A_point),
        make_event(t1, *B_point),
        make_event(t2, *A_point),
    ]
    results = mod.detect_pacing(events, zones_rect_grid, threshold_ms=5000)
    assert results == [expect_state(t2, "START", "A")]


def test_ignore_immediate_same_zone_repeats(zones_rect_grid, A_point, B_point):
    mod = require_module()
    if mod is None:
        return

    # Multiple A->A frames before moving out should not falsely trigger pacing;
    # pacing requires leaving anchor and returning within threshold.
    t0, t1, t2, t3, t4 = 0, 200, 400, 600, 1600
    events = [
        make_event(t0, *A_point),
        make_event(t1, *A_point),  # still A
        make_event(t2, *A_point),  # still A
        make_event(t3, *B_point),  # moved to B
        make_event(t4, *A_point),  # returned to A within 1s -> START here
    ]
    results = mod.detect_pacing(events, zones_rect_grid, threshold_ms=5000)
    assert results == [expect_state(t4, "START", "A")]


def test_multi_zone_between_anchors(zones_rect_grid, A_point, B_point, D_point, C_point):
    mod = require_module()
    if mod is None:
        return

    # Path A -> B -> D -> C -> A still counts as a return to the same anchor (A) within threshold
    t0, t1, t2, t3, t4 = 0, 1000, 1800, 2400, 4800  # total 4.8s from t0 to t4
    events = [
        make_event(t0, *A_point),
        make_event(t1, *B_point),
        make_event(t2, *D_point),
        make_event(t3, *C_point),
        make_event(t4, *A_point),
    ]
    results = mod.detect_pacing(events, zones_rect_grid, threshold_ms=5000)
    assert results == [expect_state(t4, "START", "A")]


def test_zone_mapping_helpers(zones_rect_grid, centroid, classify_zone_impl, A_point, B_point, C_point, D_point, E_point, F_point):
    mod = require_module()
    if mod is None:
        return

    # Verify centroid and classify_zone availability and basic behavior using our fixtures
    # Build a few points and check classify_zone outputs match expected zones
    pts = [A_point, B_point, C_point, D_point, E_point, F_point]
    expected_ids = ["A", "B", "C", "D", "E", "F"]
    # Construct small bbox around each point and verify module classify_zone matches the intended zone
    for (x, y), expected in zip(pts, expected_ids):
        bbox = bbox_from_point(x, y)
        cx, cy = mod.compute_centroid(bbox)
        # Ensure compute_centroid matches test helper output
        cx2, cy2 = centroid(bbox)
        assert (cx, cy) == (cx2, cy2)
        # Module classify_zone should correctly identify the zone
        zone_id = mod.classify_zone((cx, cy), zones_rect_grid)
        assert zone_id == expected, f"Expected {expected} for point {(x,y)}, got {zone_id}"
