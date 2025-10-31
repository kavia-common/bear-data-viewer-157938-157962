"""
Backend package initializer for bear-data-viewer project.
Exposes anteater pacing detection public APIs for tests.
"""

from .detect_anteater_pacing import (  # noqa: F401
    ZONES,
    PacingEvent,
    compute_centroid_from_bbox,
    map_centroid_to_zone,
    detect_pacing_events,
    detect_pacing_events_stream,
    classify_zone_for_bbox,
)
