#!/usr/bin/env python3
"""
test_bear_movement.py

Simple test script to validate the bear movement detection system.
This script can be used to test individual components without requiring
a full S3 setup or video processing pipeline.
"""

import os
import sys
import tempfile
import logging
from datetime import datetime, timezone

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from identify_bear_movement import (
        BoundingBox, FrameDetection, MovementState, MovementAnalyzer,
        DatabaseManager, Config, setup_logging
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure identify_bear_movement.py is in the same directory")
    sys.exit(1)


def test_bounding_box():
    """Test BoundingBox functionality."""
    print("Testing BoundingBox...")
    
    box1 = BoundingBox(10, 10, 50, 50, 0.8, "bear")
    box2 = BoundingBox(15, 15, 55, 55, 0.7, "bear")
    
    assert box1.center == (30, 30), f"Expected (30, 30), got {box1.center}"
    assert box1.area == 1600, f"Expected 1600, got {box1.area}"
    
    distance = box1.distance_to(box2)
    expected_distance = 7.07  # approximate
    assert abs(distance - expected_distance) < 0.1, f"Expected ~{expected_distance}, got {distance}"
    
    print("✓ BoundingBox tests passed")


def test_movement_analyzer():
    """Test MovementAnalyzer functionality."""
    print("Testing MovementAnalyzer...")
    
    analyzer = MovementAnalyzer()
    
    # Create test detections
    bear1 = BoundingBox(10, 10, 50, 50, 0.8, "bear")
    bear2 = BoundingBox(15, 15, 55, 55, 0.7, "bear")  # Small movement
    bear3 = BoundingBox(100, 100, 140, 140, 0.9, "bear")  # Large movement
    
    static_obj = BoundingBox(200, 200, 250, 250, 0.9, "tree")
    
    frame1 = FrameDetection(
        frame_timestamp=datetime.now(timezone.utc),
        bears=[bear1],
        static_objects=[static_obj],
        frame_number=1,
        video_timestamp_sec=0.1
    )
    
    frame2 = FrameDetection(
        frame_timestamp=datetime.now(timezone.utc),
        bears=[bear2],
        static_objects=[static_obj],
        frame_number=2,
        video_timestamp_sec=0.2
    )
    
    frame3 = FrameDetection(
        frame_timestamp=datetime.now(timezone.utc),
        bears=[bear3],
        static_objects=[static_obj],
        frame_number=3,
        video_timestamp_sec=0.3
    )
    
    # Test first frame (no previous data)
    state1, conf1, dist1 = analyzer.analyze_movement(frame1)
    assert state1 == MovementState.UNKNOWN, f"Expected UNKNOWN, got {state1}"
    
    # Test small movement
    state2, conf2, dist2 = analyzer.analyze_movement(frame2)
    print(f"Small movement: {state2.value}, confidence: {conf2:.3f}, distance: {dist2}")
    
    # Test large movement
    state3, conf3, dist3 = analyzer.analyze_movement(frame3)
    assert state3 == MovementState.MOVING, f"Expected MOVING, got {state3}"
    print(f"Large movement: {state3.value}, confidence: {conf3:.3f}, distance: {dist3}")
    
    print("✓ MovementAnalyzer tests passed")


def test_database_manager():
    """Test DatabaseManager with SQLite."""
    print("Testing DatabaseManager...")
    
    # Use a temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        test_db_url = f"sqlite:///{tmp_db.name}"
    
    # Override config for testing
    original_db_url = Config.DATABASE_URL
    Config.DATABASE_URL = test_db_url
    
    try:
        db_manager = DatabaseManager()
        
        # Create test detection
        bear = BoundingBox(10, 10, 50, 50, 0.8, "bear")
        static_obj = BoundingBox(200, 200, 250, 250, 0.9, "tree")
        
        detection = FrameDetection(
            frame_timestamp=datetime.now(timezone.utc),
            bears=[bear],
            static_objects=[static_obj],
            frame_number=1,
            video_timestamp_sec=0.1
        )
        
        # Test saving detection
        success = db_manager.save_detection(
            detection, MovementState.MOVING, 0.8, 25.5
        )
        
        assert success, "Failed to save detection"
        print("✓ Database save test passed")
        
    finally:
        # Restore original config
        Config.DATABASE_URL = original_db_url
        # Clean up temp file
        try:
            os.unlink(tmp_db.name)
        except OSError:
            pass
    
    print("✓ DatabaseManager tests passed")


def test_config():
    """Test configuration loading."""
    print("Testing Config...")
    
    # Test default values
    assert Config.S3_BUCKET == "identify_bear_bucket"
    assert Config.DETECTION_THRESHOLD == 0.3
    assert "bear" in Config.BEAR_LABELS
    assert "tree" in Config.STATIC_OBJECT_LABELS
    
    print("✓ Config tests passed")


def main():
    """Run all tests."""
    setup_logging()
    logger = logging.getLogger("test_bear_movement")
    
    print("Running Bear Movement Detection Tests...")
    print("=" * 50)
    
    try:
        test_config()
        test_bounding_box()
        test_movement_analyzer()
        test_database_manager()
        
        print("=" * 50)
        print("✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Test failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
