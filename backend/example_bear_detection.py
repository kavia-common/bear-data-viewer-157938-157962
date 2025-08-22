#!/usr/bin/env python3
"""
example_bear_detection.py

Example script demonstrating how to use the bear movement detection system
without requiring actual S3 or video files. This creates mock data to show
the system functionality.
"""

import os
import sys
import tempfile
from datetime import datetime, timezone

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from identify_bear_movement import (
        BoundingBox, FrameDetection, MovementAnalyzer,
        DatabaseManager, Config, setup_logging
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_mock_frame_data():
    """Create mock frame detection data for demonstration."""
    frames = []
    
    # Frame 1: Bear at starting position
    bear1 = BoundingBox(100, 100, 150, 150, 0.85, "bear")
    tree1 = BoundingBox(50, 50, 80, 120, 0.92, "tree")
    hammock1 = BoundingBox(200, 80, 280, 120, 0.78, "hammock")
    
    frame1 = FrameDetection(
        frame_timestamp=datetime.now(timezone.utc),
        bears=[bear1],
        static_objects=[tree1, hammock1],
        frame_number=1,
        video_timestamp_sec=0.033
    )
    frames.append(frame1)
    
    # Frame 2: Bear moves slightly (resting)
    bear2 = BoundingBox(105, 102, 155, 152, 0.87, "bear")
    frame2 = FrameDetection(
        frame_timestamp=datetime.now(timezone.utc),
        bears=[bear2],
        static_objects=[tree1, hammock1],
        frame_number=2,
        video_timestamp_sec=0.066
    )
    frames.append(frame2)
    
    # Frame 3: Bear moves significantly (moving)
    bear3 = BoundingBox(150, 120, 200, 170, 0.89, "bear")
    frame3 = FrameDetection(
        frame_timestamp=datetime.now(timezone.utc),
        bears=[bear3],
        static_objects=[tree1, hammock1],
        frame_number=3,
        video_timestamp_sec=0.099
    )
    frames.append(frame3)
    
    # Frame 4: Bear moves to hammock area (moving)
    bear4 = BoundingBox(190, 110, 240, 160, 0.91, "bear")
    frame4 = FrameDetection(
        frame_timestamp=datetime.now(timezone.utc),
        bears=[bear4],
        static_objects=[tree1, hammock1],
        frame_number=4,
        video_timestamp_sec=0.132
    )
    frames.append(frame4)
    
    # Frame 5: Bear settles near hammock (resting)
    bear5 = BoundingBox(192, 112, 242, 162, 0.88, "bear")
    frame5 = FrameDetection(
        frame_timestamp=datetime.now(timezone.utc),
        bears=[bear5],
        static_objects=[tree1, hammock1],
        frame_number=5,
        video_timestamp_sec=0.165
    )
    frames.append(frame5)
    
    return frames


def demonstrate_movement_detection():
    """Demonstrate the movement detection system with mock data."""
    print("Bear Movement Detection System Demo")
    print("=" * 50)
    
    # Setup components
    setup_logging()
    
    # Use temporary database for demo
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        test_db_url = f"sqlite:///{tmp_db.name}"
    
    # Override config for demo
    original_db_url = Config.DATABASE_URL
    Config.DATABASE_URL = test_db_url
    
    try:
        # Initialize components
        analyzer = MovementAnalyzer()
        db_manager = DatabaseManager()
        
        # Generate mock data
        frames = create_mock_frame_data()
        
        print(f"\nProcessing {len(frames)} mock video frames...")
        print("-" * 50)
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Analyze movement
            movement_state, confidence, distance = analyzer.analyze_movement(frame)
            
            # Display results
            bear_count = len(frame.bears)
            static_count = len(frame.static_objects)
            
            distance_str = f"{distance:.1f}" if distance is not None else "N/A"
            print(f"Frame {frame.frame_number:2d}: "
                  f"Bears={bear_count}, Static={static_count}, "
                  f"Movement={movement_state.value:8s}, "
                  f"Confidence={confidence:.3f}, "
                  f"Distance={distance_str:>6s}")
            
            # Save to database
            success = db_manager.save_detection(frame, movement_state, confidence, distance)
            if not success:
                print(f"  ⚠️  Failed to save frame {frame.frame_number}")
            
            # Show bear positions for first few frames
            if i < 3 and frame.bears:
                bear = frame.bears[0]
                center_x, center_y = bear.center
                print(f"      Bear center: ({center_x:.1f}, {center_y:.1f}), "
                      f"Confidence: {bear.confidence:.3f}")
        
        print("\n" + "=" * 50)
        print("✓ Demo completed successfully!")
        print(f"\nResults saved to: {test_db_url}")
        print("\nTo view the database contents, you can use:")
        print(f"sqlite3 {tmp_db.name}")
        print("SELECT frame_number, movement_state, movement_confidence, distance_moved FROM bear_movement_detections;")
        
    finally:
        # Restore original config
        Config.DATABASE_URL = original_db_url
        print(f"\nTemporary database file: {tmp_db.name}")
        print("(You can delete this file when done)")


def show_configuration():
    """Display current configuration settings."""
    print("\nCurrent Configuration:")
    print("-" * 30)
    print(f"S3 Bucket: {Config.S3_BUCKET}")
    print(f"S3 Key: {Config.S3_KEY}")
    print(f"YOLO Model: {Config.YOLO_MODEL_PATH}")
    print(f"Detection Threshold: {Config.DETECTION_THRESHOLD}")
    print(f"Movement Threshold: {Config.MOVEMENT_THRESHOLD_PIXELS} pixels")
    print(f"Static Objects: {', '.join(Config.STATIC_OBJECT_LABELS)}")
    print(f"Bear Labels: {', '.join(Config.BEAR_LABELS)}")
    print(f"Database URL: {Config.DATABASE_URL}")


def main():
    """Main demonstration function."""
    print("Bear Movement Detection System")
    print("Example Usage Demonstration")
    print("=" * 50)
    
    show_configuration()
    demonstrate_movement_detection()


if __name__ == "__main__":
    main()
