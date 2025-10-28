#!/usr/bin/env python3
"""
Test script to demonstrate OCR date label extraction integration.

This script shows how the OCR functionality integrates with the video pipeline
to extract date labels and compute in_video_time from frames.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from yolo_pipeline.ocr import extract_date_label_from_image_path, extract_date_label_from_frame

def test_image_extraction():
    """Test OCR extraction from a static image."""
    print("=" * 60)
    print("TEST 1: Extract date label from static image")
    print("=" * 60)
    
    # Test with provided attachment
    image_path = "/home/kavia/workspace/code-generation/attachments/20251028_173034_image.png"
    text, in_video_time = extract_date_label_from_image_path(image_path)
    
    print(f"Image: {image_path}")
    print(f"Extracted Text: {text}")
    print(f"In Video Time: {in_video_time} (None expected for static images)")
    print()

def test_frame_with_timing():
    """Test OCR extraction from frame with video timing information."""
    print("=" * 60)
    print("TEST 2: Extract date label with video timing")
    print("=" * 60)
    
    try:
        import cv2
        
        # Load image as if it's a video frame
        image_path = "/home/kavia/workspace/code-generation/attachments/20251028_173333_image.png"
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Failed to load image: {image_path}")
            return
        
        # Simulate video context: frame 300 at 30 fps
        frame_index = 300
        fps = 30.0
        
        text, in_video_time = extract_date_label_from_frame(
            frame,
            frame_index=frame_index,
            fps=fps
        )
        
        print(f"Frame: {image_path}")
        print(f"Frame Index: {frame_index}")
        print(f"FPS: {fps}")
        print(f"Extracted Text: {text}")
        print(f"In Video Time: {in_video_time}s (computed as {frame_index}/{fps})")
        print()
        
    except ImportError:
        print("OpenCV not available, skipping frame test")
        print()

def test_different_frame_rates():
    """Test in_video_time calculation with different frame rates."""
    print("=" * 60)
    print("TEST 3: In-video time calculation examples")
    print("=" * 60)
    
    try:
        import cv2
        
        image_path = "/home/kavia/workspace/code-generation/attachments/20251009_083748_image.png"
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Failed to load image: {image_path}")
            return
        
        # Test different scenarios
        scenarios = [
            (0, 30.0, "First frame at 30 fps"),
            (30, 30.0, "Frame 30 at 30 fps (1 second)"),
            (150, 30.0, "Frame 150 at 30 fps (5 seconds)"),
            (600, 24.0, "Frame 600 at 24 fps (25 seconds)"),
        ]
        
        for frame_idx, fps, description in scenarios:
            text, in_video_time = extract_date_label_from_frame(
                frame,
                frame_index=frame_idx,
                fps=fps
            )
            print(f"{description}:")
            print(f"  Frame {frame_idx} / {fps} fps = {in_video_time}s")
        
        print()
        
    except ImportError:
        print("OpenCV not available, skipping timing test")
        print()

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OCR Date Label Extraction Integration Tests")
    print("=" * 60 + "\n")
    
    test_image_extraction()
    test_frame_with_timing()
    test_different_frame_rates()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
