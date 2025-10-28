"""OCR operations for extracting date labels from video frames."""

import sys
from typing import Optional, Tuple

# Try to import EasyOCR with fallback handling
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("[WARN] EasyOCR not available. Install with: pip install easyocr", file=sys.stderr)
    easyocr = None
    EASYOCR_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[WARN] OpenCV (cv2) not available", file=sys.stderr)
    cv2 = None
    CV2_AVAILABLE = False

from .config import get_labels

# Global reader instance (lazy initialization)
_ocr_reader = None

def get_ocr_reader():
    """
    Get or initialize the EasyOCR reader instance.
    Uses lazy initialization to avoid loading the model unless needed.
    
    Returns:
        easyocr.Reader or None: The OCR reader instance if available.
    """
    global _ocr_reader
    
    if not EASYOCR_AVAILABLE:
        return None
    
    if _ocr_reader is None:
        try:
            # Initialize with English language support
            # gpu=False ensures it works without CUDA
            _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[INFO] EasyOCR reader initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize EasyOCR reader: {e}", file=sys.stderr)
            return None
    
    return _ocr_reader

def extract_region_from_frame(frame, polygon_points):
    """
    Extract a rectangular region from a frame based on polygon coordinates.
    
    Args:
        frame: OpenCV frame (numpy array)
        polygon_points: List of dictionaries with 'x' and 'y' keys defining the polygon
        
    Returns:
        numpy.ndarray or None: Cropped region as an image
    """
    if not CV2_AVAILABLE or frame is None or not polygon_points:
        return None
    
    try:
        # Get bounding box from polygon points
        x_coords = [int(p['x']) for p in polygon_points]
        y_coords = [int(p['y']) for p in polygon_points]
        
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract region
        region = frame[y1:y2, x1:x2].copy()
        return region
        
    except Exception as e:
        print(f"[WARN] Failed to extract region from frame: {e}", file=sys.stderr)
        return None

# PUBLIC_INTERFACE
def extract_date_label_from_frame(frame, frame_index: Optional[int] = None, fps: Optional[float] = None) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract the date label text from a video frame using OCR.
    
    This function:
    1. Extracts the DATE region from the frame using the label polygon from config.py
    2. Runs OCR on that region to extract text
    3. Computes in_video_time based on frame_index and fps if provided
    
    Args:
        frame: OpenCV frame (BGR format numpy array)
        frame_index: Optional frame number in the video (0-based)
        fps: Optional frames per second of the video
        
    Returns:
        tuple: (extracted_text: str or None, in_video_time: float or None)
            - extracted_text: The OCR-extracted text from the date label region
            - in_video_time: Time in seconds computed from frame_index/fps, or None
            
    Note:
        If frame_index and fps are both provided, in_video_time is computed as:
            in_video_time = frame_index / fps
        Otherwise, in_video_time will be None.
        
    Example:
        >>> text, time = extract_date_label_from_frame(frame, frame_index=300, fps=30.0)
        >>> print(f"Extracted: {text} at {time}s")
        Extracted: 2025-10-28 17:30:34 at 10.0s
    """
    if not EASYOCR_AVAILABLE or not CV2_AVAILABLE:
        print("[WARN] OCR extraction skipped: EasyOCR or OpenCV not available", file=sys.stderr)
        return None, None
    
    if frame is None:
        return None, None
    
    # Compute in_video_time if frame_index and fps are provided
    in_video_time = None
    if frame_index is not None and fps is not None and fps > 0:
        in_video_time = float(frame_index) / float(fps)
    
    try:
        # Get the DATE label polygon from config
        labels = get_labels()
        if 'DATE' not in labels:
            print("[WARN] DATE label not found in config", file=sys.stderr)
            return None, in_video_time
        
        date_polygon = labels['DATE']
        
        # Extract the date region from frame
        date_region = extract_region_from_frame(frame, date_polygon)
        if date_region is None or date_region.size == 0:
            print("[WARN] Failed to extract DATE region from frame", file=sys.stderr)
            return None, in_video_time
        
        # Get OCR reader
        reader = get_ocr_reader()
        if reader is None:
            return None, in_video_time
        
        # Run OCR on the date region
        # EasyOCR returns list of tuples: (bbox, text, confidence)
        results = reader.readtext(date_region)
        
        if not results:
            print("[DEBUG] No text detected in DATE region", file=sys.stderr)
            return None, in_video_time
        
        # Combine all detected text segments
        # Sort by y-coordinate first, then x-coordinate to maintain reading order
        sorted_results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
        extracted_texts = [result[1] for result in sorted_results]
        full_text = ' '.join(extracted_texts).strip()
        
        if full_text:
            print(f"[INFO] OCR extracted text: '{full_text}'", file=sys.stderr)
            return full_text, in_video_time
        else:
            return None, in_video_time
            
    except Exception as e:
        print(f"[ERROR] OCR extraction failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None, in_video_time

# PUBLIC_INTERFACE
def extract_date_label_from_image_path(image_path: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract date label from an image file.
    
    This is a convenience function for processing single images (e.g., from API uploads).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (extracted_text: str or None, in_video_time: None)
            Note: in_video_time is always None for single images since no video context exists
            
    Example:
        >>> text, _ = extract_date_label_from_image_path('/tmp/frame.jpg')
        >>> print(f"Date label: {text}")
    """
    if not CV2_AVAILABLE:
        print("[ERROR] OpenCV required for image loading", file=sys.stderr)
        return None, None
    
    try:
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Failed to load image from {image_path}", file=sys.stderr)
            return None, None
        
        # Extract date label (no frame_index/fps for single images)
        text, _ = extract_date_label_from_frame(frame, frame_index=None, fps=None)
        return text, None
        
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}", file=sys.stderr)
        return None, None
