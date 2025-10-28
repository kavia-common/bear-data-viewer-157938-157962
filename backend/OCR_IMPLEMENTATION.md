# OCR Date Label Extraction Implementation

## Overview

This implementation adds OCR-based date label extraction from video frames using EasyOCR. The system extracts text from the DATE region defined in `config.py` and computes `in_video_time` based on frame index and FPS.

## Changes Summary

### Files Modified

1. **`yolo_pipeline/config.py`**
   - Fixed `get_labels()` to return the zones dictionary containing the DATE label region

2. **`yolo_pipeline/ocr.py`** (NEW)
   - Added OCR functionality using EasyOCR
   - Implements `extract_date_label_from_frame()` for frame-by-frame extraction
   - Implements `extract_date_label_from_image_path()` for single image processing
   - Handles region extraction from frames based on polygon coordinates
   - Computes `in_video_time` from frame_index and fps

3. **`yolo_pipeline/__init__.py`**
   - Exports OCR functions for use in other modules

4. **`yolo_pipeline/database.py`**
   - Updated table schema from `animal_detections_v2` to `animal_detections_v3`
   - Added `in_video_time DECIMAL(10,3)` column
   - Added `ocr_date_text VARCHAR(512)` column
   - Updated `insert_detection()` function signature to accept new parameters

5. **`main.py`**
   - Integrated OCR extraction in the video processing loop
   - Extracts date labels for each frame
   - Computes `in_video_time` as `frame_index / fps`
   - Passes OCR results to database insertion

6. **`app/routes/bears.py`**
   - Updated `/api/bears` endpoint to query `animal_detections_v3` table
   - Added `in_video_time` and `ocr_date_text` fields to response schema
   - Added new `/api/extract-date-label` endpoint for testing OCR on uploaded images
   - Falls back to mock data if database is unavailable

7. **`requirements.txt`**
   - Added `easyocr>=1.7.0`
   - Added `opencv-python-headless>=4.8.0`

## Table Schema v3

```sql
CREATE TABLE IF NOT EXISTS animal_detections_v3 (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    frame_time_seconds DECIMAL(10,3) NOT NULL,
    in_video_time DECIMAL(10,3) DEFAULT NULL,      -- NEW COLUMN
    detection_label VARCHAR(100) NOT NULL,
    bbox_x1 INT NOT NULL,
    bbox_y1 INT NOT NULL,
    bbox_x2 INT NOT NULL,
    bbox_y2 INT NOT NULL,
    detection_confidence DECIMAL(6,4) NOT NULL,
    classification_label VARCHAR(255) DEFAULT NULL,
    classification_confidence DECIMAL(6,4) DEFAULT NULL,
    pose_status VARCHAR(50) DEFAULT NULL,
    pose_keypoints_count INT DEFAULT 0,
    frame_s3_url VARCHAR(1024) DEFAULT NULL,
    video_source VARCHAR(512) DEFAULT NULL,
    ocr_date_text VARCHAR(512) DEFAULT NULL,       -- NEW COLUMN
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_frame_time (frame_time_seconds),
    INDEX idx_in_video_time (in_video_time),       -- NEW INDEX
    INDEX idx_label (detection_label),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### New Columns

- **`in_video_time`**: Time in seconds computed as `frame_index / fps`
  - This represents the actual video timeline position
  - Computed during frame processing
  - Can be different from `frame_time_seconds` which is the sampling time

- **`ocr_date_text`**: OCR-extracted text from the DATE label region
  - Contains the date/time overlay from the video frame
  - Example: "2025-10-28 17:30:34"
  - NULL if OCR extraction failed or no text detected

## API Changes

### GET /api/bears

**Response Schema:**
```json
{
  "bearId": "B0001",
  "pose": "Sitting",
  "timestamp": "2025-01-15T10:30:00Z",
  "in_video_time": 10.5,
  "ocr_date_text": "2025-10-28 17:30:34"
}
```

### POST /api/extract-date-label (NEW)

Test endpoint to extract date labels from uploaded images.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` field containing the image file

**Response:**
```json
{
  "success": true,
  "message": "Date label extracted successfully",
  "extracted_text": "2025-10-28 17:30:34",
  "in_video_time": null
}
```

**Example Usage:**
```bash
curl -X POST -F "image=@/path/to/frame.jpg" http://localhost:5000/api/extract-date-label
```

## How `in_video_time` is Computed

The `in_video_time` field represents the actual position in the video timeline:

```python
# During video processing in main.py:
for idx, t in enumerate(sample_times):
    # Read frame at time t
    ok, frame = seek_and_read_frame(cap, t)
    
    # Compute frame index from time and fps
    frame_index = int(round(t * fps)) if fps > 0 else None
    
    # Extract OCR with timing info
    ocr_date_text, in_video_time = extract_date_label_from_frame(
        frame, 
        frame_index=frame_index,  # Actual frame number
        fps=fps                    # Frames per second
    )
    
    # in_video_time = frame_index / fps
    # Example: frame 300 at 30fps = 10.0 seconds
```

### When to Provide Timestamp vs Frame Index/FPS

**Option 1: Provide frame_index and fps**
```python
text, time = extract_date_label_from_frame(frame, frame_index=300, fps=30.0)
# time = 300 / 30.0 = 10.0 seconds
```

**Option 2: For single images (no video context)**
```python
text, time = extract_date_label_from_image_path('/path/to/image.jpg')
# time = None (no video context)
```

**Option 3: No timing information**
```python
text, time = extract_date_label_from_frame(frame)
# time = None
```

## Dependencies

### EasyOCR
- Library: `easyocr>=1.7.0`
- Purpose: Text extraction from images
- GPU support: Works with or without GPU (uses `gpu=False` flag)
- Languages: Initialized with English support

### OpenCV
- Library: `opencv-python-headless>=4.8.0`
- Purpose: Image loading and manipulation
- Headless version avoids display dependencies

## Error Handling

The implementation includes comprehensive error handling:

1. **Missing Dependencies**: Falls back gracefully if EasyOCR or OpenCV not available
2. **Import Errors**: Try/except blocks with clear warning messages
3. **OCR Failures**: Returns None for extracted_text if OCR fails
4. **Database Errors**: API falls back to mock data if DB unavailable
5. **Invalid Regions**: Validates bounding box coordinates before extraction

## Testing

### Test Script
Run `test_ocr_integration.py` to verify functionality:

```bash
python3 test_ocr_integration.py
```

### Test Scenarios
1. Extract from static image
2. Extract with video timing information
3. Calculate in_video_time with different frame rates

### Expected Output
```
Extracted Text: 2025-10-28 17:30:34
In Video Time: 10.0s (for frame 300 at 30fps)
```

## Integration with Existing Pipeline

The OCR functionality integrates seamlessly with the existing YOLO pipeline:

1. **Video Processing Loop** (`main.py`):
   - For each sampled frame, OCR extracts date label
   - Results stored alongside detection data

2. **Database Storage** (`database.py`):
   - New columns added to v3 schema
   - OCR data saved with each detection

3. **API Response** (`app/routes/bears.py`):
   - API now returns OCR and timing data
   - Clients can display in_video_time for each detection

## Limitations

1. **OCR Accuracy**: Depends on text quality in DATE region
2. **Performance**: OCR adds processing time per frame (~1-2 seconds)
3. **DATE Region**: Must be defined in `config.py` get_labels()
4. **Language Support**: Currently configured for English only

## Future Enhancements

1. Add support for multiple languages
2. Cache OCR reader initialization
3. Batch OCR processing for better performance
4. Add OCR confidence scores to database
5. Support custom DATE region coordinates via API
