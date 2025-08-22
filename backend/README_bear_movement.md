# Bear Movement Detection System

A modular Python script that downloads videos from S3, uses YOLOv8-lite to detect bears and static objects, analyzes movement patterns, and persists results to a MySQL database.

## Features

- **Video Processing**: Downloads MP4 videos from S3 bucket `identify_bear_bucket`
- **Object Detection**: Uses YOLOv8-lite to detect bears and static enclosure objects (trees, hammocks, cages)
- **Movement Analysis**: Determines if bears are moving or resting relative to static objects
- **Database Persistence**: Stores detection results with bounding boxes, confidence scores, timestamps, and movement states
- **Modular Design**: Well-structured, maintainable code with clear separation of concerns
- **Configuration**: Fully configurable via environment variables
- **Error Handling**: Comprehensive error handling and logging
- **Graceful Degradation**: Works even when optional dependencies are missing

## Architecture

### Core Components

1. **DatabaseManager**: Handles MySQL/SQLite connections and data persistence
2. **S3Downloader**: Manages video downloads from S3
3. **BearDetector**: YOLO model loading and object detection
4. **MovementAnalyzer**: Analyzes bear movement between frames
5. **VideoProcessor**: Frame-by-frame video processing
6. **BearMovementDetector**: Main orchestration class

### Data Models

- **BoundingBox**: Represents detected objects with coordinates and metadata
- **FrameDetection**: Contains all detections for a single frame
- **BearMovementDetection**: Database model for persisting results

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Required Dependencies

- **boto3**: S3 operations
- **ultralytics**: YOLOv8 model
- **opencv-python**: Video processing
- **sqlalchemy**: Database operations
- **numpy**: Numerical operations

### Optional Dependencies

- **PyMySQL**: MySQL database support (SQLite used by default)
- **python-dotenv**: Environment variable loading

## Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your configuration
```

### Key Environment Variables

#### S3 Configuration
- `BEAR_S3_BUCKET`: S3 bucket name (default: "identify_bear_bucket")
- `BEAR_S3_KEY`: Video file key (default: "bears.mp4")
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_DEFAULT_REGION`: AWS region (default: "us-east-1")

#### Detection Configuration
- `YOLO_MODEL_PATH`: YOLO model path (default: "yolov8n.pt")
- `DETECTION_THRESHOLD`: Minimum detection confidence (default: 0.3)
- `MOVEMENT_THRESHOLD_PIXELS`: Movement threshold in pixels (default: 30.0)
- `STATIC_OBJECT_LABELS`: Comma-separated static object labels (default: "tree,hammock,cage")
- `BEAR_LABELS`: Comma-separated bear labels (default: "bear")

#### Database Configuration
- `DATABASE_URL`: Database connection string
  - MySQL: `mysql+pymysql://username:password@localhost:3306/bear_data`
  - SQLite: `sqlite:///bear_movement.db` (default)

## Usage

### Running the Detection System

```bash
# Make sure your .env file is configured
python identify_bear_movement.py
```

### Testing the System

```bash
# Run validation tests
python test_bear_movement.py
```

### Integration with Existing Flask App

The detection system runs independently but shares the same database. The Flask app can query the `bear_movement_detections` table for real-time data.

## Database Schema

### `bear_movement_detections` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `frame_number` | Integer | Frame sequence number |
| `frame_timestamp` | DateTime | UTC timestamp when frame was processed |
| `video_timestamp_sec` | Float | Timestamp within the video |
| `bear_id` | String | Bear identifier (default: "011") |
| `bear_x1`, `bear_y1`, `bear_x2`, `bear_y2` | Float | Bear bounding box coordinates |
| `bear_confidence` | Float | YOLO detection confidence |
| `movement_state` | String | "moving", "resting", or "unknown" |
| `movement_confidence` | Float | Movement analysis confidence |
| `distance_moved` | Float | Distance moved since last frame |
| `static_objects` | JSON | Array of static object detections |
| `processing_timestamp` | DateTime | When the record was created |
| `metadata` | JSON | Additional processing metadata |

## Algorithm Details

### Movement Detection

1. **Object Detection**: YOLOv8 detects all objects in each frame
2. **Classification**: Separates bears from static objects based on labels
3. **Position Tracking**: Compares bear positions between consecutive frames
4. **Movement Calculation**: Uses Euclidean distance between bounding box centers
5. **State Determination**: 
   - `MOVING`: Distance > threshold
   - `RESTING`: Distance ≤ threshold
   - `UNKNOWN`: No previous frame data

### Confidence Scoring

- **Detection Confidence**: YOLO model confidence score
- **Movement Confidence**: Based on distance relative to threshold
  - Moving: `min(1.0, distance / (threshold * 2))`
  - Resting: `1.0 - (distance / threshold)`

## Performance Considerations

- **Frame Rate**: Configurable processing delay via `FRAME_PROCESSING_DELAY`
- **Memory Usage**: Uses temporary files for video storage
- **Database**: Connection pooling for efficient database operations
- **Model Size**: YOLOv8n (nano) for optimal speed/accuracy balance

## Error Handling

- **Graceful Degradation**: Continues operation even if optional dependencies are missing
- **Retry Logic**: Automatic retry for S3 downloads and database operations
- **Logging**: Comprehensive logging at multiple levels (DEBUG, INFO, WARNING, ERROR)
- **Exception Handling**: Proper exception handling with recovery mechanisms

## Monitoring and Logging

The system provides detailed logging for:
- Video download status
- Detection statistics
- Movement analysis results
- Database operations
- Error conditions

Log levels can be controlled via the `LOG_LEVEL` environment variable.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install boto3 ultralytics opencv-python sqlalchemy
   ```

2. **S3 Access Issues**
   - Verify AWS credentials in `.env`
   - Check bucket permissions
   - Ensure bucket and key exist

3. **YOLO Model Issues**
   - Model will be downloaded automatically on first run
   - Ensure internet connectivity for model download
   - Check `YOLO_MODEL_PATH` configuration

4. **Database Connection Issues**
   - Verify `DATABASE_URL` format
   - For MySQL: ensure server is running and accessible
   - For SQLite: ensure write permissions in directory

### Performance Tuning

- Adjust `FRAME_PROCESSING_DELAY` for speed vs. CPU usage
- Modify `DETECTION_THRESHOLD` for detection sensitivity
- Configure `MOVEMENT_THRESHOLD_PIXELS` based on video resolution
- Use database connection pooling settings for high throughput

## Development

### Adding New Features

1. **New Object Types**: Add labels to `STATIC_OBJECT_LABELS` or `BEAR_LABELS`
2. **Movement Algorithms**: Extend `MovementAnalyzer` class
3. **Database Schema**: Use SQLAlchemy migrations for schema changes
4. **Detection Models**: Replace YOLO model via `YOLO_MODEL_PATH`

### Testing

```bash
# Run comprehensive tests
python test_bear_movement.py

# Test individual components
python -c "from identify_bear_movement import BoundingBox; print('Import successful')"
```

## License

This project follows the same license as the parent project.
