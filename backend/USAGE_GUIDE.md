# Bear Movement Detection System - Usage Guide

This guide provides step-by-step instructions for setting up and using the Bear Movement Detection System.

## Overview

The Bear Movement Detection System consists of:

1. **Core Detection Script** (`identify_bear_movement.py`) - Downloads videos from S3, detects bears and static objects, analyzes movement
2. **Flask API Integration** - REST endpoints to access detection data
3. **Database Storage** - MySQL/SQLite persistence for detection results
4. **Testing Tools** - Scripts to validate and demonstrate functionality

## Quick Start

### 1. Environment Setup

```bash
# Navigate to backend directory
cd bear-data-viewer-157938-157962/backend

# Install dependencies (already done)
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your settings:

```bash
# Required: AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
BEAR_S3_BUCKET=identify_bear_bucket
BEAR_S3_KEY=bears.mp4

# Optional: Database (defaults to SQLite)
DATABASE_URL=sqlite:///bear_movement.db
# For MySQL: DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/beardb

# Optional: Detection tuning
DETECTION_THRESHOLD=0.3
MOVEMENT_THRESHOLD_PIXELS=30.0
```

### 3. Test the System

```bash
# Run basic functionality tests
python test_bear_movement.py

# Run demonstration with mock data
python example_bear_detection.py
```

### 4. Start Detection System

```bash
# Run the main detection script
python identify_bear_movement.py
```

### 5. Access Data via API

```bash
# Start Flask application (in another terminal)
python run.py

# Test API endpoints
curl http://localhost:5000/api/bear-movements
curl http://localhost:5000/api/bear-movements/summary
```

## System Components

### Core Detection Script

**File:** `identify_bear_movement.py`

**Purpose:** Main detection pipeline that:
- Downloads MP4 videos from S3 bucket `identify_bear_bucket`
- Uses YOLOv8-lite (nano) for object detection
- Identifies bears and static objects (trees, hammocks, cages)
- Analyzes movement between frames
- Stores results in database

**Key Features:**
- Modular design with clear separation of concerns
- Configurable via environment variables
- Comprehensive error handling and logging
- Graceful degradation when dependencies are missing
- Real-time processing with configurable frame rate

### Database Schema

**Table:** `bear_movement_detections`

**Key Fields:**
- `frame_number` - Video frame sequence
- `movement_state` - "moving", "resting", or "unknown"
- `bear_x1`, `bear_y1`, `bear_x2`, `bear_y2` - Bounding box coordinates
- `bear_confidence` - YOLO detection confidence
- `movement_confidence` - Movement analysis confidence
- `distance_moved` - Pixels moved since previous frame
- `static_objects` - JSON array of detected static objects

### API Endpoints

**Base URL:** `http://localhost:5000/api`

#### GET /bear-movements
Returns recent bear movement detection records.

**Query Parameters:**
- `limit` (default: 50, max: 500) - Number of records
- `hours` (default: 24) - Hours back to query
- `movement_state` - Filter by state: "moving", "resting", "unknown"

**Example:**
```bash
curl "http://localhost:5000/api/bear-movements?limit=10&movement_state=moving"
```

#### GET /bear-movements/summary
Returns summary statistics for movement data.

**Query Parameters:**
- `hours` (default: 24) - Hours back to analyze

**Example Response:**
```json
{
  "total_detections": 1247,
  "moving_count": 523,
  "resting_count": 678,
  "unknown_count": 46,
  "avg_confidence": 0.847,
  "latest_timestamp": "2024-01-15T14:30:22.123Z"
}
```

## Configuration Options

### Detection Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | "yolov8n.pt" | YOLO model file (nano for speed) |
| `DETECTION_THRESHOLD` | 0.3 | Minimum detection confidence |
| `MOVEMENT_THRESHOLD_PIXELS` | 30.0 | Movement detection threshold |
| `STATIC_OBJECT_LABELS` | "tree,hammock,cage" | Static object types |
| `BEAR_LABELS` | "bear" | Bear detection labels |

### Processing Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `DOWNLOAD_INTERVAL_SEC` | 300 | S3 video refresh interval |
| `FRAME_PROCESSING_DELAY` | 0.1 | Delay between frames (seconds) |
| `LOG_LEVEL` | "INFO" | Logging level |

### Database Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | "sqlite:///bear_movement.db" | Database connection |
| `DB_POOL_SIZE` | 5 | Connection pool size |
| `DB_MAX_OVERFLOW` | 10 | Max overflow connections |

## Development Workflow

### Adding New Object Types

1. Update `STATIC_OBJECT_LABELS` or `BEAR_LABELS` in `.env`
2. Restart detection system
3. New objects will be automatically detected

### Custom Movement Algorithms

1. Extend the `MovementAnalyzer` class in `identify_bear_movement.py`
2. Override the `_calculate_movement` method
3. Test with `test_bear_movement.py`

### Database Schema Changes

1. Modify the `BearMovementDetection` model
2. Use SQLAlchemy migrations for production
3. Update API schemas in `bear_movement.py`

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install boto3 ultralytics opencv-python sqlalchemy
```

**2. S3 Access Denied**
```bash
# Check AWS credentials in .env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Verify bucket permissions
aws s3 ls s3://identify_bear_bucket
```

**3. YOLO Model Download**
```bash
# Model downloads automatically on first run
# Ensure internet connectivity
# Check logs for download progress
```

**4. Database Connection**
```bash
# For SQLite (default)
DATABASE_URL=sqlite:///bear_movement.db

# For MySQL
DATABASE_URL=mysql+pymysql://user:pass@host:3306/db
```

### Performance Tuning

**Speed vs. Accuracy:**
- Use `yolov8n.pt` (nano) for speed
- Use `yolov8s.pt` (small) for better accuracy
- Adjust `FRAME_PROCESSING_DELAY` for CPU usage

**Memory Usage:**
- Increase `DOWNLOAD_INTERVAL_SEC` for less frequent downloads
- Reduce video resolution if processing is slow
- Monitor database size and implement data retention

### Monitoring

**Log Analysis:**
```bash
# Real-time monitoring
tail -f bear_movement.log

# Error analysis
grep ERROR bear_movement.log

# Performance metrics
grep "Processed.*frames" bear_movement.log
```

**Database Monitoring:**
```sql
-- Recent activity
SELECT movement_state, COUNT(*) 
FROM bear_movement_detections 
WHERE frame_timestamp > datetime('now', '-1 hour')
GROUP BY movement_state;

-- Performance stats
SELECT 
    AVG(movement_confidence) as avg_confidence,
    AVG(distance_moved) as avg_distance
FROM bear_movement_detections 
WHERE movement_state = 'moving';
```

## Integration Examples

### React Frontend Integration

```javascript
// Fetch recent movement data
const fetchBearMovements = async () => {
  const response = await fetch('/api/bear-movements?limit=20');
  const movements = await response.json();
  return movements;
};

// Real-time updates
setInterval(async () => {
  const summary = await fetch('/api/bear-movements/summary');
  updateDashboard(await summary.json());
}, 10000); // Update every 10 seconds
```

### Data Analysis

```python
# Connect to database for analysis
from identify_bear_movement import DatabaseManager

db = DatabaseManager()
session = db.SessionLocal()

# Analyze movement patterns
movements = session.query(BearMovementDetection)\
    .filter(BearMovementDetection.movement_state == 'moving')\
    .all()

# Calculate average movement distance
avg_distance = sum(m.distance_moved for m in movements if m.distance_moved) / len(movements)
```

## Production Deployment

### Security Considerations

1. **Environment Variables:** Never commit `.env` files
2. **Database Security:** Use encrypted connections for MySQL
3. **API Security:** Implement authentication for production APIs
4. **S3 Security:** Use IAM roles with minimal required permissions

### Scaling Considerations

1. **Database:** Use MySQL/PostgreSQL for production
2. **Processing:** Consider multi-threading for high-volume videos
3. **Storage:** Implement data retention policies
4. **Monitoring:** Add health checks and alerting

### Docker Deployment

```dockerfile
# Example Dockerfile (not included)
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "identify_bear_movement.py"]
```

## Support

For issues or questions:

1. Check the logs for error messages
2. Run `test_bear_movement.py` to verify functionality
3. Refer to individual component documentation in `README_bear_movement.md`
4. Check OpenAPI documentation at `/docs` when Flask app is running

## Next Steps

1. **Enhanced Detection:** Train custom YOLO models for better bear detection
2. **Advanced Analytics:** Implement behavioral pattern analysis
3. **Real-time Streaming:** Add support for live video streams
4. **Alert System:** Implement notifications for specific movement patterns
5. **Visualization:** Create dashboards for movement tracking
