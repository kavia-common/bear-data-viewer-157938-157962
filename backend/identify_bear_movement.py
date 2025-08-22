#!/usr/bin/env python3
"""
identify_bear_movement.py

A modular Python script that downloads an mp4 from S3 bucket 'identify_bear_bucket',
uses YOLOv8-lite to detect bear and static enclosure objects (tree, hammock, cage) per frame,
determines if the bear is moving (relative to static objects), and logs each frame's
bear/static object rectangles, YOLO probability, timestamp, and movement/rest state in MySQL DB.

Design principles:
- Modular architecture with clear separation of concerns
- Configuration via environment variables
- Comprehensive error handling and logging
- Type hints and documentation for maintainability
- Database persistence with proper schema design
"""

import os
import sys
import time
import logging
import tempfile
import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Optional dependencies - graceful degradation if not available
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None
    BotoCoreError = ClientError = Exception

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    from sqlalchemy import (
        create_engine, Column, Integer, String, Float, DateTime, 
        JSON, Boolean, Text, Index, UniqueConstraint
    )
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    create_engine = sessionmaker = declarative_base = None
    Column = Integer = String = Float = DateTime = JSON = Boolean = Text = None
    Index = UniqueConstraint = SQLAlchemyError = None


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Centralized configuration management using environment variables."""
    
    # S3 Configuration
    S3_BUCKET = os.environ.get("BEAR_S3_BUCKET", "identify_bear_bucket")
    S3_KEY = os.environ.get("BEAR_S3_KEY", "bears.mp4")
    AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    
    # YOLO Configuration
    YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")  # Using nano for lite version
    DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.3"))
    
    # Movement Detection Configuration
    MOVEMENT_THRESHOLD_PIXELS = float(os.environ.get("MOVEMENT_THRESHOLD_PIXELS", "30.0"))
    STATIC_OBJECT_LABELS = set(os.environ.get("STATIC_OBJECT_LABELS", "tree,hammock,cage").split(","))
    BEAR_LABELS = set(os.environ.get("BEAR_LABELS", "bear").split(","))
    
    # Processing Configuration
    DOWNLOAD_INTERVAL_SEC = float(os.environ.get("DOWNLOAD_INTERVAL_SEC", "300"))  # 5 minutes
    FRAME_PROCESSING_DELAY = float(os.environ.get("FRAME_PROCESSING_DELAY", "0.1"))
    
    # Database Configuration
    DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("MYSQL_URL") or "sqlite:///bear_movement.db"
    DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW = int(os.environ.get("DB_MAX_OVERFLOW", "10"))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


# =============================================================================
# Data Models
# =============================================================================

class MovementState(Enum):
    """Enumeration of possible bear movement states."""
    MOVING = "moving"
    RESTING = "resting"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and metadata."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate the center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        """Calculate the area of the bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """Calculate Euclidean distance to another bounding box center."""
        x1, y1 = self.center
        x2, y2 = other.center
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


@dataclass
class FrameDetection:
    """Represents all detections in a single frame."""
    frame_timestamp: datetime
    bears: List[BoundingBox]
    static_objects: List[BoundingBox]
    frame_number: int
    video_timestamp_sec: float


# =============================================================================
# Database Schema
# =============================================================================

Base = declarative_base() if declarative_base else None

if Base is not None:
    class BearMovementDetection(Base):
        """Database model for bear movement detection results."""
        __tablename__ = "bear_movement_detections"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        
        # Frame metadata
        frame_number = Column(Integer, nullable=False, index=True)
        frame_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
        video_timestamp_sec = Column(Float, nullable=False)
        
        # Detection data
        bear_id = Column(String(64), nullable=False, default="011", index=True)
        bear_x1 = Column(Float, nullable=True)
        bear_y1 = Column(Float, nullable=True)
        bear_x2 = Column(Float, nullable=True)
        bear_y2 = Column(Float, nullable=True)
        bear_confidence = Column(Float, nullable=True)
        
        # Movement analysis
        movement_state = Column(String(20), nullable=False, index=True)
        movement_confidence = Column(Float, nullable=False, default=0.0)
        distance_moved = Column(Float, nullable=True)
        
        # Static objects context (JSON array of objects)
        static_objects = Column(JSON, nullable=True)
        
        # Additional metadata
        processing_timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
        extra_metadata = Column(JSON, nullable=True)
        
        # Indexes for performance
        __table_args__ = (
            Index('idx_bear_frame_time', 'bear_id', 'frame_timestamp'),
            Index('idx_movement_time', 'movement_state', 'frame_timestamp'),
        )


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("bear_movement_detector")


logger = setup_logging()


# =============================================================================
# Database Management
# =============================================================================

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database connection and create schema."""
        if create_engine is None:
            logger.warning("SQLAlchemy not available - using logging fallback")
            return
        
        try:
            self.engine = create_engine(
                Config.DATABASE_URL,
                pool_size=Config.DB_POOL_SIZE,
                max_overflow=Config.DB_MAX_OVERFLOW,
                echo=False,
                future=True
            )
            
            # Create tables if they don't exist
            if Base is not None:
                Base.metadata.create_all(self.engine)
            
            self.SessionLocal = sessionmaker(bind=self.engine, future=True)
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.engine = None
            self.SessionLocal = None
    
    def save_detection(self, detection: FrameDetection, movement_state: MovementState, 
                      movement_confidence: float, distance_moved: Optional[float] = None) -> bool:
        """Save detection results to database."""
        if not self.SessionLocal or BearMovementDetection is None:
            self._log_detection_fallback(detection, movement_state, movement_confidence, distance_moved)
            return True
        
        session = None
        try:
            session = self.SessionLocal()
            
            # Handle multiple bears in frame (though typically one)
            for bear in detection.bears:
                detection_record = BearMovementDetection(
                    frame_number=detection.frame_number,
                    frame_timestamp=detection.frame_timestamp,
                    video_timestamp_sec=detection.video_timestamp_sec,
                    bear_x1=bear.x1,
                    bear_y1=bear.y1,
                    bear_x2=bear.x2,
                    bear_y2=bear.y2,
                    bear_confidence=bear.confidence,
                    movement_state=movement_state.value,
                    movement_confidence=movement_confidence,
                    distance_moved=distance_moved,
                    static_objects=[
                        {
                            "label": obj.label,
                            "x1": obj.x1, "y1": obj.y1, "x2": obj.x2, "y2": obj.y2,
                            "confidence": obj.confidence
                        }
                        for obj in detection.static_objects
                    ],
                    extra_metadata={
                        "total_bears": len(detection.bears),
                        "total_static_objects": len(detection.static_objects),
                        "detection_threshold": Config.DETECTION_THRESHOLD,
                        "movement_threshold": Config.MOVEMENT_THRESHOLD_PIXELS
                    }
                )
                session.add(detection_record)
            
            session.commit()
            logger.debug(f"Saved detection for frame {detection.frame_number}")
            return True
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Failed to save detection to database: {e}")
            return False
        finally:
            if session:
                session.close()
    
    def _log_detection_fallback(self, detection: FrameDetection, movement_state: MovementState, 
                               movement_confidence: float, distance_moved: Optional[float]) -> None:
        """Fallback logging when database is unavailable."""
        logger.info(
            f"Frame {detection.frame_number} | Bears: {len(detection.bears)} | "
            f"Static: {len(detection.static_objects)} | Movement: {movement_state.value} "
            f"(confidence: {movement_confidence:.3f}) | Distance: {distance_moved}"
        )


# =============================================================================
# S3 Operations
# =============================================================================

class S3Downloader:
    """Handles S3 video download operations."""
    
    def __init__(self):
        self.s3_client = self._create_s3_client()
    
    def _create_s3_client(self):
        """Create S3 client with proper configuration."""
        if boto3 is None:
            logger.warning("boto3 not available - S3 operations disabled")
            return None
        
        try:
            return boto3.client('s3', region_name=Config.AWS_REGION)
        except Exception as e:
            logger.error(f"Failed to create S3 client: {e}")
            return None
    
    # PUBLIC_INTERFACE
    def download_video(self, dest_path: str) -> bool:
        """
        Download video from S3 to local path.
        
        Args:
            dest_path: Local destination path for the video file
            
        Returns:
            True if download successful, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not available")
            return False
        
        try:
            logger.info(f"Downloading s3://{Config.S3_BUCKET}/{Config.S3_KEY} to {dest_path}")
            self.s3_client.download_file(Config.S3_BUCKET, Config.S3_KEY, dest_path)
            
            # Verify file exists and has content
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                logger.info(f"Successfully downloaded video ({os.path.getsize(dest_path)} bytes)")
                return True
            else:
                logger.error("Downloaded file is empty or doesn't exist")
                return False
                
        except (BotoCoreError, ClientError) as e:
            logger.error(f"S3 download failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False


# =============================================================================
# YOLO Detection
# =============================================================================

class BearDetector:
    """Handles YOLO model loading and inference."""
    
    def __init__(self):
        self.model = self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        if YOLO is None:
            logger.error("ultralytics not available - detection disabled")
            return None
        
        try:
            logger.info(f"Loading YOLO model: {Config.YOLO_MODEL_PATH}")
            model = YOLO(Config.YOLO_MODEL_PATH)
            logger.info("YOLO model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None
    
    # PUBLIC_INTERFACE
    def detect_objects(self, frame) -> List[BoundingBox]:
        """
        Detect objects in a frame using YOLO.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            List of BoundingBox objects for detected objects
        """
        if not self.model:
            return []
        
        try:
            results = self.model(frame, verbose=False)
            detections = []
            
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                
                boxes = result.boxes
                names = getattr(result, 'names', {})
                
                # Extract tensors and convert to lists
                if hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
                    coords = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                    classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                    
                    for coord, conf, cls in zip(coords, confs, classes):
                        if conf >= Config.DETECTION_THRESHOLD:
                            label = names.get(int(cls), str(int(cls)))
                            
                            detection = BoundingBox(
                                x1=float(coord[0]),
                                y1=float(coord[1]),
                                x2=float(coord[2]),
                                y2=float(coord[3]),
                                confidence=float(conf),
                                label=label
                            )
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []


# =============================================================================
# Movement Analysis
# =============================================================================

class MovementAnalyzer:
    """Analyzes bear movement relative to static objects."""
    
    def __init__(self):
        self.previous_bear_positions: List[BoundingBox] = []
        self.reference_static_objects: List[BoundingBox] = []
    
    # PUBLIC_INTERFACE
    def analyze_movement(self, detection: FrameDetection) -> Tuple[MovementState, float, Optional[float]]:
        """
        Analyze bear movement in the current frame.
        
        Args:
            detection: Current frame detection results
            
        Returns:
            Tuple of (movement_state, confidence, distance_moved)
        """
        if not detection.bears:
            return MovementState.UNKNOWN, 0.0, None
        
        # Update reference static objects if we have new ones
        if detection.static_objects:
            self.reference_static_objects = detection.static_objects
        
        # If no previous positions, can't determine movement
        if not self.previous_bear_positions:
            self.previous_bear_positions = detection.bears
            return MovementState.UNKNOWN, 0.5, None
        
        # Calculate movement
        movement_state, confidence, distance = self._calculate_movement(detection.bears)
        
        # Update previous positions
        self.previous_bear_positions = detection.bears
        
        return movement_state, confidence, distance
    
    def _calculate_movement(self, current_bears: List[BoundingBox]) -> Tuple[MovementState, float, Optional[float]]:
        """Calculate movement between current and previous bear positions."""
        if not self.previous_bear_positions:
            return MovementState.UNKNOWN, 0.0, None
        
        # Find the closest matching bear from previous frame
        min_distance = float('inf')
        best_match_distance = None
        
        for current_bear in current_bears:
            for prev_bear in self.previous_bear_positions:
                distance = current_bear.distance_to(prev_bear)
                if distance < min_distance:
                    min_distance = distance
                    best_match_distance = distance
        
        if best_match_distance is None:
            return MovementState.UNKNOWN, 0.0, None
        
        # Determine movement state based on distance threshold
        if best_match_distance > Config.MOVEMENT_THRESHOLD_PIXELS:
            confidence = min(1.0, best_match_distance / (Config.MOVEMENT_THRESHOLD_PIXELS * 2))
            return MovementState.MOVING, confidence, best_match_distance
        else:
            confidence = 1.0 - (best_match_distance / Config.MOVEMENT_THRESHOLD_PIXELS)
            return MovementState.RESTING, confidence, best_match_distance


# =============================================================================
# Video Processing
# =============================================================================

class VideoProcessor:
    """Handles video file processing and frame extraction."""
    
    def __init__(self):
        self.current_video_path: Optional[str] = None
    
    # PUBLIC_INTERFACE
    def process_video(self, video_path: str, detector: BearDetector, 
                     analyzer: MovementAnalyzer, db_manager: DatabaseManager) -> None:
        """
        Process video file frame by frame.
        
        Args:
            video_path: Path to video file
            detector: YOLO detector instance
            analyzer: Movement analyzer instance
            db_manager: Database manager instance
        """
        if cv2 is None:
            logger.error("OpenCV not available - video processing disabled")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = 0
        
        logger.info(f"Processing video: {video_path} (FPS: {fps})")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Reached end of video")
                    break
                
                frame_count += 1
                video_timestamp = frame_count / fps
                
                # Process frame
                self._process_frame(
                    frame, frame_count, video_timestamp, 
                    detector, analyzer, db_manager
                )
                
                # Rate limiting
                if Config.FRAME_PROCESSING_DELAY > 0:
                    time.sleep(Config.FRAME_PROCESSING_DELAY)
                
        except KeyboardInterrupt:
            logger.info("Video processing interrupted by user")
        except Exception as e:
            logger.error(f"Error processing video: {e}")
        finally:
            cap.release()
            logger.info(f"Processed {frame_count} frames")
    
    def _process_frame(self, frame, frame_number: int, video_timestamp: float,
                      detector: BearDetector, analyzer: MovementAnalyzer, 
                      db_manager: DatabaseManager) -> None:
        """Process a single video frame."""
        try:
            # Detect objects
            all_detections = detector.detect_objects(frame)
            
            # Separate bears from static objects
            bears = [d for d in all_detections if d.label.lower() in Config.BEAR_LABELS]
            static_objects = [d for d in all_detections if d.label.lower() in Config.STATIC_OBJECT_LABELS]
            
            # Create frame detection
            detection = FrameDetection(
                frame_timestamp=datetime.now(timezone.utc),
                bears=bears,
                static_objects=static_objects,
                frame_number=frame_number,
                video_timestamp_sec=video_timestamp
            )
            
            # Analyze movement
            movement_state, confidence, distance = analyzer.analyze_movement(detection)
            
            # Save to database
            db_manager.save_detection(detection, movement_state, confidence, distance)
            
            # Log progress
            if frame_number % 100 == 0:
                logger.info(f"Processed frame {frame_number} - Bears: {len(bears)}, "
                           f"Static: {len(static_objects)}, Movement: {movement_state.value}")
                
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")


# =============================================================================
# Main Application
# =============================================================================

class BearMovementDetector:
    """Main application class that orchestrates the detection pipeline."""
    
    def __init__(self):
        self.s3_downloader = S3Downloader()
        self.detector = BearDetector()
        self.analyzer = MovementAnalyzer()
        self.db_manager = DatabaseManager()
        self.video_processor = VideoProcessor()
        self.last_download_time = 0.0
    
    # PUBLIC_INTERFACE
    def run(self) -> None:
        """
        Main execution loop that continuously processes videos.
        Downloads video from S3, processes frames, and analyzes movement.
        """
        logger.info("Starting Bear Movement Detection System")
        logger.info(f"Configuration: S3={Config.S3_BUCKET}/{Config.S3_KEY}, "
                   f"Model={Config.YOLO_MODEL_PATH}, Threshold={Config.DETECTION_THRESHOLD}")
        
        with tempfile.TemporaryDirectory(prefix="bear_movement_") as temp_dir:
            video_path = os.path.join(temp_dir, "bears.mp4")
            
            while True:
                try:
                    # Check if we need to download new video
                    current_time = time.time()
                    if (current_time - self.last_download_time) >= Config.DOWNLOAD_INTERVAL_SEC:
                        if self._download_video(video_path):
                            self.last_download_time = current_time
                        else:
                            logger.warning("Video download failed, retrying in 60 seconds")
                            time.sleep(60)
                            continue
                    
                    # Process video if available
                    if os.path.exists(video_path):
                        self.video_processor.process_video(
                            video_path, self.detector, self.analyzer, self.db_manager
                        )
                    
                    # Wait before next iteration
                    time.sleep(30)  # Wait 30 seconds before checking for new video
                    
                except KeyboardInterrupt:
                    logger.info("Shutting down gracefully...")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in main loop: {e}")
                    time.sleep(60)  # Wait before retrying
    
    def _download_video(self, video_path: str) -> bool:
        """Download video from S3."""
        try:
            return self.s3_downloader.download_video(video_path)
        except Exception as e:
            logger.error(f"Video download failed: {e}")
            return False


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Main CLI entry point."""
    try:
        # Validate dependencies
        missing_deps = []
        if boto3 is None:
            missing_deps.append("boto3")
        if YOLO is None:
            missing_deps.append("ultralytics")
        if cv2 is None:
            missing_deps.append("opencv-python")
        if create_engine is None:
            missing_deps.append("sqlalchemy")
        
        if missing_deps:
            logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
            logger.warning("Some features may be disabled. Install with: pip install " + " ".join(missing_deps))
        
        # Start the detector
        detector = BearMovementDetector()
        detector.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
