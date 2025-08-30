from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from .base import PipelineContext, PipelineData, PipelineStage, StageResult


@dataclass
class VideoConfig:
    """
    Configuration for the VideoToFramesStage.

    Attributes:
        source_path: Path or URL to the input video file.
        read_with_opencv: If True, uses cv2.VideoCapture; otherwise expects upstream to fill frames.
    """
    source_path: Optional[str] = None
    read_with_opencv: bool = True


class VideoToFramesStage(PipelineStage):
    """
    Stage 1: Video stream input and decomposition into frames.

    This stage can either:
    - Read frames from a video file using OpenCV if configured
    - Or act as a pass-through when frames are provided externally by a generator

    For stepwise processing, this stage expects the current PipelineData.frame to already
    be provided. When read_with_opencv is True and frame is None, it will attempt to
    read the next frame from the configured source.
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self._cap = None
        self._frame_index = 0
        self._fps = 0.0

    def on_open(self, context: PipelineContext) -> None:
        if not self.config.read_with_opencv:
            return
        try:
            import cv2  # type: ignore
        except Exception:
            context.set("video_error", "cv2 not available")
            return

        if not self.config.source_path:
            context.set("video_error", "No source_path provided to VideoToFramesStage")
            return

        cap = cv2.VideoCapture(self.config.source_path)
        if not cap.isOpened():
            context.set("video_error", f"Failed to open video: {self.config.source_path}")
            return

        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._frame_index = 0

    def process(self, data: PipelineData, context: PipelineContext) -> StageResult:
        if data.frame is None and self.config.read_with_opencv and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                # End of video
                return StageResult(data=data, emit=[], continue_pipeline=False)
            data.frame = frame
            data.frame_index = self._frame_index
            data.frame_ts = (self._frame_index / self._fps) if self._fps > 0 else 0.0
            self._frame_index += 1

        # Pass-through when frame is already present
        return StageResult(data=data, emit=[])

    def on_close(self, context: PipelineContext) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class ObjectDetectionStage(PipelineStage):
    """
    Stage 2: Object detection in frames.

    Minimal implementation returns a placeholder detection if a frame exists.
    TODO:
      - Integrate a detection model (e.g., YOLO object detection or custom detector).
      - Support dynamic classes and thresholds from context.config.
    """

    def process(self, data: PipelineData, context: PipelineContext) -> StageResult:
        detections: List[Dict[str, Any]] = []

        # Placeholder: if frame exists, return a single dummy detection
        if data.frame is not None:
            detections.append(
                {
                    "label": "bear",
                    "probability": 0.9,
                    "bbox": (10.0, 10.0, 100.0, 100.0),
                }
            )
        data.detections = detections
        return StageResult(data=data, emit=[])


class PoseEstimationStage(PipelineStage):
    """
    Stage 3: Pose identification using YOLO (per object).

    This stage expects object detections in data.detections and a frame in data.frame.
    Minimal implementation assigns a dummy pose to each detection.

    TODO:
      - Load a YOLO/pose model in on_open() and store it in context.state["pose_model"].
      - Run the model per detection bbox and populate richer keypoint outputs.
      - Make threshold/label mappings configurable via context.config.
    """

    def on_open(self, context: PipelineContext) -> None:
        # Example of where you'd load a model:
        # if "pose_model" not in context.state:
        #     from ultralytics import YOLO
        #     model_path = context.get("pose_model_path", "yolov8n-pose.pt")
        #     context.set("pose_model", YOLO(model_path))
        pass

    def process(self, data: PipelineData, context: PipelineContext) -> StageResult:
        poses: List[Dict[str, Any]] = []
        for det in data.detections:
            # Placeholder pose information
            poses.append(
                {
                    "label": det.get("label", "unknown"),
                    "bbox": det.get("bbox"),
                    "pose": "standing",  # e.g., standing, sitting, walking
                    "score": 0.85,
                    # "keypoints": [...],  # TODO integrate real keypoints
                }
            )
        data.poses = poses
        return StageResult(data=data, emit=[])


class BehaviorAnalysisStage(PipelineStage):
    """
    Stage 4: Behavioral observation over a temporal window.

    Maintains a fixed-size deque per object label to infer behavior trends across time.
    Minimal implementation aggregates recent poses to produce a dominant behavior.

    TODO:
      - Track behavior per object ID rather than label once ID tracking is available.
      - Use time-weighted smoothing with timestamps.
      - Add motion features (optical flow, velocity) for better behavior inference.
    """

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self._history: Dict[str, Deque[str]] = {}

    def process(self, data: PipelineData, context: PipelineContext) -> StageResult:
        behaviors: List[Dict[str, Any]] = []

        for pose in data.poses:
            label = str(pose.get("label", "unknown"))
            pose_name = str(pose.get("pose", "unknown"))

            if label not in self._history:
                self._history[label] = deque(maxlen=self.window_size)
            self._history[label].append(pose_name)

            # Naive dominant behavior by frequency
            counts: Dict[str, int] = {}
            for p in self._history[label]:
                counts[p] = counts.get(p, 0) + 1
            dominant = max(counts, key=counts.get) if counts else "unknown"

            behaviors.append(
                {
                    "label": label,
                    "dominant_behavior": dominant,
                    "window_size": len(self._history[label]),
                }
            )
        data.behaviors = behaviors
        return StageResult(data=data, emit=[])


class EventDeterminationStage(PipelineStage):
    """
    Stage 5: Event determination combining pose, behavior, and context.

    Minimal implementation emits an "observation" event whenever a 'bear' is present.
    TODO:
      - Combine multiple context signals (location, time-of-day, weather).
      - Derive events like "bear_sitting_near_stream", "cub_with_mother", etc.
      - Implement rule-based and/or ML-based event fusion.
    """

    def process(self, data: PipelineData, context: PipelineContext) -> StageResult:
        events: List[Dict[str, Any]] = []
        emitted: List[Dict[str, Any]] = []

        for pose in data.poses:
            if pose.get("label") == "bear":
                event = {
                    "type": "observation",
                    "label": "bear",
                    "pose": pose.get("pose"),
                    "bbox": pose.get("bbox"),
                    "frame_index": data.frame_index,
                    "frame_ts": data.frame_ts,
                }
                events.append(event)
                # Emit so that upstream runner (e.g., identify_bears) can persist/log
                emitted.append(event)

        data.events = events
        return StageResult(data=data, emit=emitted)
