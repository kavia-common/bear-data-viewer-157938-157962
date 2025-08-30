"""
Pipeline package initializer for modular video-processing stages.

This package provides:
- Base interfaces for stages and pipeline context
- Stage implementations for:
  1) Video stream input & frame decomposition
  2) Object detection
  3) Pose identification (YOLO placeholder)
  4) Behavioral observation over a temporal window
  5) Event determination combining pose, behavior, and context
- A simple Pipeline class to compose stages and pass data stepwise

Stages are intentionally minimal and include TODOs for real model integrations.
"""
from .base import (
    PipelineStage,
    PipelineContext,
    PipelineData,
    StageResult,
    Pipeline,
)
from .stages import (
    VideoToFramesStage,
    ObjectDetectionStage,
    PoseEstimationStage,
    BehaviorAnalysisStage,
    EventDeterminationStage,
)

__all__ = [
    "PipelineStage",
    "PipelineContext",
    "PipelineData",
    "StageResult",
    "Pipeline",
    "VideoToFramesStage",
    "ObjectDetectionStage",
    "PoseEstimationStage",
    "BehaviorAnalysisStage",
    "EventDeterminationStage",
]
