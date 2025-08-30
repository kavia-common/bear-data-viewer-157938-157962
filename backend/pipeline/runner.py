from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

from .base import Pipeline, PipelineContext, PipelineData
from .stages import (
    VideoToFramesStage,
    ObjectDetectionStage,
    PoseEstimationStage,
    BehaviorAnalysisStage,
    EventDeterminationStage,
    VideoConfig,
)

logger = logging.getLogger(__name__)


# PUBLIC_INTERFACE
def build_default_pipeline(
    video_source_path: Optional[str] = None,
    read_with_opencv: bool = False,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, PipelineContext]:
    """
    Build a default pipeline composed of the five requested stages.

    Args:
        video_source_path: Optional path to a video file; used if read_with_opencv=True.
        read_with_opencv: If True, VideoToFramesStage will read frames itself.
        extra_config: Additional configuration to include in the PipelineContext.

    Returns:
        A tuple of (pipeline, context).
    """
    context = PipelineContext(config=extra_config or {})
    stages = [
        VideoToFramesStage(config=VideoConfig(source_path=video_source_path, read_with_opencv=read_with_opencv)),
        ObjectDetectionStage(),
        PoseEstimationStage(),
        BehaviorAnalysisStage(),
        EventDeterminationStage(),
    ]
    pipe = Pipeline(stages=stages, context=context)
    return pipe, context


# PUBLIC_INTERFACE
def run_on_frames(
    frames: Iterable[Tuple[Any, Optional[int], Optional[float]]],
    pipeline: Pipeline,
) -> Iterable[Dict[str, Any]]:
    """
    Run the pipeline stepwise on an iterable of frames.

    Args:
        frames: Iterable yielding (frame, frame_index, frame_ts) for each step.
        pipeline: A Pipeline instance previously built via build_default_pipeline or custom.

    Yields:
        Emitted items from stages (e.g., events) as dictionaries.
    """
    try:
        pipeline.open()
        for frame, frame_index, frame_ts in frames:
            step = PipelineData(frame=frame, frame_index=frame_index, frame_ts=frame_ts)
            _, emitted = pipeline.run_step(step)
            for item in emitted:
                yield item
    finally:
        pipeline.close()
