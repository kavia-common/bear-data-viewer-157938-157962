from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class PipelineContext:
    """
    Shared, mutable context dictionary for pipeline execution.

    Use this to store configuration, model handles, and runtime state that should
    be accessible to multiple stages (e.g., YOLO model instance, thresholds, etc.).
    """
    config: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    # PUBLIC_INTERFACE
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from combined config/state with state taking precedence."""
        return self.state.get(key, self.config.get(key, default))

    # PUBLIC_INTERFACE
    def set(self, key: str, value: Any, in_state: bool = True) -> None:
        """Set a value either in state (default) or config."""
        if in_state:
            self.state[key] = value
        else:
            self.config[key] = value


@dataclass
class PipelineData:
    """
    Standard container for data flowing between stages.

    Attributes:
        frame: The current video frame (e.g., numpy array) or None.
        frame_index: Index of the current frame in the stream.
        frame_ts: Timestamp (seconds) of the current frame in the stream.
        detections: List of object detections for the frame.
        poses: Pose estimation results at the per-object level.
        behaviors: Behavioral observations extracted over a temporal window.
        events: High-level events determined from pose, behavior, and context.
        meta: Additional custom metadata for extensibility.
    """
    frame: Any = None
    frame_index: Optional[int] = None
    frame_ts: Optional[float] = None
    detections: List[Dict[str, Any]] = field(default_factory=list)
    poses: List[Dict[str, Any]] = field(default_factory=list)
    behaviors: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """
    Result wrapper returned by each stage.

    Attributes:
        data: The (possibly updated) PipelineData to pass to the next stage.
        emit: Optional output items the stage wants to emit downstream immediately
              (e.g., detections to persist). The pipeline runner may consume these.
        continue_pipeline: Whether the pipeline should continue to subsequent stages.
    """
    data: PipelineData
    emit: List[Dict[str, Any]] = field(default_factory=list)
    continue_pipeline: bool = True


class PipelineStage(abc.ABC):
    """
    Abstract base class for a pipeline stage.

    Stages should implement process() and can optionally override on_open() and on_close()
    for resource initialization and cleanup.
    """

    # PUBLIC_INTERFACE
    def on_open(self, context: PipelineContext) -> None:
        """
        Hook called before the first process() of this stage.

        Use this to load models, allocate buffers, or initialize state.
        """
        return None

    # PUBLIC_INTERFACE
    @abc.abstractmethod
    def process(self, data: PipelineData, context: PipelineContext) -> StageResult:
        """
        Process input PipelineData and return a StageResult.

        Args:
            data: Current step's PipelineData.
            context: Shared PipelineContext for configuration and state.

        Returns:
            StageResult with updated PipelineData and optional emitted items.
        """
        raise NotImplementedError

    # PUBLIC_INTERFACE
    def on_close(self, context: PipelineContext) -> None:
        """
        Hook called after the last process() of this stage.

        Use this to release resources and save trailing state if needed.
        """
        return None


class Pipeline:
    """
    Orchestrates a sequence of stages with explicit data passing between them.

    The pipeline supports stepwise processing of frames or other units of work
    by calling run_step() with the current PipelineData. Each stage can modify
    the data and optionally emit outputs. Stages can also be replaced or composed.
    """

    def __init__(self, stages: Sequence[PipelineStage], context: Optional[PipelineContext] = None):
        self._stages: List[PipelineStage] = list(stages)
        self.context: PipelineContext = context or PipelineContext()
        self._opened: bool = False

    # PUBLIC_INTERFACE
    def open(self) -> None:
        """
        Initialize all stages via on_open().

        Call this before the first run_step() and once per pipeline lifecycle.
        """
        if self._opened:
            return
        for stage in self._stages:
            stage.on_open(self.context)
        self._opened = True

    # PUBLIC_INTERFACE
    def close(self) -> None:
        """
        Finalize all stages via on_close().

        Call this after the last run_step() of a lifecycle.
        """
        for stage in self._stages:
            try:
                stage.on_close(self.context)
            except Exception:
                # Swallow close errors to ensure all stages have the chance to close.
                pass
        self._opened = False

    # PUBLIC_INTERFACE
    def run_step(self, data: PipelineData) -> Tuple[PipelineData, List[Dict[str, Any]]]:
        """
        Run a single step through all stages with explicit data passing.

        Args:
            data: The PipelineData for the current step (e.g., a frame and metadata).

        Returns:
            A tuple of (PipelineData, emitted_items) where emitted_items is a
            flattened list of outputs emitted by stages for this step.
        """
        if not self._opened:
            self.open()

        emitted: List[Dict[str, Any]] = []
        current = data
        for stage in self._stages:
            result = stage.process(current, self.context)
            emitted.extend(result.emit)
            current = result.data
            if not result.continue_pipeline:
                break
        return current, emitted
