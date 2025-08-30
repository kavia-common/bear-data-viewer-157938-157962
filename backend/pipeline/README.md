# Modular Video Processing Pipeline (backend/pipeline)

This package provides a minimal, extensible framework for a multi-stage video analysis pipeline:

Stages:
1) VideoToFramesStage: Reads/provides frames
2) ObjectDetectionStage: Detects objects in a frame
3) PoseEstimationStage: Estimates pose per detection (YOLO pose TODO)
4) BehaviorAnalysisStage: Aggregates poses across time
5) EventDeterminationStage: Produces high-level events

Usage from identify_bears.py:
- Use run_pipeline_over_video(video_path) to process a local video stepwise.
- Or build a pipeline with build_default_pipeline() and feed frames via run_on_frames().

Extension points:
- Replace or add stages by creating new classes extending PipelineStage.
- Use PipelineContext.config/state to share models and settings across stages.

TODOs:
- Integrate actual object and pose detection models (e.g., ultralytics YOLO).
- Enhance behavior analysis and event fusion with robust features.
