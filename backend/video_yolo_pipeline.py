#!/usr/bin/env python3
"""
video_yolo_pipeline.py

A simple, modular script that:
1. Loads an MP4 video from the same directory as this script.
2. Iterates through every frame of the video.
3. Uses YOLOv8m (detection) to find objects in each frame.
4. Filters detections labeled as 'bear', 'dog', or 'giraffe'.
5. Crops these regions from the original frame.
6. Runs YOLOv8-cls (classification) on each crop and prints the results.
7. Runs YOLOv8-pose (pose) on each crop and prints simple pose information.

Assumptions:
- ultralytics, opencv-python, numpy are installed (requirements already include them).
- YOLOv8m detection weights are accessible (e.g., 'yolov8m.pt').
- Classification and pose models can be instantiated via ultralytics.YOLO with appropriate weights
  such as 'yolov8m-cls.pt' and 'yolov8m-pose.pt'. If these aren't present locally, ultralytics will
  attempt to fetch them as per its standard behavior.

Note:
- The script automatically searches for the first .mp4 in the same directory if no specific
  filename is passed into run().
- This version processes all frames in the video.
"""

import os
import sys
from typing import List, Tuple, Optional, Dict, Any, Generator

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None
    print(f"[WARN] OpenCV import failed: {e}", file=sys.stderr)

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print(f"[WARN] ultralytics import failed: {e}", file=sys.stderr)


# -------------------------- Utility/Helper Functions --------------------------

# PUBLIC_INTERFACE
def find_local_mp4(directory: str) -> Optional[str]:
    """
    Find the first .mp4 file in the specified directory.

    Args:
        directory: Directory to scan for .mp4 files.

    Returns:
        Path to the first .mp4 file found or None if none found.
    """
    for name in os.listdir(directory):
        if name.lower().endswith(".mp4"):
            return os.path.join(directory, name)
    return None


def _load_video_capture(path: str):
    """Load a cv2.VideoCapture for the given file path."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required but not available.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


# PUBLIC_INTERFACE
def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    """
    Extract the first frame from a video.

    Args:
        video_path: Path to the local video file (.mp4).

    Returns:
        The first frame as a numpy array in BGR format, or None on failure.
    """
    try:
        cap = _load_video_capture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"[ERROR] Could not read first frame from {video_path}", file=sys.stderr)
            return None
        return frame
    except Exception as e:
        print(f"[ERROR] extract_first_frame failed: {e}", file=sys.stderr)
        return None


# PUBLIC_INTERFACE
def load_models(
    detect_weights: str = "yolov8m.pt",
    cls_weights: str = "yolov8m-cls.pt",
    pose_weights: str = "yolov8m-pose.pt",
) -> Tuple[Any, Any, Any]:
    """
    Load YOLOv8 models for detection, classification, and pose.

    Args:
        detect_weights: Weights for detection model (e.g., 'yolov8m.pt').
        cls_weights: Weights for classification model (e.g., 'yolov8m-cls.pt').
        pose_weights: Weights for pose model (e.g., 'yolov8m-pose.pt').

    Returns:
        (detector, classifier, poser)
    """
    if YOLO is None:
        raise RuntimeError("ultralytics is required but not available.")
    detector = YOLO(detect_weights)
    classifier = YOLO(cls_weights)
    poser = YOLO(pose_weights)
    return detector, classifier, poser


# PUBLIC_INTERFACE
def run_detection(detector: Any, frame_bgr: np.ndarray) -> Any:
    """
    Run object detection on a BGR frame.

    Args:
        detector: YOLO detection model.
        frame_bgr: Frame in BGR format.

    Returns:
        Ultralytics results object/list.
    """
    return detector(frame_bgr)


# PUBLIC_INTERFACE
def filter_detections(results: Any, labels_of_interest: List[str]) -> List[Dict[str, Any]]:
    """
    Filter detections to include only those with target labels.

    Args:
        results: Ultralytics detection results (list-like).
        labels_of_interest: List of label strings to keep (e.g., ['bear','dog','giraffe']).

    Returns:
        List of dicts with keys: label, conf, bbox(x1,y1,x2,y2), and image_index.
    """
    filtered: List[Dict[str, Any]] = []
    try:
        for img_idx, res in enumerate(results):
            names = getattr(res, "names", {})
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            confs = getattr(boxes, "conf", None)
            clses = getattr(boxes, "cls", None)
            xyxy = getattr(boxes, "xyxy", None)
            if confs is None or clses is None or xyxy is None:
                continue

            conf_list = confs.squeeze(-1).tolist() if hasattr(confs, "squeeze") else (confs.tolist() if hasattr(confs, "tolist") else list(confs))
            cls_list = clses.squeeze(-1).tolist() if hasattr(clses, "squeeze") else (clses.tolist() if hasattr(clses, "tolist") else list(clses))
            xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)

            for i, conf in enumerate(conf_list):
                try:
                    c = float(conf)
                except Exception:
                    c = 0.0
                cls_id = int(cls_list[i]) if i < len(cls_list) else -1
                label = names.get(cls_id, str(cls_id))
                if label not in labels_of_interest:
                    continue
                if i < len(xyxy_list):
                    b = xyxy_list[i]
                    if isinstance(b, (list, tuple)) and len(b) >= 4:
                        x1, y1, x2, y2 = map(float, b[:4])
                    else:
                        x1 = y1 = x2 = y2 = 0.0
                else:
                    x1 = y1 = x2 = y2 = 0.0
                filtered.append({
                    "image_index": img_idx,
                    "label": label,
                    "conf": c,
                    "bbox": (x1, y1, x2, y2),
                })
    except Exception as e:
        print(f"[ERROR] filter_detections failed: {e}", file=sys.stderr)
    return filtered


# PUBLIC_INTERFACE
def crop_detections(frame_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], np.ndarray]]:
    """
    Crop detection regions from the original frame.

    Args:
        frame_bgr: Original frame in BGR.
        detections: Filtered detection dicts.

    Returns:
        List of (det_dict, crop_bgr) tuples.
    """
    crops: List[Tuple[Dict[str, Any], np.ndarray]] = []
    h, w = frame_bgr.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        crop = frame_bgr[yi1:yi2, xi1:xi2].copy()
        crops.append((det, crop))
    return crops


# PUBLIC_INTERFACE
def classify_crops(classifier: Any, crops: List[Tuple[Dict[str, Any], np.ndarray]]) -> List[Dict[str, Any]]:
    """
    Classify each cropped image using YOLOv8-cls.

    Args:
        classifier: YOLO classification model.
        crops: List of (det, crop_bgr).

    Returns:
        List of classification results with detection context.
    """
    results: List[Dict[str, Any]] = []
    for det, crop in crops:
        try:
            cls_res = classifier(crop)
            # Extract top-1 class prediction if available
            top_label = None
            top_conf = None
            if cls_res and len(cls_res) > 0:
                r = cls_res[0]
                names = getattr(r, "names", {})
                probs = getattr(r, "probs", None)
                if probs is not None:
                    # probs has .top1 and .top1conf in recent ultralytics versions
                    top1 = getattr(probs, "top1", None)
                    top1conf = getattr(probs, "top1conf", None)
                    try:
                        if top1 is not None:
                            top_label = names.get(int(top1), str(int(top1)))
                        if top1conf is not None:
                            if hasattr(top1conf, "item"):
                                top_conf = float(top1conf.item())
                            else:
                                top_conf = float(top1conf)
                    except Exception:
                        pass
            results.append({
                "det": det,
                "cls_label": top_label,
                "cls_conf": top_conf
            })
        except Exception as e:
            print(f"[WARN] Classification failed for a crop: {e}", file=sys.stderr)
            results.append({
                "det": det,
                "cls_label": None,
                "cls_conf": None
            })
    return results


# PUBLIC_INTERFACE
def pose_on_crops(pose_model: Any, crops: List[Tuple[Dict[str, Any], np.ndarray]]) -> List[Dict[str, Any]]:
    """
    Run pose estimation on each cropped image using YOLOv8-pose.

    Args:
        pose_model: YOLO pose model.
        crops: List of (det, crop_bgr).

    Returns:
        List of pose results with detection context.
        Example fields: 'keypoints' (if available), number_of_keypoints, etc.
    """
    results: List[Dict[str, Any]] = []
    for det, crop in crops:
        try:
            pose_res = pose_model(crop)
            keypoints_info = None
            num_kpts = 0
            if pose_res and len(pose_res) > 0:
                r = pose_res[0]
                kpts = getattr(r, "keypoints", None)
                if kpts is not None:
                    # kpts.xy is often a list of arrays of shape [n_kpts, 2]
                    xy = getattr(kpts, "xy", None)
                    if xy is not None:
                        try:
                            arr = xy[0] if isinstance(xy, (list, tuple)) and len(xy) > 0 else xy
                            if hasattr(arr, "shape"):
                                num_kpts = int(arr.shape[0])
                            keypoints_info = "OK"
                        except Exception:
                            keypoints_info = "PARSE_ERROR"
            results.append({
                "det": det,
                "pose_keypoints_status": keypoints_info,
                "pose_num_keypoints": num_kpts,
            })
        except Exception as e:
            print(f"[WARN] Pose estimation failed for a crop: {e}", file=sys.stderr)
            results.append({
                "det": det,
                "pose_keypoints_status": None,
                "pose_num_keypoints": 0,
            })
    return results


def _print_summary_for_frame(
    frame_idx: int,
    dets: List[Dict[str, Any]],
    cls_out: List[Dict[str, Any]],
    pose_out: List[Dict[str, Any]],
) -> None:
    """Pretty-print a per-frame summary to stdout."""
    print(f"\n====== Frame {frame_idx} ======")
    if not dets:
        print("No target detections (bear/dog/giraffe).")
        return

    print("=== Detection Summary (bear/dog/giraffe) ===")
    for i, d in enumerate(dets):
        bbox = d["bbox"]
        print(f"- {i+1}. label={d['label']} conf={d['conf']:.3f} bbox=({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f})")

    print("\n=== Classification Results ===")
    for i, c in enumerate(cls_out):
        det = c["det"]
        print(f"- {i+1}. det_label={det['label']} cls_label={c['cls_label']} cls_conf={c['cls_conf']}")

    print("\n=== Pose Results ===")
    for i, p in enumerate(pose_out):
        det = p["det"]
        print(f"- {i+1}. det_label={det['label']} pose_keypoints_status={p['pose_keypoints_status']} num_keypoints={p['pose_num_keypoints']}")


def _frame_reader(video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield (frame_index, frame_bgr) for every frame in the video."""
    cap = _load_video_capture(video_path)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def _process_single_frame(
    frame_idx: int,
    frame_bgr: np.ndarray,
    detector: Any,
    classifier: Any,
    poser: Any,
    labels: List[str],
) -> None:
    """Run the full pipeline for a single frame and print results."""
    try:
        det_results = run_detection(detector, frame_bgr)
        dets = filter_detections(det_results, labels)
    except Exception as e:
        print(f"[ERROR] Detection failed on frame {frame_idx}: {e}", file=sys.stderr)
        return

    if not dets:
        _print_summary_for_frame(frame_idx, [], [], [])
        return

    crops = crop_detections(frame_bgr, dets)
    if not crops:
        _print_summary_for_frame(frame_idx, dets, [], [])
        return

    cls_results = classify_crops(classifier, crops)
    pose_results = pose_on_crops(poser, crops)
    _print_summary_for_frame(frame_idx, dets, cls_results, pose_results)


# PUBLIC_INTERFACE
def run(
    video_filename: Optional[str] = None,
    detection_weights: str = "yolov8m.pt",
    classification_weights: str = "yolov8m-cls.pt",
    pose_weights: str = "yolov8m-pose.pt",
    labels_to_keep: Optional[List[str]] = None,
) -> None:
    """
    Orchestrates the pipeline for a single local MP4 and processes all frames.

    Args:
        video_filename: Name of the MP4 file in the same directory. If None, auto-detects first .mp4.
        detection_weights: Path/name for YOLOv8 detection model.
        classification_weights: Path/name for YOLOv8 classification model.
        pose_weights: Path/name for YOLOv8 pose model.
        labels_to_keep: Labels to filter from detection. Defaults to ['bear','dog','giraffe'].
    """
    labels = labels_to_keep or ["bear", "dog", "giraffe"]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if video_filename is None:
        video_path = find_local_mp4(base_dir)
        if video_path is None:
            print("[ERROR] No .mp4 file found in this directory.", file=sys.stderr)
            return
    else:
        video_path = os.path.join(base_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"[ERROR] File not found: {video_path}", file=sys.stderr)
            return

    print(f"[INFO] Using video: {video_path}")

    # Load models once for efficiency
    try:
        detector, classifier, poser = load_models(
            detect_weights=detection_weights,
            cls_weights=classification_weights,
            pose_weights=pose_weights,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO models: {e}", file=sys.stderr)
        return

    # Iterate over every frame and process
    for frame_idx, frame in _frame_reader(video_path):
        _process_single_frame(frame_idx, frame, detector, classifier, poser, labels)


if __name__ == "__main__":
    # Run with defaults: auto-pick first .mp4 in this script's directory.
    run()
