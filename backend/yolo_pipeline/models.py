"""YOLO model operations for the pipeline."""

import sys
from typing import Tuple, Optional, Any

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not available", file=sys.stderr)
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

from .config import DETECTION_WEIGHTS, CLASSIFICATION_WEIGHTS, POSE_WEIGHTS

def load_models() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load YOLOv8 detection, classification, and pose models."""
    if not ULTRALYTICS_AVAILABLE:
        print("[ERROR] ultralytics is required but not available.", file=sys.stderr)
        return None, None, None
        
    try:
        detector = YOLO(DETECTION_WEIGHTS)
        print(f"[INFO] Loaded detection model: {DETECTION_WEIGHTS}")
        classifier = YOLO(CLASSIFICATION_WEIGHTS)
        print(f"[INFO] Loaded classification model: {CLASSIFICATION_WEIGHTS}")
        poser = YOLO(POSE_WEIGHTS)
        print(f"[INFO] Loaded pose model: {POSE_WEIGHTS}")
        return detector, classifier, poser
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO models: {e}", file=sys.stderr)
        return None, None, None

def process_detection(detector_result, classifier, poser, frame):
    """Process a single detection result with classification and pose estimation."""
    if not detector_result:
        return []
    
    r0 = detector_result[0]
    boxes = getattr(r0, "boxes", None)
    names = getattr(r0, "names", {})
    if boxes is None:
        return []

    confs = getattr(boxes, "conf", None)
    clses = getattr(boxes, "cls", None)
    xyxy = getattr(boxes, "xyxy", None)
    if confs is None or clses is None or xyxy is None:
        return []

    # Convert to lists
    if hasattr(confs, 'ndim') and confs.ndim == 0:
        conf_list = [float(confs.item())]
    elif hasattr(confs, 'squeeze'):
        conf_list = confs.squeeze(-1).tolist()
        if not isinstance(conf_list, list):
            conf_list = [conf_list]
    else:
        conf_list = [float(confs)] if not isinstance(confs, list) else confs

    # Handle class IDs
    if hasattr(clses, 'ndim') and clses.ndim == 0:
        cls_list = [int(clses.item())]
    elif hasattr(clses, 'squeeze'):
        cls_list = clses.squeeze(-1).tolist()
        if not isinstance(cls_list, list):
            cls_list = [cls_list]
    else:
        cls_list = [int(clses)] if not isinstance(clses, list) else clses

    # Handle bounding boxes
    xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
    if len(xyxy_list) > 0 and not isinstance(xyxy_list[0], list):
        xyxy_list = [xyxy_list]

    results = []
    h, w = frame.shape[:2]

    for i, raw_c in enumerate(conf_list):
        try:
            conf = float(raw_c)
        except Exception:
            conf = 0.0

        # Get label
        cls_id = int(cls_list[i]) if i < len(cls_list) else -1
        label = names.get(cls_id, None)
        if not isinstance(label, str):
            continue

        # Filter for bears only (currently disabled)
        # norm_label = label.strip().lower()
        # if norm_label != "bear":
        #     continue

        # Get bounding box
        if i < len(xyxy_list):
            b = xyxy_list[i]
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            else:
                x1 = y1 = x2 = y2 = 0
        else:
            x1 = y1 = x2 = y2 = 0

        # Crop detection region
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(w, int(x2)), min(h, int(y2))
        crop = frame[yi1:yi2, xi1:xi2].copy() if xi2 > xi1 and yi2 > yi1 else None

        # Run classification
        cls_label, cls_conf = None, None
        if crop is not None and crop.size > 0:
            try:
                cls_res = classifier(crop, verbose=False)
                if cls_res and len(cls_res) > 0:
                    r = cls_res[0]
                    cls_names = getattr(r, "names", {})
                    probs = getattr(r, "probs", None)
                    if probs is not None:
                        top1 = getattr(probs, "top1", None)
                        top1conf = getattr(probs, "top1conf", None)
                        if top1 is not None:
                            cls_label = cls_names.get(int(top1), str(int(top1)))
                        if top1conf is not None:
                            cls_conf = float(top1conf.item()) if hasattr(top1conf, "item") else float(top1conf)
            except Exception as e:
                print(f"[WARN] Classification failed: {e}", file=sys.stderr)

        # Run pose estimation
        pose_status, num_keypoints = None, 0
        if crop is not None and crop.size > 0:
            try:
                pose_res = poser(crop, verbose=False)
                if pose_res and len(pose_res) > 0:
                    r = pose_res[0]
                    kpts = getattr(r, "keypoints", None)
                    if kpts is not None:
                        xy = getattr(kpts, "xy", None)
                        if xy is not None:
                            if hasattr(xy, '__len__') and len(xy) > 0:
                                arr = xy[0] if isinstance(xy, (list, tuple)) else xy
                                if hasattr(arr, "shape"):
                                    num_keypoints = int(arr.shape[0])
                                elif hasattr(arr, '__len__'):
                                    num_keypoints = len(arr)
                            pose_status = "OK"
            except Exception as e:
                print(f"[WARN] Pose estimation failed: {e}", file=sys.stderr)

        results.append({
            'label': label,
            'bbox': (x1, y1, x2, y2),
            'confidence': conf,
            'classification': (cls_label, cls_conf),
            'pose': (pose_status, num_keypoints)
        })

    return results