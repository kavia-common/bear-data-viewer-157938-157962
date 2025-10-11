#!/usr/bin/env python3
"""
Bear Activity Classifier - Single-file module.

This module provides:
1) DeepLabCut (DLC) inference for pose estimation on videos or image folders.
2) Feature extraction from DLC CSV outputs (velocities, spans, spine angle).
3) Activity classification into sleeping/standing/moving via:
   - Heuristic fallback (no scikit-learn required), or
   - ML model (RandomForest) with optional training.
4) CLI utilities:
   - estimate: run pose estimation (real DLC or mock).
   - classify: end-to-end pose -> features -> predictions.
   - train: train a classifier from labeled CSVs.

The code includes dependency checks and mock/stub fallbacks for DLC and sklearn.
If DLC is not installed and use_mock=False, a clear error is raised; use --mock to
simulate DLC output for development/testing.

Usage examples:
- Run pose estimation (DLC installed):
    python bear_activity_classifier.py estimate --input /path/to/video.mp4 --out out_dir

- Run pose estimation in mock mode:
    python bear_activity_classifier.py estimate --input /path/to/video.mp4 --out out_dir --mock

- Classify with heuristic (no trained model):
    python bear_activity_classifier.py classify --input /path/to/video.mp4 --out out_dir --mock

- Classify with trained model:
    python bear_activity_classifier.py classify --input /path/to/video.mp4 --out out_dir --model out_dir/model.joblib

- Train a model from DLC CSV(s) and label CSV(s):
    python bear_activity_classifier.py train --dlc_csvs dlc1.csv dlc2.csv --label_csvs lab1.csv lab2.csv --out_model out_dir/model.joblib

Notes:
- DLC CSV format: MultiIndex columns [scorer, bodypart, coords], coords in [x, y, likelihood].
- Label CSV format (for training): columns = ["frame","label"] where label in ["sleeping","standing","moving"].

Environment & Dependencies:
- Optional DLC integration tries to import 'deeplabcut' as dlc.
- Uses numpy, pandas, opencv-python (cv2), pathlib, joblib (for model I/O), scikit-learn (optional).
- If scikit-learn is unavailable, ML mode is disabled; fallback heuristic is used.

"""

from __future__ import annotations

import argparse
import sys
import warnings
import json
from pathlib import Path
from typing import List, Optional, Tuple

# Minimal dependencies (numpy, pandas) are required.
import numpy as np
import pandas as pd

# Optional dependencies
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    warnings.warn("OpenCV (cv2) not found. Video frame count in mock may be limited. Install opencv-python for full support.")

try:
    import deeplabcut as dlc  # type: ignore
    _DLC_AVAILABLE = True
except Exception:
    dlc = None
    _DLC_AVAILABLE = False
    warnings.warn("DeepLabCut not available. Use --mock for simulated pose estimation output.")

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.model_selection import train_test_split  # noqa: F401
    _SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    _SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Falling back to heuristic classifier.")

try:
    import joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:
    joblib = None
    _JOBLIB_AVAILABLE = False
    warnings.warn("joblib not available. Model load/save will be disabled.")

# ==================================
# Configuration
# ==================================

# Optional paths to DLC project/model; not required in mock mode
DLC_PROJECT_PATH: Optional[str] = None  # e.g., "/path/to/DLC/project/config.yaml"
DLC_MODEL_PATH: Optional[str] = None    # Optional: path to specific DLC model

# Default body parts commonly used (variable; can be inferred from CSV header)
BODY_PARTS: List[str] = [
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "spine", "left_hip", "right_hip"
]

# Default FPS (frames per second) used for velocity computation
FPS: int = 30

# Activity labels
ACTIVITY_LABELS: List[str] = ["sleeping", "standing", "moving"]

# ==================================
# Utilities
# ==================================

def _ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _is_video_file(p: Path) -> bool:
    """Quick heuristic to determine if a path points to a video file."""
    return p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def _list_images_in_dir(p: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


def _read_video_frame_count(video_path: Path) -> int:
    """Read video frame count using cv2 if available; otherwise returns 300 by default."""
    if cv2 is None:
        return 300
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 300
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        frame_count = 300
    return frame_count


def _generate_mock_dlc_dataframe(
    n_frames: int,
    body_parts: List[str],
    image_size: Tuple[int, int] = (1280, 720),
    seed: int = 42
) -> pd.DataFrame:
    """Generate a synthetic DLC-like DataFrame with MultiIndex columns."""
    rng = np.random.default_rng(seed)

    # Simulate a bear path: mixture of low/high movement segments
    # Start near center
    x = np.cumsum(rng.normal(loc=0.0, scale=2.0, size=n_frames)) + image_size[0] / 2
    y = np.cumsum(rng.normal(loc=0.0, scale=1.5, size=n_frames)) + image_size[1] / 2

    # Random velocity bursts
    for i in range(0, n_frames, 150):
        burst_len = min(rng.integers(20, 60), n_frames - i)
        x[i:i+burst_len] += np.cumsum(rng.normal(0, 5.0, size=burst_len))
        y[i:i+burst_len] += np.cumsum(rng.normal(0, 4.0, size=burst_len))

    # Create columns multiindex: (scorer, bodypart, coord)
    scorer = "mockDLC"
    arrays = []
    for bp in body_parts:
        arrays.extend([(scorer, bp, "x"), (scorer, bp, "y"), (scorer, bp, "likelihood")])

    tuples = arrays
    index = pd.MultiIndex.from_tuples(tuples, names=["scorer", "bodypart", "coords"])

    data = np.zeros((n_frames, len(body_parts) * 3), dtype=float)
    for i, bp in enumerate(body_parts):
        # Offset per body part to simulate realistic spread
        dx = rng.normal(0, 20.0)
        dy = rng.normal(0, 15.0)
        noise_x = rng.normal(0, 3.0, size=n_frames)
        noise_y = rng.normal(0, 3.0, size=n_frames)
        base_x = x + dx + noise_x
        base_y = y + dy + noise_y

        # Likelihood fluctuates; most frames confident
        like = np.clip(rng.normal(0.85, 0.1, size=n_frames), 0.0, 1.0)
        # Add occasional low-likelihood segments
        for j in range(0, n_frames, 200):
            drop_len = min(rng.integers(5, 15), n_frames - j)
            like[j:j+drop_len] *= rng.uniform(0.1, 0.4)

        col_start = i * 3
        data[:, col_start + 0] = base_x
        data[:, col_start + 1] = base_y
        data[:, col_start + 2] = like

    df = pd.DataFrame(data, columns=index)
    df.index.name = "frame"
    return df


def _save_dlc_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DLC-like DataFrame to CSV with header structure preserved."""
    df.to_csv(path, index=True)


def _infer_body_parts_from_csv(df: pd.DataFrame) -> List[str]:
    """Infer body parts from DLC CSV multiindex columns."""
    if isinstance(df.columns, pd.MultiIndex) and "bodypart" in df.columns.names:
        return sorted(set(df.columns.get_level_values("bodypart")))
    # Fallback: attempt to parse known pattern
    parts = []
    for col in df.columns:
        if isinstance(col, tuple) and len(col) >= 2:
            parts.append(col[1])
        else:
            # flat columns: try pattern like "nose_x"
            if isinstance(col, str) and "_" in col:
                parts.append(col.split("_")[0])
    return sorted(set(parts) or BODY_PARTS)


# ==================================
# Pose Estimation
# ==================================

# PUBLIC_INTERFACE
def run_pose_estimation(
    input_path: str,
    output_dir: str,
    use_mock: bool = False,
    shuffle: int = 1,
    trainingsetindex: int = 0
) -> str:
    """Run DeepLabCut pose estimation or mock generation.

    Parameters
    ----------
    input_path : str
        Path to a video file or a directory of images.
    output_dir : str
        Directory to store DLC output CSV file(s).
    use_mock : bool, optional
        If True, generate a synthetic DLC-like CSV output for demonstration/testing.
    shuffle : int, optional
        DLC shuffle parameter.
    trainingsetindex : int, optional
        DLC trainingsetindex parameter.

    Returns
    -------
    str
        Path to the resulting DLC CSV file.

    Raises
    ------
    RuntimeError
        If DLC is not available and use_mock is False.
    FileNotFoundError
        If input_path does not exist.
    """
    input_p = Path(input_path)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    if not input_p.exists():
        raise FileNotFoundError(f"Input path not found: {input_p}")

    # Determine mode: video or images
    is_video = _is_video_file(input_p)
    is_dir = input_p.is_dir()

    if not use_mock and not _DLC_AVAILABLE:
        raise RuntimeError(
            "DeepLabCut is not available. Install 'deeplabcut' or use --mock to simulate outputs."
        )

    # DLC real inference
    if not use_mock and _DLC_AVAILABLE:
        # Validate DLC project/model if provided
        if DLC_PROJECT_PATH is None:
            warnings.warn(
                "DLC_PROJECT_PATH is not set. Using DLC defaults; ensure your environment is configured."
            )
        config_path = DLC_PROJECT_PATH

        # Prepare list of items for DLC
        if is_video:
            videos = [str(input_p)]
            print(f"[DLC] Analyzing video: {videos[0]}")
            dlc.analyze_videos(
                config_path,
                videos,
                videotype=input_p.suffix,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                save_as_csv=True
            )
            # DLC typically writes alongside the video; locate CSV
            # We'll try to find the most recent CSV in same dir
            candidate_dir = input_p.parent
            csvs = sorted(candidate_dir.glob("*filtered*.csv")) + sorted(candidate_dir.glob("*csv"))
            if not csvs:
                raise RuntimeError("No DLC CSV output found after analyze_videos.")
            # Copy the latest CSV to output_dir
            csv_path = csvs[-1]
            final_csv = out_dir / f"{input_p.stem}_DLC_output.csv"
            pd.read_csv(csv_path, header=[0, 1, 2], index_col=0).to_csv(final_csv)
            print(f"[DLC] Saved CSV to {final_csv}")
            return str(final_csv)

        elif is_dir:
            images = _list_images_in_dir(input_p)
            if not images:
                raise RuntimeError(f"No images found in directory: {input_p}")
            print(f"[DLC] Analyzing time-lapse images in: {input_p}")
            dlc.analyze_time_lapse_images(
                config_path,
                str(input_p),
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                save_as_csv=True
            )
            csvs = sorted(input_p.glob("*filtered*.csv")) + sorted(input_p.glob("*csv"))
            if not csvs:
                raise RuntimeError("No DLC CSV output found after analyze_time_lapse_images.")
            csv_path = csvs[-1]
            final_csv = out_dir / f"{input_p.name}_DLC_output.csv"
            pd.read_csv(csv_path, header=[0, 1, 2], index_col=0).to_csv(final_csv)
            print(f"[DLC] Saved CSV to {final_csv}")
            return str(final_csv)

        else:
            raise RuntimeError("Input path must be a video file or a directory of images.")

    # Mock mode
    # Decide frame count
    if is_video:
        n_frames = _read_video_frame_count(input_p)
    elif is_dir:
        n_frames = len(_list_images_in_dir(input_p))
        if n_frames == 0:
            # Try a default if no image; still allow demonstration
            n_frames = 300
    else:
        raise RuntimeError("Input path must be a video file or a directory of images.")

    print(f"[MOCK] Generating synthetic DLC CSV for {n_frames} frames...")
    df = _generate_mock_dlc_dataframe(n_frames, BODY_PARTS)
    final_csv = out_dir / (f"{input_p.stem}_mock_DLC.csv" if is_video else f"{input_p.name}_mock_DLC.csv")
    _save_dlc_csv(df, final_csv)
    print(f"[MOCK] Saved synthetic CSV to {final_csv}")
    return str(final_csv)


# ==================================
# Feature Extraction
# ==================================

def _rolling_median(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).median()


def _compute_spine_angle(
    df: pd.DataFrame,
    parts: List[str],
    likelihood_threshold: float
) -> pd.Series:
    """Compute spine angle (degrees) using vector between shoulders and hips."""
    # Attempt to use left/right shoulders and hips; fallback to any shoulder/hip-like parts
    def _first_available(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in parts:
                return c
        return None

    ls = _first_available(["left_shoulder", "l_shoulder", "LShoulder", "ShoulderL"])
    rs = _first_available(["right_shoulder", "r_shoulder", "RShoulder", "ShoulderR"])
    lh = _first_available(["left_hip", "l_hip", "LHip", "HipL"])
    rh = _first_available(["right_hip", "r_hip", "RHip", "HipR"])

    def _get_xy_like(bp: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        # MultiIndex access: (:, bp, x/y/likelihood)
        if isinstance(df.columns, pd.MultiIndex):
            x = df.xs((bp, "x"), level=("bodypart", "coords"), axis=1, drop_level=False).iloc[:, 0]
            y = df.xs((bp, "y"), level=("bodypart", "coords"), axis=1, drop_level=False).iloc[:, 0]
            l = df.xs((bp, "likelihood"), level=("bodypart", "coords"), axis=1, drop_level=False).iloc[:, 0]
            return x, y, l
        else:
            # flat fallback: bp_x, bp_y, bp_likelihood
            x = df.get(f"{bp}_x", pd.Series(np.nan, index=df.index))
            y = df.get(f"{bp}_y", pd.Series(np.nan, index=df.index))
            l = df.get(f"{bp}_likelihood", pd.Series(np.nan, index=df.index))
            return x, y, l

    def _mean_point(bps: List[str]) -> Tuple[pd.Series, pd.Series, pd.Series]:
        xs, ys, ls = [], [], []
        for bp in bps:
            if bp is None:
                continue
            x, y, l = _get_xy_like(bp)
            xs.append(x)
            ys.append(y)
            ls.append(l)
        if not xs:
            nan_series = pd.Series(np.nan, index=df.index)
            return nan_series, nan_series, nan_series
        xs = pd.concat(xs, axis=1)
        ys = pd.concat(ys, axis=1)
        ls = pd.concat(ls, axis=1)
        mask = ls >= likelihood_threshold
        # Avoid all-false rows
        xs_masked = xs.where(mask)
        ys_masked = ys.where(mask)
        x_mean = xs_masked.mean(axis=1)
        y_mean = ys_masked.mean(axis=1)
        l_mean = ls.mean(axis=1)
        return x_mean, y_mean, l_mean

    shoulder_x, shoulder_y, _ = _mean_point([ls, rs])
    hip_x, hip_y, _ = _mean_point([lh, rh])

    # Vector shoulder->hip
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    # Angle relative to vertical: compute angle of vector with respect to y-axis
    # angle = arctan2(dx, dy) in degrees; near 0 means vertical, near 90 means horizontal
    angle_rad = np.arctan2(dx, dy)
    angle_deg = np.degrees(angle_rad)
    return angle_deg.abs()  # absolute distance from vertical direction


# PUBLIC_INTERFACE
def extract_features_from_dlc_csv(
    csv_path: str,
    fps: int = FPS,
    likelihood_threshold: float = 0.6
) -> pd.DataFrame:
    """Extract per-frame features from a DLC CSV file.

    Features include:
    - mean_likelihood: average confidence across parts.
    - com_x, com_y: center of mass of confident keypoints.
    - com_velocity: magnitude of COM velocity (pixels/sec).
    - vertical_span, horizontal_span: bbox span across confident keypoints.
    - spine_angle_from_vertical: abs angle (deg) between spine axis and vertical.

    Parameters
    ----------
    csv_path : str
        Path to DLC CSV file (multiindex columns recommended).
    fps : int, optional
        Frames per second used to compute velocities.
    likelihood_threshold : float, optional
        Minimum likelihood for a keypoint to be considered in aggregations.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by frame with computed features.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    # If not multiindex header, try fallback
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 3:
        # Fallback to single-level columns
        df = pd.read_csv(csv_path, index_col=0)

    parts = _infer_body_parts_from_csv(df)

    # Build arrays of x,y,likelihood for all parts
    xs, ys, ls = [], [], []
    for bp in parts:
        if isinstance(df.columns, pd.MultiIndex):
            x = df.xs((bp, "x"), level=("bodypart", "coords"), axis=1, drop_level=False)
            y = df.xs((bp, "y"), level=("bodypart", "coords"), axis=1, drop_level=False)
            l = df.xs((bp, "likelihood"), level=("bodypart", "coords"), axis=1, drop_level=False)
            xs.append(x.iloc[:, 0])
            ys.append(y.iloc[:, 0])
            ls.append(l.iloc[:, 0])
        else:
            xs.append(df.get(f"{bp}_x", pd.Series(np.nan, index=df.index)))
            ys.append(df.get(f"{bp}_y", pd.Series(np.nan, index=df.index)))
            ls.append(df.get(f"{bp}_likelihood", pd.Series(np.nan, index=df.index)))

    X = pd.concat(xs, axis=1) if xs else pd.DataFrame(index=df.index)
    Y = pd.concat(ys, axis=1) if ys else pd.DataFrame(index=df.index)
    L = pd.concat(ls, axis=1) if ls else pd.DataFrame(index=df.index)
    X.columns = parts[:X.shape[1]]
    Y.columns = parts[:Y.shape[1]]
    L.columns = parts[:L.shape[1]]

    # Mean likelihood
    mean_likelihood = L.mean(axis=1).fillna(0.0)

    # Confident masks
    conf_mask = L >= likelihood_threshold

    # Center of mass over confident points
    Xc = X.where(conf_mask)
    Yc = Y.where(conf_mask)
    com_x = Xc.mean(axis=1)
    com_y = Yc.mean(axis=1)

    # Velocity of COM
    dcom_x = com_x.diff().fillna(0.0)
    dcom_y = com_y.diff().fillna(0.0)
    com_velocity = np.sqrt(dcom_x.pow(2) + dcom_y.pow(2)) * float(fps)

    # Posture spans (bbox across confident keypoints)
    x_min = Xc.min(axis=1)
    x_max = Xc.max(axis=1)
    y_min = Yc.min(axis=1)
    y_max = Yc.max(axis=1)
    horizontal_span = (x_max - x_min).fillna(0.0)
    vertical_span = (y_max - y_min).fillna(0.0)

    # Spine angle from vertical
    spine_angle_from_vertical = _compute_spine_angle(df, parts, likelihood_threshold)

    # Smooth selected signals
    com_x_s = _rolling_median(com_x, 5)
    com_y_s = _rolling_median(com_y, 5)
    com_velocity_s = _rolling_median(com_velocity, 5)
    vertical_span_s = _rolling_median(vertical_span, 5)
    horizontal_span_s = _rolling_median(horizontal_span, 5)
    spine_angle_s = _rolling_median(spine_angle_from_vertical, 5)
    mean_like_s = _rolling_median(mean_likelihood, 5)

    features = pd.DataFrame({
        "mean_likelihood": mean_like_s,
        "com_x": com_x_s,
        "com_y": com_y_s,
        "com_velocity": com_velocity_s,
        "vertical_span": vertical_span_s,
        "horizontal_span": horizontal_span_s,
        "spine_angle_from_vertical": spine_angle_s
    })
    features.index.name = "frame"
    # Fill remaining NaNs with nearest reasonable values
    features = features.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return features


# ==================================
# Classification
# ==================================

def _adaptive_thresholds(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Compute adaptive thresholds for velocity and spans using quantiles."""
    # Use quantiles robust to outliers
    v = df["com_velocity"].clip(lower=0)
    vert = df["vertical_span"].clip(lower=0)
    # Low and moving thresholds for velocity
    low_vel = float(np.quantile(v, 0.2))  # under this ~ sleeping/standing
    move_vel = float(np.quantile(v, 0.7))  # above this ~ moving
    # Vertical span boundary between sleeping and standing
    vert_split = float(np.quantile(vert, 0.5))
    # Ensure ordering
    if move_vel <= low_vel:
        move_vel = low_vel * 1.5 + 1.0
    return low_vel, move_vel, vert_split


# PUBLIC_INTERFACE
def classify_features_heuristic(df_features: pd.DataFrame) -> List[str]:
    """Heuristic classifier mapping features to activity labels.

    Rules:
    - moving: com_velocity >= moving_threshold
    - sleeping: mean_likelihood>=0.5, com_velocity < low_threshold, vertical_span small
    - standing: otherwise low velocity, vertical_span larger and spine near vertical

    Thresholds are computed adaptively using quantiles of the input features.
    """
    if df_features.empty:
        return []

    low_vel, move_vel, vert_split = _adaptive_thresholds(df_features)

    labels = []
    for _, r in df_features.iterrows():
        v = float(r["com_velocity"])
        vert = float(r["vertical_span"])
        like = float(r["mean_likelihood"])
        spine = float(r["spine_angle_from_vertical"])

        if v >= move_vel:
            labels.append("moving")
            continue

        # Low velocity region
        if like >= 0.5 and v < low_vel:
            # spine near vertical (small angle) suggests standing; very small vertical span suggests sleeping
            if vert <= 0.5 * vert_split:
                labels.append("sleeping")
            else:
                # Spine near vertical (<= 25 deg) -> standing; else sleeping/standing tie-breaker by span
                if spine <= 25.0:
                    labels.append("standing")
                else:
                    labels.append("sleeping")
        else:
            # Intermediate region: prefer standing if spine closer to vertical and span is moderate
            if spine <= 25.0 and vert >= 0.5 * vert_split:
                labels.append("standing")
            else:
                # Default to moving if velocity moderate
                labels.append("moving" if v >= 0.5 * move_vel else "standing")
    return labels


# PUBLIC_INTERFACE
def train_classifier(
    features_df: pd.DataFrame,
    labels_series: pd.Series,
    random_state: int = 42
):
    """Train a RandomForest classifier on features.

    Returns a trained model. Requires scikit-learn to be installed.
    """
    if not _SKLEARN_AVAILABLE or RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is not available. Cannot train ML classifier.")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced"
    )
    # Ensure alignment
    X = features_df[
        ["mean_likelihood", "com_velocity", "vertical_span", "horizontal_span", "spine_angle_from_vertical"]
    ].copy()
    y = labels_series.astype(str)
    model.fit(X, y)
    return model


# PUBLIC_INTERFACE
def predict_classifier(model, features_df: pd.DataFrame) -> List[str]:
    """Predict labels with a trained classifier."""
    X = features_df[
        ["mean_likelihood", "com_velocity", "vertical_span", "horizontal_span", "spine_angle_from_vertical"]
    ].copy()
    preds = model.predict(X)
    return list(map(str, preds))


# ==================================
# End-to-end pipeline
# ==================================

# PUBLIC_INTERFACE
def classify_bear_activity(
    input_path: str,
    output_dir: str,
    use_mock: bool = False,
    model_path: Optional[str] = None,
    fps: int = FPS
) -> pd.DataFrame:
    """Run end-to-end: pose estimation -> features -> classification.

    Parameters
    ----------
    input_path : str
        Path to a video file or directory of images.
    output_dir : str
        Directory to store intermediate CSVs and predictions.
    use_mock : bool, optional
        Use mock DLC output instead of real DLC.
    model_path : Optional[str], optional
        Path to a joblib model for ML classification; if missing/unloadable, fallback to heuristic.
    fps : int, optional
        Frames per second for feature computations.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: frame, timestamp, predicted_label.
    """
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    dlc_csv = run_pose_estimation(input_path=input_path, output_dir=output_dir, use_mock=use_mock)
    features = extract_features_from_dlc_csv(dlc_csv, fps=fps)

    # timestamp from frame index and FPS
    frames = features.index.values
    timestamps = frames / float(fps)

    # Try ML classifier if provided
    labels: List[str]
    used_ml = False
    if model_path and _JOBLIB_AVAILABLE:
        try:
            model = joblib.load(model_path)  # type: ignore
            labels = predict_classifier(model, features)
            used_ml = True
            print(f"[CLASSIFY] Used ML model: {model_path}")
        except Exception as e:
            warnings.warn(f"Failed to load/apply model '{model_path}': {e}. Falling back to heuristic.")
            labels = classify_features_heuristic(features)
    else:
        if model_path and not _JOBLIB_AVAILABLE:
            warnings.warn("joblib not available; cannot load model. Using heuristic instead.")
        labels = classify_features_heuristic(features)

    result = pd.DataFrame({
        "frame": frames,
        "timestamp": timestamps,
        "predicted_label": labels
    })
    pred_path = out_dir / "predictions.csv"
    result.to_csv(pred_path, index=False)
    print(f"[CLASSIFY] Saved predictions to {pred_path} ({'ML' if used_ml else 'Heuristic'})")
    return result


# ==================================
# Training utility
# ==================================

# PUBLIC_INTERFACE
def train_activity_model(
    dlc_csv_paths: List[str],
    label_csv_paths: List[str],
    output_model_path: str,
    fps: int = FPS
) -> str:
    """Train a RandomForest model from DLC CSVs and per-frame labels CSVs.

    Parameters
    ----------
    dlc_csv_paths : List[str]
        List of DLC CSV paths.
    label_csv_paths : List[str]
        List of label CSV paths matching dlc_csv_paths (columns: frame,label).
    output_model_path : str
        Where to save the trained model (joblib).
    fps : int, optional
        Frames per second to use for feature extraction.

    Returns
    -------
    str
        Path to the saved model.

    Raises
    ------
    RuntimeError
        If scikit-learn or joblib are not available.
    """
    if not _SKLEARN_AVAILABLE or RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is not available. Cannot train ML classifier.")
    if not _JOBLIB_AVAILABLE:
        raise RuntimeError("joblib is not available. Cannot save model.")

    if len(dlc_csv_paths) != len(label_csv_paths):
        raise ValueError("dlc_csv_paths and label_csv_paths must have the same length.")

    all_features = []
    all_labels = []
    for dlc_csv, lab_csv in zip(dlc_csv_paths, label_csv_paths):
        feats = extract_features_from_dlc_csv(dlc_csv, fps=fps)
        labels_df = pd.read_csv(lab_csv)
        if not {"frame", "label"}.issubset(labels_df.columns):
            raise ValueError(f"Label CSV must contain 'frame' and 'label' columns: {lab_csv}")
        labels_df = labels_df.set_index("frame").loc[feats.index]
        y = labels_df["label"].astype(str)
        # Keep only aligned rows
        mask = y.notna()
        all_features.append(feats[mask])
        all_labels.append(y[mask])

    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_labels, axis=0)

    model = train_classifier(X, y)
    out_path = Path(output_model_path)
    _ensure_dir(out_path.parent)
    joblib.dump(model, out_path)  # type: ignore
    print(f"[TRAIN] Saved model to {out_path} (n={len(y)} samples)")
    return str(out_path)


# ==================================
# CLI
# ==================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bear Activity Classifier with DeepLabCut or mock fallback."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # estimate
    p_est = sub.add_parser("estimate", help="Run pose estimation only (DLC or mock).")
    p_est.add_argument("--input", required=True, help="Path to video file or image directory.")
    p_est.add_argument("--out", required=True, help="Output directory for DLC CSV.")
    p_est.add_argument("--mock", action="store_true", help="Use mock DLC output.")
    p_est.add_argument("--shuffle", type=int, default=1, help="DLC shuffle parameter.")
    p_est.add_argument("--trainingsetindex", type=int, default=0, help="DLC trainingsetindex parameter.")

    # classify
    p_cls = sub.add_parser("classify", help="End-to-end: pose -> features -> predictions.")
    p_cls.add_argument("--input", required=True, help="Path to video file or image directory.")
    p_cls.add_argument("--out", required=True, help="Output directory for predictions.")
    p_cls.add_argument("--mock", action="store_true", help="Use mock DLC output.")
    p_cls.add_argument("--model", default=None, help="Path to trained model (.joblib).")
    p_cls.add_argument("--fps", type=int, default=FPS, help="Frames per second for features.")

    # train
    p_tr = sub.add_parser("train", help="Train a model from DLC CSVs and label CSVs.")
    p_tr.add_argument("--dlc_csvs", nargs="+", required=True, help="List of DLC CSV paths.")
    p_tr.add_argument("--label_csvs", nargs="+", required=True, help="List of label CSV paths.")
    p_tr.add_argument("--out_model", required=True, help="Output path for trained model (.joblib).")
    p_tr.add_argument("--fps", type=int, default=FPS, help="Frames per second for features.")

    return parser.parse_args(argv)


def _cmd_estimate(args: argparse.Namespace) -> int:
    try:
        csv_path = run_pose_estimation(
            input_path=args.input,
            output_dir=args.out,
            use_mock=args.mock,
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex
        )
        print(json.dumps({"status": "ok", "csv": csv_path}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


def _cmd_classify(args: argparse.Namespace) -> int:
    try:
        df = classify_bear_activity(
            input_path=args.input,
            output_dir=args.out,
            use_mock=args.mock,
            model_path=args.model,
            fps=args.fps
        )
        # Print brief summary
        counts = df["predicted_label"].value_counts().to_dict()
        print(json.dumps({"status": "ok", "counts": counts}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


def _cmd_train(args: argparse.Namespace) -> int:
    try:
        path = train_activity_model(
            dlc_csv_paths=args.dlc_csvs,
            label_csv_paths=args.label_csvs,
            output_model_path=args.out_model,
            fps=args.fps
        )
        print(json.dumps({"status": "ok", "model": path}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


if __name__ == "__main__":
    args = _parse_args()
    if args.command == "estimate":
        sys.exit(_cmd_estimate(args))
    elif args.command == "classify":
        sys.exit(_cmd_classify(args))
    elif args.command == "train":
        sys.exit(_cmd_train(args))
    else:
        print(json.dumps({"status": "error", "message": "Unknown command"}))
        sys.exit(2)
