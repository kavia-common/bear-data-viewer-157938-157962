#!/usr/bin/env python3
"""
Bear Activity Classifier - DeepLabCut required, heuristic-only.

This single-file module performs:
1) DeepLabCut (DLC) inference for pose estimation on videos or image folders.
2) Feature extraction from DLC CSV outputs (velocities, spans, spine angle).
3) Rule-based heuristic classification into ['sleeping', 'standing', 'moving'].

Important:
- No model training utilities. No scikit-learn/joblib usage.
- No mock/stub pose fallback. DeepLabCut must be installed and configured.
- You must provide a valid DLC config/project path in CONFIG.

Usage:
- Open this file and edit the CONFIG block inside main() to your environment.
- Then run:
    python backend/bear_activity_classifier.py
- If DeepLabCut is missing or the DLC config path is invalid, the script prints a clear error and exits.

DLC CSV format note:
- DLC CSV typically uses a MultiIndex header [scorer, bodypart, coords], where coords in [x, y, likelihood].
"""

from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _is_video_file(p: Path) -> bool:
    """Quick heuristic to determine if a path points to a video file."""
    return p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def _list_images_in_dir(p: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


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
            if isinstance(col, str) and "_" in col:
                parts.append(col.split("_")[0])
    return sorted(set(parts))


# PUBLIC_INTERFACE
def run_pose_estimation(
    input_path: str,
    output_dir: str,
    dlc_config_path: str,
    shuffle: int = 1,
    trainingsetindex: int = 0
) -> str:
    """
    Run DeepLabCut pose estimation on a video file or directory of images.

    Parameters
    ----------
    input_path : str
        Path to a video file or a directory of images.
    output_dir : str
        Directory to store DLC output CSV file(s).
    dlc_config_path : str
        Path to the DeepLabCut project/config YAML.
    shuffle : int, optional
        DLC shuffle parameter.
    trainingsetindex : int, optional
        DLC trainingsetindex parameter.

    Returns
    -------
    str
        Path to the resulting DLC CSV file (copied into output_dir).

    Raises
    ------
    RuntimeError
        If DeepLabCut is not importable.
    FileNotFoundError
        If input_path or dlc_config_path do not exist.
    ValueError
        If input_path is neither a video nor a directory with images.
    """
    # Import DLC strictly here to raise clear error early in pipeline.
    try:
        import deeplabcut as dlc  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "DeepLabCut is required but not installed. Please install 'deeplabcut' "
            "and ensure it is importable in this environment."
        ) from e

    input_p = Path(input_path)
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    if not input_p.exists():
        raise FileNotFoundError(f"Input path not found: {input_p}")

    cfg_p = Path(dlc_config_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f"DLC config/project path not found: {cfg_p}")

    is_video = _is_video_file(input_p)
    is_dir = input_p.is_dir()

    if is_video:
        videos = [str(input_p)]
        print(f"[DLC] Analyzing video: {videos[0]}")
        try:
            dlc.analyze_videos(
                str(cfg_p),
                videos,
                videotype=input_p.suffix,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                save_as_csv=True
            )
        except Exception as e:
            raise RuntimeError(f"DeepLabCut analyze_videos failed: {e}") from e

        candidate_dir = input_p.parent
        csvs = sorted(candidate_dir.glob("*filtered*.csv")) + sorted(candidate_dir.glob("*.csv"))
        if not csvs:
            raise RuntimeError("No DLC CSV output found after analyze_videos.")
        csv_path = csvs[-1]
        final_csv = out_dir / f"{input_p.stem}_DLC_output.csv"
        # Ensure we read with proper header; if it fails, fallback to single header
        try:
            pd.read_csv(csv_path, header=[0, 1, 2], index_col=0).to_csv(final_csv)
        except Exception:
            warnings.warn("CSV did not have a 3-level header; saving as flat header CSV.")
            pd.read_csv(csv_path, index_col=0).to_csv(final_csv)
        print(f"[DLC] Saved CSV to {final_csv}")
        return str(final_csv)

    if is_dir:
        images = _list_images_in_dir(input_p)
        if not images:
            raise ValueError(f"No images found in directory: {input_p}")
        print(f"[DLC] Analyzing time-lapse images in: {input_p}")
        try:
            dlc.analyze_time_lapse_images(
                str(cfg_p),
                str(input_p),
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                save_as_csv=True
            )
        except Exception as e:
            raise RuntimeError(f"DeepLabCut analyze_time_lapse_images failed: {e}") from e

        csvs = sorted(input_p.glob("*filtered*.csv")) + sorted(input_p.glob("*.csv"))
        if not csvs:
            raise RuntimeError("No DLC CSV output found after analyze_time_lapse_images.")
        csv_path = csvs[-1]
        final_csv = out_dir / f"{input_p.name}_DLC_output.csv"
        try:
            pd.read_csv(csv_path, header=[0, 1, 2], index_col=0).to_csv(final_csv)
        except Exception:
            warnings.warn("CSV did not have a 3-level header; saving as flat header CSV.")
            pd.read_csv(csv_path, index_col=0).to_csv(final_csv)
        print(f"[DLC] Saved CSV to {final_csv}")
        return str(final_csv)

    raise ValueError("Input path must be a video file or a directory of images.")


def _rolling_median(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).median()


def _compute_spine_angle(
    df: pd.DataFrame,
    parts: List[str],
    likelihood_threshold: float
) -> pd.Series:
    """Compute spine angle (degrees) using vector between shoulders and hips."""
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
        if isinstance(df.columns, pd.MultiIndex):
            x = df.xs((bp, "x"), level=("bodypart", "coords"), axis=1, drop_level=False).iloc[:, 0]
            y = df.xs((bp, "y"), level=("bodypart", "coords"), axis=1, drop_level=False).iloc[:, 0]
            l = df.xs((bp, "likelihood"), level=("bodypart", "coords"), axis=1, drop_level=False).iloc[:, 0]
            return x, y, l
        # flat columns: bp_x, bp_y, bp_likelihood
        x = df.get(f"{bp}_x", pd.Series(np.nan, index=df.index))
        y = df.get(f"{bp}_y", pd.Series(np.nan, index=df.index))
        l = df.get(f"{bp}_likelihood", pd.Series(np.nan, index=df.index))
        return x, y, l

    def _mean_point(bps: List[Optional[str]]) -> Tuple[pd.Series, pd.Series, pd.Series]:
        xs, ys, ls = [], [], []
        for bp in bps:
            if bp is None:
                continue
            x, y, l = _get_xy_like(bp)
            xs.append(x); ys.append(y); ls.append(l)
        if not xs:
            nan_series = pd.Series(np.nan, index=df.index)
            return nan_series, nan_series, nan_series
        xs = pd.concat(xs, axis=1)
        ys = pd.concat(ys, axis=1)
        ls = pd.concat(ls, axis=1)
        mask = ls >= likelihood_threshold
        xs_masked = xs.where(mask)
        ys_masked = ys.where(mask)
        x_mean = xs_masked.mean(axis=1)
        y_mean = ys_masked.mean(axis=1)
        l_mean = ls.mean(axis=1)
        return x_mean, y_mean, l_mean

    shoulder_x, shoulder_y, _ = _mean_point([ls, rs])
    hip_x, hip_y, _ = _mean_point([lh, rh])

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    angle_rad = np.arctan2(dx, dy)
    angle_deg = np.degrees(angle_rad)
    return angle_deg.abs()


# PUBLIC_INTERFACE
def extract_features_from_dlc_csv(
    csv_path: str,
    fps: int = 30,
    likelihood_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Extract per-frame features from a DLC CSV file.

    Features:
    - mean_likelihood: average confidence across parts.
    - com_x, com_y: center of mass of confident keypoints.
    - com_velocity: magnitude of COM velocity (pixels/sec).
    - vertical_span, horizontal_span: bbox spans of confident keypoints.
    - spine_angle_from_vertical: abs angle (deg) between spine axis and vertical.

    Returns a DataFrame indexed by frame with these features.
    """
    # Try reading as multiindex; fallback to flat header
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    except Exception:
        df = pd.read_csv(csv_path, index_col=0)

    parts = _infer_body_parts_from_csv(df)

    xs, ys, ls = [], [], []
    for bp in parts:
        if isinstance(df.columns, pd.MultiIndex):
            x = df.xs((bp, "x"), level=("bodypart", "coords"), axis=1, drop_level=False)
            y = df.xs((bp, "y"), level=("bodypart", "coords"), axis=1, drop_level=False)
            l = df.xs((bp, "likelihood"), level=("bodypart", "coords"), axis=1, drop_level=False)
            xs.append(x.iloc[:, 0]); ys.append(y.iloc[:, 0]); ls.append(l.iloc[:, 0])
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

    mean_likelihood = L.mean(axis=1).fillna(0.0)
    conf_mask = L >= likelihood_threshold
    Xc = X.where(conf_mask)
    Yc = Y.where(conf_mask)

    com_x = Xc.mean(axis=1)
    com_y = Yc.mean(axis=1)

    dcom_x = com_x.diff().fillna(0.0)
    dcom_y = com_y.diff().fillna(0.0)
    com_velocity = np.sqrt(dcom_x.pow(2) + dcom_y.pow(2)) * float(fps)

    x_min = Xc.min(axis=1)
    x_max = Xc.max(axis=1)
    y_min = Yc.min(axis=1)
    y_max = Yc.max(axis=1)
    horizontal_span = (x_max - x_min).fillna(0.0)
    vertical_span = (y_max - y_min).fillna(0.0)

    spine_angle_from_vertical = _compute_spine_angle(df, parts, likelihood_threshold)

    features = pd.DataFrame({
        "mean_likelihood": _rolling_median(mean_likelihood, 5),
        "com_x": _rolling_median(com_x, 5),
        "com_y": _rolling_median(com_y, 5),
        "com_velocity": _rolling_median(com_velocity, 5),
        "vertical_span": _rolling_median(vertical_span, 5),
        "horizontal_span": _rolling_median(horizontal_span, 5),
        "spine_angle_from_vertical": _rolling_median(spine_angle_from_vertical, 5),
    })
    features.index.name = "frame"
    features = features.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return features


def _adaptive_thresholds(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Compute adaptive thresholds for velocity and spans using quantiles."""
    v = df["com_velocity"].clip(lower=0)
    vert = df["vertical_span"].clip(lower=0)
    low_vel = float(np.quantile(v, 0.2))
    move_vel = float(np.quantile(v, 0.7))
    vert_split = float(np.quantile(vert, 0.5))
    if move_vel <= low_vel:
        move_vel = low_vel * 1.5 + 1.0
    return low_vel, move_vel, vert_split


# PUBLIC_INTERFACE
def classify_features_heuristic(df_features: pd.DataFrame) -> List[str]:
    """
    Heuristic classifier mapping features to activity labels.

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

        if like >= 0.5 and v < low_vel:
            if vert <= 0.5 * vert_split:
                labels.append("sleeping")
            else:
                if spine <= 25.0:
                    labels.append("standing")
                else:
                    labels.append("sleeping")
        else:
            if spine <= 25.0 and vert >= 0.5 * vert_split:
                labels.append("standing")
            else:
                labels.append("moving" if v >= 0.5 * move_vel else "standing")
    return labels


# PUBLIC_INTERFACE
def classify_bear_activity(
    input_path: str,
    output_dir: str,
    dlc_config_path: str,
    fps: int = 30,
    shuffle: int = 1,
    trainingsetindex: int = 0
) -> pd.DataFrame:
    """
    End-to-end: DLC pose estimation -> feature extraction -> heuristic classification.

    Parameters
    ----------
    input_path : str
        Path to a video file or directory of images.
    output_dir : str
        Directory to store intermediate CSVs and predictions.
    dlc_config_path : str
        Path to DeepLabCut config/project YAML.
    fps : int, optional
        Frames per second for feature computations.
    shuffle : int, optional
        DLC shuffle parameter.
    trainingsetindex : int, optional
        DLC trainingsetindex parameter.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: frame, timestamp, predicted_label.
    """
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    dlc_csv = run_pose_estimation(
        input_path=input_path,
        output_dir=output_dir,
        dlc_config_path=dlc_config_path,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex
    )
    features = extract_features_from_dlc_csv(dlc_csv, fps=fps)

    frames = features.index.values
    timestamps = frames / float(fps)
    labels = classify_features_heuristic(features)

    result = pd.DataFrame({
        "frame": frames,
        "timestamp": timestamps,
        "predicted_label": labels
    })
    pred_path = out_dir / "predictions.csv"
    result.to_csv(pred_path, index=False)
    print(f"[CLASSIFY] Saved predictions to {pred_path} (Heuristic)")
    return result


# PUBLIC_INTERFACE
def main() -> int:
    """
    Entry point without CLI arguments.

    Edit the CONFIG block below to control behavior.

    Required CONFIG keys:
    - mode: 'estimate' | 'classify'
    - input_path: path to video file or image directory
    - output_dir: directory to write outputs (CSV, predictions)
    - dlc_config_path: path to DLC project/config YAML (REQUIRED)

    Optional:
    - fps: frames per second for features/velocity (default 30)
    - shuffle: DLC shuffle parameter (int)
    - trainingsetindex: DLC trainingsetindex parameter (int)
    """
    # ======== CONFIG: EDIT THESE VALUES ========
    CONFIG = {
        "mode": "classify",                        # 'estimate' or 'classify'
        "input_path": "sample_data/video.mp4",     # path to video or image dir
        "output_dir": "outputs",                   # directory to write CSVs/predictions

        # DLC: REQUIRED - set to your DLC project/config YAML
        "dlc_config_path": "/absolute/path/to/your/DLC/project/config.yaml",

        # Optional processing parameters
        "fps": 30,
        "shuffle": 1,
        "trainingsetindex": 0,
    }
    # ======== END CONFIG ========

    mode = str(CONFIG.get("mode", "classify")).strip().lower()
    input_path = str(CONFIG.get("input_path", "")).strip()
    output_dir = str(CONFIG.get("output_dir", "")).strip()
    dlc_config_path = str(CONFIG.get("dlc_config_path", "")).strip()

    # Validate DLC import early to give a clear message
    try:
        import deeplabcut as _  # noqa: F401
    except Exception:
        print(json.dumps({
            "status": "error",
            "message": "DeepLabCut is required but not installed. Please install 'deeplabcut' and retry."
        }))
        return 1

    # Validate paths
    if not input_path:
        print(json.dumps({"status": "error", "message": "CONFIG['input_path'] is required."}))
        return 1
    if not output_dir:
        print(json.dumps({"status": "error", "message": "CONFIG['output_dir'] is required."}))
        return 1
    if not dlc_config_path:
        print(json.dumps({"status": "error", "message": "CONFIG['dlc_config_path'] is required and must point to a valid DLC config YAML."}))
        return 1
    if not Path(dlc_config_path).exists():
        print(json.dumps({"status": "error", "message": f"DLC config path not found: {dlc_config_path}"}))
        return 1

    try:
        if mode == "estimate":
            csv_path = run_pose_estimation(
                input_path=input_path,
                output_dir=output_dir,
                dlc_config_path=dlc_config_path,
                shuffle=int(CONFIG.get("shuffle", 1)),
                trainingsetindex=int(CONFIG.get("trainingsetindex", 0)),
            )
            print(json.dumps({"status": "ok", "mode": "estimate", "csv": csv_path}))
            return 0

        if mode == "classify":
            df = classify_bear_activity(
                input_path=input_path,
                output_dir=output_dir,
                dlc_config_path=dlc_config_path,
                fps=int(CONFIG.get("fps", 30)),
                shuffle=int(CONFIG.get("shuffle", 1)),
                trainingsetindex=int(CONFIG.get("trainingsetindex", 0)),
            )
            counts = df["predicted_label"].value_counts().to_dict()
            print(json.dumps({"status": "ok", "mode": "classify", "counts": counts}))
            return 0

        print(json.dumps({"status": "error", "message": f"Unknown mode '{mode}'"}))
        return 2

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
