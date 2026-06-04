"""
Data loading utilities for sessions, aligned data, templates, and calibration.
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from config import (
    DATA_ROOT, SESSION_CSV, DANNCE_CODE, SLEAP_CODE, SYNC_CODE,
    TONE_CODE, LICK_CODE, REWARD_ONSET_CODE, REWARD_WINDOW_EXPIRED_CODE,
    REWARD_COMPLETION_CODE, processed_path, dannce_path, sleap_path,
    template_path,
)


# ---------------------------------------------------------------------------
# Session list
# ---------------------------------------------------------------------------
def load_session_df(csv_path=SESSION_CSV):
    """Load the canonical session CSV. Columns: rat, session, task, template_file, etc."""
    return pd.read_csv(csv_path)


def get_sessions(rat=None, task=None, csv_path=SESSION_CSV):
    """Filter sessions by rat and/or task type."""
    df = load_session_df(csv_path)
    if rat is not None:
        df = df[df["rat"] == rat]
    if task is not None:
        df = df[df["task"] == task]
    return df


# ---------------------------------------------------------------------------
# Processed / aligned data
# ---------------------------------------------------------------------------
def load_aligned_data(rat, session, fmt="mat"):
    """Load aligned_data from processed folder. Returns dict."""
    pp = processed_path(rat, session)
    if fmt == "mat":
        return sio.loadmat(os.path.join(pp, "aligned_data.mat"), simplify_cells=True)
    else:
        return dict(np.load(os.path.join(pp, "aligned_data.npz"), allow_pickle=True))
    

def load_sleap_dannce_keys(rat, session, fmt="mat"):
    """Load sleap_dannce_keys from processed folder. Returns dict."""
    pp = processed_path(rat, session)
    
    if fmt == 'mat':
        return sio.loadmat(os.path.join(pp, "sleap_dannce_keys.mat"), simplify_cells=True)
    else:
        return dict(np.load(os.path.join(pp, "sleap_dannce_keys.npz"), allow_pickle=True))


# ---------------------------------------------------------------------------
# DANNCE predictions (raw)
# ---------------------------------------------------------------------------
def load_dannce_predictions(rat, session):
    """Load raw DANNCE predictions. Returns (n_frames, 23, 3) array."""
    dp = dannce_path(rat, session)
    pred_path = os.path.join(dp, "DANNCE", "predict_results", "save_data_AVG0.mat")
    data = sio.loadmat(pred_path, simplify_cells=True)
    pred = data["pred"]
    # DANNCE stores as (n_frames, 1, 3, 23) — reshape to (n_frames, 23, 3)
    if pred.ndim == 4:
        pred = pred.squeeze(axis=1)  # (n_frames, 3, 23)
        pred = pred.transpose(0, 2, 1)  # (n_frames, 23, 3)
    return pred


# ---------------------------------------------------------------------------
# SLEAP 3D triangulated keypoints
# ---------------------------------------------------------------------------
def load_sleap_keys_3d(rat, session):
    """Load SLEAP triangulated 3D keypoints. Returns (n_frames, 23, 3)."""
    sp = sleap_path(rat, session)
    return np.load(os.path.join(sp, "triang_keys_3D.npy"))


def load_sleap_keys_2d(rat, session):
    """Load SLEAP 2D keypoints. Returns (n_frames, n_cams, 23, 3) where last dim is (x, y, confidence)."""
    sp = sleap_path(rat, session)
    return np.load(os.path.join(sp, "sleap_keys_2D.npy"))


# ---------------------------------------------------------------------------
# OPCON events
# ---------------------------------------------------------------------------
def load_opcon_events(rat, session):
    """Load ratBoops event file. Returns dict of event arrays keyed by name."""
    dp = dannce_path(rat, session)
    df = pd.read_csv(
        os.path.join(dp, "ratBoops"),
        header=None,
        names=["event_code", "value", "timestamp_ms"],
        dtype=int,
    )
    return {
        "dannce_frame_times_ms": df.loc[df.event_code == DANNCE_CODE, "timestamp_ms"].to_numpy(),
        "sleap_frame_times_ms": df.loc[df.event_code == SLEAP_CODE, "timestamp_ms"].to_numpy(),
        "sync_times_ms": df.loc[df.event_code == SYNC_CODE, "timestamp_ms"].to_numpy(),
        "tone_times_ms": df.loc[df.event_code == TONE_CODE, "timestamp_ms"].to_numpy(),
        "lick_times_ms": df.loc[df.event_code == LICK_CODE, "timestamp_ms"].to_numpy(),
        "reward_start_times_ms": df.loc[df.event_code == REWARD_ONSET_CODE, "timestamp_ms"].to_numpy(),
        "reward_window_end_times_ms": df.loc[df.event_code == REWARD_WINDOW_EXPIRED_CODE, "timestamp_ms"].to_numpy(),
        "reward_end_times_ms": df.loc[df.event_code == REWARD_COMPLETION_CODE, "timestamp_ms"].to_numpy(),
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
def load_template(rat, template_file):
    """Load a template .npz file. Returns dict with template, pc_weights, feature_means, bounds, etc."""
    tp = template_path(rat, template_file)
    return dict(np.load(tp, allow_pickle=True))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def load_calibration(calibration_folder, cam_files=None):
    """
    Load camera calibration parameters from a calibration folder.

    Parameters
    ----------
    calibration_folder : str
        Path to the calibration folder (e.g., .../calibration/2025_07_27)
    cam_files : list of str, optional
        Ordered list of calibration .mat filenames. If None, auto-detects
        files matching hires_cam*_params.mat.

    Returns
    -------
    P_list : list of (3, 4) arrays
        Projection matrices for each camera.
    K_list : list of (3, 3) arrays
        Intrinsic matrices.
    dist_coefs : list of lists
        Distortion coefficients [k1, k2, p1, p2] per camera.
    """
    if cam_files is None:
        cam_files = sorted(
            f for f in os.listdir(calibration_folder)
            if f.startswith("hires_cam") and f.endswith("_params.mat")
        )

    P_list, K_list, dist_coefs = [], [], []
    for fname in cam_files:
        params = sio.loadmat(os.path.join(calibration_folder, fname), simplify_cells=True)
        K = np.transpose(params["K"])
        r = np.transpose(params["r"])
        t = -params["t"]
        Rdist = params["RDistort"]
        Tdist = params["TDistort"]
        # P = I @ [R | t] because undistortion already applies K
        P = np.eye(3) @ np.hstack((r, t.reshape(3, 1)))
        P_list.append(P)
        K_list.append(K)
        dist_coefs.append([Rdist[0], Rdist[1], Tdist[0], Tdist[1]])

    return P_list, K_list, dist_coefs


# ---------------------------------------------------------------------------
# Frame metadata
# ---------------------------------------------------------------------------
def load_frame_metadata(rat, session, source="dannce"):
    """Load metadata.csv to get total frame count."""
    if source == "dannce":
        meta_path = os.path.join(dannce_path(rat, session), "videos", "Camera1", "metadata.csv")
    else:
        meta_path = os.path.join(sleap_path(rat, session), "Camera1", "metadata.csv")
    df = pd.read_csv(meta_path)
    total_row = df[df.iloc[:, 0] == "totalFrames"]
    return int(total_row.iloc[0, 1])


def load_behavior_log(rat, session):
    """Load behavior_log.csv from SLEAP folder."""
    sp = sleap_path(rat, session)
    return pd.read_csv(os.path.join(sp, "behavior_log.csv"))


def load_frame_mapping(rat, session):
    """Load frame_mapping.csv from SLEAP folder."""
    sp = sleap_path(rat, session)
    return pd.read_csv(os.path.join(sp, "frame_mapping.csv"))
