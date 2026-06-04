"""
Shared utilities for template matching experiments.
All approaches produce comparable outputs for fair comparison.
"""
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.ndimage import median_filter
from collections import deque
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io import load_sleap_dannce_keys, load_aligned_data, load_template, get_sessions
from skeleton import normalize_skeleton_batch, project_to_pcs
from config import NODE_IDX

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_session_data(rat, session):
    """
    Load and orient SLEAP + DANNCE keypoints for a session.

    DANNCE is resampled to the SLEAP frame rate (20 Hz) using
    dannce_idx_for_sleap_cams, so both arrays share the same time axis.
    This matches the approach in template_matching.ipynb.

    Returns:
        sleap_3d  : (n_sleap, 23, 3) — z-flipped, at 20 Hz
        dannce_3d : (n_sleap, 23, 3) — resampled to 20 Hz via dannce_idx_for_sleap_cams
        aligned   : alignment dict
    """
    d = load_sleap_dannce_keys(rat, session)
    sleap_3d = d['sleap_keys_3D'].astype(np.float64)      # (n_sleap, 23, 3)

    # SLEAP Z-flip to match DANNCE orientation
    sleap_3d[:, :, 2] = -sleap_3d[:, :, 2]

    # DANNCE: transpose from (n_dannce, 3, 23) → (n_dannce, 23, 3)
    dn = d['dannce_keys_3D'].astype(np.float64)
    if dn.ndim == 4:
        dn = dn.squeeze(axis=1).transpose(0, 2, 1)
    elif dn.ndim == 3 and dn.shape[1] == 3 and dn.shape[2] == 23:
        dn = dn.transpose(0, 2, 1)

    try:
        aligned = load_aligned_data(rat, session)
    except Exception:
        aligned = None

    # Resample DANNCE to SLEAP frame rate using dannce_idx_for_sleap_cams
    # This index maps each SLEAP frame to the nearest DANNCE frame.
    # Skip index 0 (starter frame) to match notebook convention.
    if aligned is not None and 'dannce_idx_for_sleap_cams' in aligned:
        idx = np.array(aligned['dannce_idx_for_sleap_cams'], dtype=int).ravel()
        # The notebook skips index 0 (starter frame); clip to valid range
        idx = idx[1:] if len(idx) > len(sleap_3d) else idx
        idx = np.clip(idx, 0, len(dn) - 1)
        dannce_3d = dn[idx]                                # (n_sleap, 23, 3)
    else:
        # Fallback: nearest-neighbour resample by ratio
        ratio = len(dn) / len(sleap_3d)
        idx = np.clip(np.round(np.arange(len(sleap_3d)) * ratio).astype(int), 0, len(dn) - 1)
        dannce_3d = dn[idx]

    return sleap_3d, dannce_3d, aligned


def get_alignment_index(aligned):
    """
    Return sleap_idx_for_dannce_cams (maps each DANNCE frame → nearest SLEAP frame).
    Returns None if not available.
    """
    if aligned is None:
        return None
    for key in ['sleap_idx_for_dannce_cams', 'sleap_idx']:
        if key in aligned:
            return np.array(aligned[key], dtype=int).ravel()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing / smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_keypoints(keys_3d, method='median', window=5):
    """
    Smooth keypoints along time axis.
    keys_3d : (T, 23, 3)
    method  : 'median', 'savgol', 'ema', 'none'
    window  : int (kernel size for median/savgol; half-life for EMA)
    Returns smoothed (T, 23, 3).
    """
    out = keys_3d.copy()
    T, nk, nd = keys_3d.shape
    if method == 'none' or window <= 1:
        return out
    for k in range(nk):
        for d in range(nd):
            s = keys_3d[:, k, d]
            if method == 'median':
                out[:, k, d] = signal.medfilt(s, kernel_size=window | 1)  # ensure odd
            elif method == 'savgol':
                w = window | 1
                poly = min(3, w - 1)
                out[:, k, d] = signal.savgol_filter(s, w, poly)
            elif method == 'ema':
                # Causal exponential moving average — online-compatible
                alpha = 2.0 / (window + 1)
                out[:, k, d] = _ema(s, alpha)
    return out


def _ema(s, alpha):
    """Vectorized EMA using scipy lfilter (much faster than Python loop)."""
    from scipy.signal import lfilter
    b = [alpha]
    a = [1, -(1 - alpha)]
    zi = np.array([s[0]])
    out, _ = lfilter(b, a, s, zi=zi)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_pairwise_distances(keys_3d):
    """
    Compute upper-triangle pairwise Euclidean distances between all keypoints.
    keys_3d : (T, 23, 3)
    Returns : (T, 253)  — 23*22/2 = 253 unique pairs
    Uses vectorized computation for speed.
    """
    T, nk, _ = keys_3d.shape
    idx_i, idx_j = np.triu_indices(nk, k=1)
    # Vectorized: (T, 253, 3) difference
    diff = keys_3d[:, idx_i, :] - keys_3d[:, idx_j, :]
    # Squared norm then sqrt — faster than np.linalg.norm for large arrays
    return np.sqrt(np.einsum('tpd,tpd->tp', diff, diff))    # (T, 253)


def compute_xyz_flat(keys_3d):
    """Flatten (T,23,3) → (T,69). Used for baseline."""
    T = keys_3d.shape[0]
    return keys_3d.reshape(T, -1)


def compute_velocity(keys_3d, causal=True):
    """
    Per-keypoint speed (L2 norm of frame difference).
    causal=True: uses only past frames (online-compatible).
    Returns (T, 23).
    """
    diff = np.diff(keys_3d, axis=0)                         # (T-1, 23, 3)
    speed = np.linalg.norm(diff, axis=2)                    # (T-1, 23)
    return np.vstack([np.zeros((1, keys_3d.shape[1])), speed])  # (T, 23)


def project_features(features, weights, means):
    """
    Project feature matrix into PC space.
    features : (T, n_feat)
    weights  : (n_pcs, n_feat)
    means    : (n_feat,)
    Returns  : (T, n_pcs)
    """
    return (features - means) @ weights.T


# ─────────────────────────────────────────────────────────────────────────────
# Template matching
# ─────────────────────────────────────────────────────────────────────────────

def run_template_matching(pcs, template, bounds_arr, refractory_frames=30, max_outside=3):
    """
    Sliding-window template matching — vectorized implementation.

    pcs            : (T, n_pcs)
    template       : (win, n_pcs)
    bounds_arr     : (win, n_pcs) — per-timepoint per-PC tolerance
    refractory_frames : int
    max_outside    : int — allow up to this many timepoints outside bounds

    Returns list of match frame indices (end of window).
    """
    T, n_pcs = pcs.shape
    win = template.shape[0]

    # Build sliding window view: (T - win + 1, win, n_pcs)
    # Using stride tricks for zero-copy view
    from numpy.lib.stride_tricks import as_strided
    shape = (T - win + 1, win, n_pcs)
    strides = (pcs.strides[0], pcs.strides[0], pcs.strides[1])
    windows = as_strided(pcs, shape=shape, strides=strides)

    # Count points outside bounds: (T - win + 1,)
    n_outside = np.sum(np.abs(windows - template[None, :, :]) > bounds_arr[None, :, :], axis=(1, 2))

    # Find candidate frames (window END frame = idx + win - 1)
    candidates = np.where(n_outside <= max_outside)[0] + win - 1  # actual frame indices

    # Apply refractory period
    if len(candidates) == 0:
        return []

    matches = [candidates[0]]
    for c in candidates[1:]:
        if c - matches[-1] >= refractory_frames:
            matches.append(c)

    return matches


def compute_bounds(template, feature_stds, bounds_scalar, pcs_to_use):
    """
    Build (win, n_pcs) bounds array using feature_stds * bounds_scalar.
    """
    win, n_pcs = template.shape
    bounds = np.zeros((win, n_pcs))
    for j in range(n_pcs):
        pc_idx = pcs_to_use[j]
        bounds[:, j] = feature_stds[pc_idx] * bounds_scalar
    return bounds


def compute_adaptive_bounds(template, feature_stds, pcs_to_use, scale=1.0):
    """
    Set bounds per-timepoint proportional to template deviation from mean
    (wider at peaks, narrower near baseline). Online-compatible.
    Min bound = 0.25 * std so we never get zero bounds.
    """
    win, n_pcs = template.shape
    bounds = np.zeros((win, n_pcs))
    for j in range(n_pcs):
        pc_idx = pcs_to_use[j]
        std = feature_stds[pc_idx]
        amp = np.abs(template[:, j])
        # bound = scale * (0.25*std + 0.75 * amplitude)
        bounds[:, j] = scale * (0.25 * std + 0.75 * amp)
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# Alignment scoring
# ─────────────────────────────────────────────────────────────────────────────

SLEAP_HZ = 20.0
DANNCE_HZ = 50.0


def estimate_temporal_offset(sleap_matches, dannce_matches,
                              sleap_times_ms, dannce_times_ms,
                              max_pairing_ms=3000):
    """
    Estimate the median temporal offset (SLEAP_time - DANNCE_time) for
    best-matched pairs within max_pairing_ms of each other.
    Returns offset in ms (positive = SLEAP is later).
    """
    sl_t = np.array([sleap_times_ms[s] for s in sleap_matches], dtype=float)
    dn_t = np.array([dannce_times_ms[d] for d in dannce_matches], dtype=float)
    if len(sl_t) == 0 or len(dn_t) == 0:
        return 0.0
    diffs = []
    for dt in dn_t:
        d = sl_t - dt
        best = np.argmin(np.abs(d))
        if np.abs(d[best]) < max_pairing_ms:
            diffs.append(d[best])
    return float(np.median(diffs)) if diffs else 0.0


def compute_alignment(sleap_matches, dannce_matches, tolerance_ms=300.0,
                       sleap_times_ms=None, dannce_times_ms=None,
                       offset_ms=0.0):
    """
    Compare two lists of match frame indices (SLEAP frames vs DANNCE frames).

    If sleap_times_ms and dannce_times_ms (absolute timestamps) are provided,
    they are used to convert frame indices to absolute time for comparison.
    Otherwise, relative frame-based time is used (assumes both start at t=0).

    Returns dict with recall, precision, f1, and counts.
    recall = fraction of ground-truth (dannce) matches also found by sleap.
    """
    if sleap_times_ms is not None and dannce_times_ms is not None:
        sleap_ms = np.array([sleap_times_ms[f] for f in sleap_matches], dtype=float)
        dannce_ms = np.array([dannce_times_ms[f] for f in dannce_matches], dtype=float)
    else:
        # Fallback: relative time from frame 0 (only valid if both start simultaneously)
        sleap_ms = np.array(sleap_matches, dtype=float) / SLEAP_HZ * 1000.0
        dannce_ms = np.array(dannce_matches, dtype=float) / DANNCE_HZ * 1000.0

    # Apply offset correction (subtract median offset from SLEAP times)
    sleap_ms = sleap_ms - offset_ms

    if len(dannce_ms) == 0 or len(sleap_ms) == 0:
        return dict(
            n_sleap=len(sleap_ms), n_dannce=len(dannce_ms),
            n_both=0, n_sleap_only=len(sleap_ms), n_dannce_only=len(dannce_ms),
            recall=0.0, precision=0.0, f1=0.0
        )

    # For each DANNCE match, find closest SLEAP match
    matched_sleap_set = set()
    matched_dannce = 0
    for d_ms in dannce_ms:
        diffs = np.abs(sleap_ms - d_ms)
        best = np.argmin(diffs)
        if diffs[best] <= tolerance_ms:
            matched_dannce += 1
            matched_sleap_set.add(best)

    n_both = matched_dannce
    recall    = n_both / len(dannce_ms) if len(dannce_ms) > 0 else 0.0
    precision = n_both / len(sleap_ms)  if len(sleap_ms) > 0  else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return dict(
        n_sleap=len(sleap_ms),
        n_dannce=len(dannce_ms),
        n_both=n_both,
        n_sleap_only=len(sleap_ms) - len(matched_sleap_set),
        n_dannce_only=len(dannce_ms) - n_both,
        recall=recall,
        precision=precision,
        f1=f1,
    )


def compute_alignment_multi_tol(sleap_matches, dannce_matches,
                                 tolerances_ms=(100, 300, 500),
                                 sleap_times_ms=None, dannce_times_ms=None,
                                 offset_ms=0.0):
    """Run alignment scoring at multiple tolerances."""
    return {
        f'tol_{t}ms': compute_alignment(sleap_matches, dannce_matches, t,
                                         sleap_times_ms, dannce_times_ms,
                                         offset_ms=offset_ms)
        for t in tolerances_ms
    }
