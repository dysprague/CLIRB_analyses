"""
Quality check utilities for SLEAP-DANNCE keypoint alignment and tracking.

Main capabilities:
  1. Find linear transformation (rotation, scale, translation) to align SLEAP ↔ DANNCE
  2. Track per-keypoint distances within and across sessions
  3. Generate QC video with 2D overlay + 3D comparison
  4. Temporal alignment quality analysis
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from config import NODES, EDGES, N_KEYPOINTS

from scipy.ndimage import median_filter
from visualization import session_to_datetime


# ===========================================================================
# 1. Linear alignment: find best transform from SLEAP to DANNCE
# ===========================================================================
def _try_alignment(sleap_pts, dannce_pts):
    """
    Find affine transform (rotation + scale + translation) that maps
    sleap_pts to dannce_pts using Procrustes analysis.

    Parameters
    ----------
    sleap_pts : (M, 3) — flattened keypoints from selected frames
    dannce_pts : (M, 3) — corresponding DANNCE keypoints

    Returns
    -------
    R : (3, 3) rotation matrix
    s : float scale factor
    t : (3,) translation vector
    residual : float mean per-point error after alignment
    """
    # Center both
    mu_s = sleap_pts.mean(axis=0)
    mu_d = dannce_pts.mean(axis=0)
    S = sleap_pts - mu_s
    D = dannce_pts - mu_d

    # Optimal rotation via SVD
    H = S.T @ D
    U, Sigma, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.diag([1, 1, np.sign(det)])
    R = Vt.T @ sign_mat @ U.T

    # Scale: ratio of norms
    s = np.trace(R @ H) / np.trace(S.T @ S)

    # Translation
    t = mu_d - s * R @ mu_s

    # Residual
    aligned = s * (sleap_pts @ R.T) + t
    residual = np.mean(np.linalg.norm(aligned - dannce_pts, axis=1))

    return R, s, t, residual


def find_sleap_dannce_alignment(sleap_3d, dannce_3d, n_sample_frames=10, seed=42,
                                 try_z_flip=True):
    """
    Find the best linear transformation to align SLEAP 3D keypoints
    to DANNCE 3D keypoints.

    Optionally tries both z-flipped and non-flipped SLEAP and picks
    whichever gives lower residual.

    Parameters
    ----------
    sleap_3d : (n_sleap_frames, 23, 3) — SLEAP triangulated 3D keypoints
    dannce_3d : (n_dannce_frames, 23, 3) — DANNCE 3D keypoints
    aligned_indices : (n_sleap_frames,) — for each SLEAP frame, the
        corresponding DANNCE frame index
    n_sample_frames : int — number of random frames for fitting
    seed : int
    try_z_flip : bool — if True, try both z-flipped and unflipped

    Returns
    -------
    dict with keys:
        R, s, t : transform parameters
        residual : fit residual
        z_flipped : bool — whether z was flipped
        apply : callable — function(sleap_3d) -> aligned_sleap_3d
    """
    rng = np.random.default_rng(seed)

    # Pick random frames that have valid aligned indices

    if sleap_3d.shape[0] < n_sample_frames:
        sample_idx = np.arange(sleap_3d.shape[0])
    else:
        sample_idx = rng.choice(np.arange(sleap_3d.shape[0]), size=n_sample_frames, replace=False)

    # Gather matched keypoints
    sleap_sample = sleap_3d[sample_idx]  # (n, 23, 3)
    dannce_sample = dannce_3d[sample_idx]  # (n, 23, 3)

    # Flatten: (n * 23, 3)
    sleap_flat = sleap_sample.reshape(-1, 3)
    dannce_flat = dannce_sample.reshape(-1, 3)

    results = []

    # Try without z-flip
    R, s, t, res = _try_alignment(sleap_flat, dannce_flat)
    results.append({"R": R, "s": s, "t": t, "residual": res, "z_flipped": False})

    if try_z_flip:
        # Try with z-flip
        sleap_flipped = sleap_flat.copy()
        sleap_flipped[:, 2] = -sleap_flipped[:, 2]
        R_f, s_f, t_f, res_f = _try_alignment(sleap_flipped, dannce_flat)
        results.append({"R": R_f, "s": s_f, "t": t_f, "residual": res_f, "z_flipped": True})

    # Pick best
    best = min(results, key=lambda x: x["residual"])

    def apply_transform(pts):
        """Apply the fitted transform to SLEAP points. pts: (..., 3)"""
        shape = pts.shape
        flat = pts.reshape(-1, 3)
        if best["z_flipped"]:
            flat = flat.copy()
            flat[:, 2] = -flat[:, 2]
        aligned = best["s"] * (flat @ best["R"].T) + best["t"]
        return aligned.reshape(shape)

    best["apply"] = apply_transform

    # Also store the inverse transform (DANNCE -> SLEAP space)
    R_inv = best["R"].T
    s_inv = 1.0 / best["s"]
    t_inv = -s_inv * (R_inv @ best["t"])

    def apply_inverse(pts):
        """Apply the inverse transform: map DANNCE points to SLEAP space. pts: (..., 3)"""
        shape = pts.shape
        flat = pts.reshape(-1, 3)
        result = s_inv * (flat @ R_inv.T) + t_inv
        if best["z_flipped"]:
            result = result.copy()
            result[:, 2] = -result[:, 2]
        return result.reshape(shape)

    best["apply_inverse"] = apply_inverse
    return best


# ===========================================================================
# 2. Per-keypoint distance tracking
# ===========================================================================
def compute_per_keypoint_distances(sleap_3d_aligned, dannce_3d):
    """
    Compute per-keypoint Euclidean distances between aligned SLEAP and DANNCE.

    Parameters
    ----------
    sleap_3d_aligned : (n_sleap, 23, 3) — SLEAP already transformed to DANNCE space
    dannce_3d : (n_dannce, 23, 3)
    aligned_indices : (n_sleap,) — DANNCE frame index for each SLEAP frame

    Returns
    -------
    distances : (n_sleap, 23) — per-keypoint per-frame distances
    """
    n = len(sleap_3d_aligned)

    distances = np.full((n, N_KEYPOINTS), np.nan)
    sleap_matched = sleap_3d_aligned
    dannce_matched = dannce_3d
    distances = np.linalg.norm(sleap_matched - dannce_matched, axis=2)

    return distances


def summarize_keypoint_distances(distances):
    """
    Summarize per-keypoint distance statistics.

    Parameters
    ----------
    distances : (n_frames, 23) from compute_per_keypoint_distances

    Returns
    -------
    dict with per-keypoint mean, median, std, and overall stats
    """
    valid = ~np.isnan(distances)
    per_kp_mean = np.nanmean(distances, axis=0)
    per_kp_median = np.nanmedian(distances, axis=0)
    per_kp_std = np.nanstd(distances, axis=0)
    overall_mean = np.nanmean(distances)

    return {
        "per_keypoint_mean": per_kp_mean,
        "per_keypoint_median": per_kp_median,
        "per_keypoint_std": per_kp_std,
        "keypoint_names": NODES,
        "overall_mean": overall_mean,
        "overall_median": np.nanmedian(distances),
        "n_valid_frames": np.sum(valid[:, 0]),
    }


def compute_session_qc_summary(rat, session, alignment_result=None,
                                n_sample_frames=10, seed=42):
    """
    Compute a full QC summary for a single session.

    Parameters
    ----------
    rat : str
    session : str
    alignment_result : dict, optional — pre-computed alignment. If None,
        computes fresh alignment.
    n_sample_frames : int — frames for alignment fitting
    seed : int

    Returns
    -------
    dict with alignment info, distance summary, temporal stats
    """
    from data_io import load_sleap_dannce_keys, load_aligned_data

    keys = load_sleap_dannce_keys(rat, session, fmt='mat')
    aligned = load_aligned_data(rat, session, fmt='mat')

    sleap_3d = median_filter(keys['sleap_keys_3D'], size=(11, 1, 1))
    dannce_3d = keys['dannce_keys_3D']

    # Handle DANNCE shape if needed
    if dannce_3d.ndim == 4:
        dannce_3d = dannce_3d.squeeze(axis=1).transpose(0, 2, 1)
    else:
        dannce_3d = dannce_3d.transpose(0, 2, 1)

    dannce_3d = median_filter(dannce_3d, size=(25, 1, 1))

    # Get alignment indices
    aligned_indices = aligned["dannce_idx_for_sleap_cams"].astype(int).ravel()

    # Compute or use provided alignment
    if alignment_result is None:
        alignment_result = find_sleap_dannce_alignment(
            sleap_3d, dannce_3d, aligned_indices,
            n_sample_frames=n_sample_frames, seed=seed
        )

    # Apply alignment and compute distances
    sleap_aligned = alignment_result["apply"](sleap_3d)
    distances = compute_per_keypoint_distances(sleap_aligned, dannce_3d, aligned_indices)
    summary = summarize_keypoint_distances(distances)

    # Temporal alignment quality
    temporal = compute_temporal_alignment_quality(aligned, keys)

    return {
        "alignment": alignment_result,
        "distances": distances,
        "summary": summary,
        "temporal": temporal,
        "rat": rat,
        "session": session,
    }


# ===========================================================================
# 3. Temporal alignment quality
# ===========================================================================
def compute_temporal_alignment_quality(aligned_data, keys_data=None):
    """
    Assess the quality of temporal alignment between SLEAP and DANNCE.

    Returns
    -------
    dict with:
        sleap_dannce_time_diffs_ms : per-frame time difference between
            matched SLEAP and DANNCE frames
        mean_time_diff_ms : mean absolute time difference
        max_time_diff_ms : max absolute time difference
        sleap_frame_intervals_ms : inter-frame intervals for SLEAP
        dannce_frame_intervals_ms : inter-frame intervals for DANNCE
    """
    sleap_times = aligned_data["sleap_times_ms"].ravel()
    dannce_times = aligned_data["dannce_times_ms"].ravel()
    dannce_idx_for_sleap = aligned_data["dannce_idx_for_sleap_cams"].astype(int).ravel()

    n = min(len(sleap_times), len(dannce_idx_for_sleap))
    valid = dannce_idx_for_sleap[:n] < len(dannce_times)
    idx = np.where(valid)[0]

    time_diffs = np.full(n, np.nan)
    time_diffs[idx] = np.abs(
        sleap_times[idx] - dannce_times[dannce_idx_for_sleap[idx]]
    )

    return {
        "sleap_dannce_time_diffs_ms": time_diffs,
        "mean_time_diff_ms": np.nanmean(time_diffs),
        "max_time_diff_ms": np.nanmax(time_diffs) if np.any(~np.isnan(time_diffs)) else np.nan,
        "median_time_diff_ms": np.nanmedian(time_diffs),
        "sleap_frame_intervals_ms": np.diff(sleap_times),
        "dannce_frame_intervals_ms": np.diff(dannce_times),
    }


# ===========================================================================
# 4. Multi-session distance tracking
# ===========================================================================
def track_distances_across_sessions(session_list, alignment_per_session=None,
                                     n_sample_frames=10, seed=42):
    """
    Track per-keypoint SLEAP-DANNCE distances across multiple sessions.

    Parameters
    ----------
    session_list : list of (rat, session) tuples
    alignment_per_session : dict, optional — pre-computed alignments keyed
        by (rat, session). If None, computes per-session.
    n_sample_frames : int
    seed : int

    Returns
    -------
    results : list of dicts (one per session) from compute_session_qc_summary
    """
    results = []
    for rat, session in session_list:
        print(f"Processing {rat}/{session}...")
        alignment = None
        if alignment_per_session and (rat, session) in alignment_per_session:
            alignment = alignment_per_session[(rat, session)]
        try:
            qc = compute_session_qc_summary(
                rat, session, alignment_result=alignment,
                n_sample_frames=n_sample_frames, seed=seed
            )
            results.append(qc)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "rat": rat, "session": session,
                "error": str(e),
            })
    return results


# ===========================================================================
# 5. Calibration epoch analyses
# ===========================================================================
def assign_calibration_epochs(qc_results, calibration_dates):
    """
    Assign each session to a calibration epoch based on calibration_dates.

    Epochs are defined by the most recent calibration date preceding each
    session. Sessions before the first calibration date are assigned epoch 0
    ('pre_<first_cal_date>').

    Parameters
    ----------
    qc_results : list of dicts (must include 'session' key)
    calibration_dates : list of str — date strings in 'YYYY_MM_DD' format

    Returns
    -------
    epochs : list of str — epoch label for each result (same order as input)
    epoch_order : list of str — unique epoch labels in chronological order
    """
    from datetime import datetime

    cal_dts = []
    for cd in sorted(calibration_dates):
        try:
            cal_dts.append((datetime.strptime(cd, "%Y_%m_%d"), cd))
        except ValueError:
            pass

    epochs = []
    for r in qc_results:
        try:
            sess_dt = session_to_datetime(r["session"])
        except Exception:
            epochs.append("unknown")
            continue

        # Find the most recent calibration on or before this session
        epoch_label = f"pre_{cal_dts[0][1]}" if cal_dts else "all"
        for cal_dt, cd in cal_dts:
            if sess_dt >= cal_dt:
                epoch_label = cd
        epochs.append(epoch_label)

    # Build ordered epoch list
    epoch_order = ([f"pre_{cal_dts[0][1]}"] if cal_dts else ["all"]) + [cd for _, cd in cal_dts]

    return epochs, epoch_order


def test_calibration_epoch_effect(qc_results, calibration_dates, outlier_threshold=None):
    """
    Test whether different calibration epochs show significantly different
    overall mean keypoint distances (Kruskal-Wallis test).

    Parameters
    ----------
    qc_results : list of dicts
    calibration_dates : list of str
    outlier_threshold : float or None — exclude sessions above this mean distance

    Returns
    -------
    dict with:
        epoch_data : dict mapping epoch_label -> list of overall_mean values
        epoch_order : list of epoch labels in chronological order
        kruskal_stat : float
        kruskal_pvalue : float
        posthoc : dict of (epoch_a, epoch_b) -> (stat, pvalue) Mann-Whitney U tests
    """
    from scipy.stats import kruskal, mannwhitneyu

    epochs, epoch_order = assign_calibration_epochs(qc_results, calibration_dates)

    epoch_data = {e: [] for e in epoch_order}
    epoch_data["unknown"] = []

    for r, epoch in zip(qc_results, epochs):
        if "error" in r:
            continue
        mean_val = r["summary"]["overall_mean"]
        if outlier_threshold is not None and mean_val > outlier_threshold:
            continue
        if epoch in epoch_data:
            epoch_data[epoch].append(mean_val)

    # Remove empty epochs
    epoch_data = {e: v for e, v in epoch_data.items() if len(v) >= 2}
    epoch_order = [e for e in epoch_order if e in epoch_data]

    groups = [epoch_data[e] for e in epoch_order]
    if len(groups) >= 2:
        stat, pvalue = kruskal(*groups)
    else:
        stat, pvalue = np.nan, np.nan

    # Pairwise Mann-Whitney U post-hoc
    posthoc = {}
    for i, ea in enumerate(epoch_order):
        for eb in epoch_order[i + 1:]:
            if epoch_data[ea] and epoch_data[eb]:
                u, p = mannwhitneyu(epoch_data[ea], epoch_data[eb], alternative="two-sided")
                posthoc[(ea, eb)] = (u, p)

    return {
        "epoch_data": epoch_data,
        "epoch_order": epoch_order,
        "kruskal_stat": stat,
        "kruskal_pvalue": pvalue,
        "posthoc": posthoc,
    }


def compute_days_since_calibration(qc_results, calibration_dates):
    """
    For each session, compute the number of days since the most recent
    preceding calibration date.

    Parameters
    ----------
    qc_results : list of dicts
    calibration_dates : list of str

    Returns
    -------
    days_list : list of float — NaN for sessions before any calibration
    overall_means : list of float — overall_mean for each result
    labels : list of str — session labels
    """
    from datetime import datetime

    cal_dts = sorted([
        datetime.strptime(cd, "%Y_%m_%d")
        for cd in calibration_dates
        if len(cd) == 10  # 'YYYY_MM_DD' is 10 chars
    ])

    days_list, overall_means, labels = [], [], []
    for r in qc_results:
        if "error" in r:
            continue
        try:
            sess_dt = session_to_datetime(r["session"])
        except Exception:
            continue

        # Most recent calibration on or before this session
        preceding = [c for c in cal_dts if c <= sess_dt]
        if not preceding:
            days_since = np.nan
        else:
            days_since = (sess_dt - max(preceding)).days

        days_list.append(days_since)
        overall_means.append(r["summary"]["overall_mean"])
        labels.append(f"{r['rat']}/{r['session']}")

    return days_list, overall_means, labels


def plot_calibration_epoch_effect(epoch_result, rat=None, ax=None):
    """
    Box + strip plot of overall mean distance per calibration epoch.

    Parameters
    ----------
    epoch_result : dict from test_calibration_epoch_effect
    rat : str, optional — for title
    ax : matplotlib Axes, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    epoch_order = epoch_result["epoch_order"]
    epoch_data = epoch_result["epoch_data"]

    positions = np.arange(len(epoch_order))
    for i, ep in enumerate(epoch_order):
        vals = epoch_data[ep]
        # Box
        bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.4),
                        medianprops=dict(color="navy", lw=2),
                        whiskerprops=dict(color="steelblue"),
                        capprops=dict(color="steelblue"),
                        flierprops=dict(marker="", linestyle="none"),
                        manage_ticks=False)
        # Jittered strip
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=30, alpha=0.7,
                   color="steelblue", zorder=3)

    stat = epoch_result["kruskal_stat"]
    pval = epoch_result["kruskal_pvalue"]
    pval_str = f"p={pval:.4f}" if pval >= 0.0001 else "p<0.0001"
    title = f"{'%s — ' % rat if rat else ''}Distance by calibration epoch\n" \
            f"Kruskal-Wallis H={stat:.2f}, {pval_str}"
    ax.set_xticks(positions)
    ax.set_xticklabels(epoch_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Overall mean distance (calibration units)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_days_since_calibration(days_list, overall_means, labels=None,
                                 outlier_threshold=None, rat=None, ax=None):
    """
    Scatter plot of overall mean distance vs. days since last calibration,
    with Spearman correlation.

    Parameters
    ----------
    days_list : list of float
    overall_means : list of float
    labels : list of str, optional
    outlier_threshold : float or None
    rat : str, optional
    ax : matplotlib Axes, optional
    """
    from scipy.stats import spearmanr

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    days = np.array(days_list, dtype=float)
    means = np.array(overall_means, dtype=float)

    # Apply outlier filter
    if outlier_threshold is not None:
        mask = means <= outlier_threshold
    else:
        mask = np.ones(len(means), dtype=bool)

    # Also drop NaN days (sessions before first calibration)
    mask &= ~np.isnan(days)

    d = days[mask]
    m = means[mask]

    ax.scatter(d, m, s=40, alpha=0.7, color="steelblue")

    # Regression line
    if len(d) >= 3:
        coeffs = np.polyfit(d, m, 1)
        x_line = np.linspace(d.min(), d.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r--", alpha=0.7,
                label=f"Linear fit (slope={coeffs[0]:.3f})")

    rho, pval = spearmanr(d, m)
    pval_str = f"p={pval:.4f}" if pval >= 0.0001 else "p<0.0001"
    ax.set_xlabel("Days since last calibration")
    ax.set_ylabel("Overall mean distance (calibration units)")
    title = f"{'%s — ' % rat if rat else ''}Distance vs. time since calibration\n" \
            f"Spearman ρ={rho:.3f}, {pval_str}  (n={len(d)} sessions)"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax, {"spearman_rho": rho, "spearman_pvalue": pval, "n": len(d)}


def plot_snout_error_spatial_density(qc_results, calibration_dates,
                                      snout_idx=0, n_bins=40,
                                      outlier_threshold=None, min_frames_per_bin=5):
    """
    For each calibration epoch, plot two 2D heatmaps:
      - Top row: frame count (occupancy) — how much time was spent in each bin
      - Bottom row: mean Snout keypoint distance in each spatial bin

    Spatial location is determined by the DANNCE centre-of-mass (mean over all
    keypoints) rather than the Snout position, so that high Snout error at a
    particular location does not corrupt the position estimate used for binning.
    Bins with fewer than min_frames_per_bin frames are masked in the error map
    to avoid noisy estimates from poorly-sampled regions.

    Parameters
    ----------
    qc_results : list of dicts
    calibration_dates : list of str
    snout_idx : int — keypoint index for Snout (default 0)
    n_bins : int — histogram bin count along each axis
    outlier_threshold : float or None — exclude sessions with overall_mean above this
    min_frames_per_bin : int — bins with fewer frames are masked in the error map

    Returns
    -------
    fig : matplotlib Figure
    """
    from data_io import load_sleap_dannce_keys, load_aligned_data
    from scipy.ndimage import gaussian_filter

    epochs, epoch_order = assign_calibration_epochs(qc_results, calibration_dates)

    # Collect per-epoch arrays of (com_xy, snout_distance) pairs
    epoch_xy = {e: [] for e in epoch_order}
    epoch_dist = {e: [] for e in epoch_order}

    for r, epoch in zip(qc_results, epochs):
        if "error" in r or epoch not in epoch_xy:
            continue
        if outlier_threshold is not None and r["summary"]["overall_mean"] > outlier_threshold:
            continue

        snout_dist = r["distances"][:, snout_idx]
        valid = ~np.isnan(snout_dist)
        if np.sum(valid) < 10:
            continue

        try:
            keys = load_sleap_dannce_keys(r["rat"], r["session"], fmt="mat")
            dannce_3d = keys["dannce_keys_3D"]
            if dannce_3d.ndim == 4:
                dannce_3d = dannce_3d.squeeze(axis=1).transpose(0, 2, 1)
            else:
                dannce_3d = dannce_3d.transpose(0, 2, 1)
        except Exception as e:
            print(f"  Skipping {r['session']}: {e}")
            continue

        try:
            aligned = load_aligned_data(r["rat"], r["session"], fmt="mat")
            aligned_indices = aligned["dannce_idx_for_sleap_cams"].astype(int).ravel()
        except Exception as e:
            print(f"  Skipping alignment data {r['session']}: {e}")
            continue

        n = min(len(snout_dist), len(aligned_indices))
        valid_idx = np.where(valid[:n] & (aligned_indices[:n] < len(dannce_3d)))[0]
        dannce_idx = aligned_indices[valid_idx]

        # Centre of mass over all keypoints: (n_valid, 2)
        com_xy = dannce_3d[dannce_idx].mean(axis=1)[:, :2]

        epoch_xy[epoch].append(com_xy)
        epoch_dist[epoch].append(snout_dist[valid_idx])

    # Global XY range for consistent axes across epochs
    all_pts = np.concatenate([
        np.concatenate(v) for v in epoch_xy.values() if v
    ], axis=0) if any(epoch_xy.values()) else np.zeros((1, 2))

    x_range = (np.percentile(all_pts[:, 0], 1), np.percentile(all_pts[:, 0], 99))
    y_range = (np.percentile(all_pts[:, 1], 1), np.percentile(all_pts[:, 1], 99))
    bins = [np.linspace(x_range[0], x_range[1], n_bins + 1),
            np.linspace(y_range[0], y_range[1], n_bins + 1)]

    active_epochs = [e for e in epoch_order if epoch_xy.get(e)]
    n_epochs = len(active_epochs)
    if n_epochs == 0:
        print("No data available for spatial density plot.")
        return None

    # Shared color scale for error maps across epochs
    all_mean_errors = []
    for epoch in active_epochs:
        xy = np.concatenate(epoch_xy[epoch], axis=0)
        dist = np.concatenate(epoch_dist[epoch], axis=0)
        xi = np.digitize(xy[:, 0], bins[0]) - 1
        yi = np.digitize(xy[:, 1], bins[1]) - 1
        for bxi in range(n_bins):
            for byi in range(n_bins):
                mask = (xi == bxi) & (yi == byi)
                if mask.sum() >= min_frames_per_bin:
                    all_mean_errors.append(dist[mask].mean())
    vmin = 0
    vmax = np.percentile(all_mean_errors, 95) if all_mean_errors else 1

    # Use a copy of 'hot' with masked (NaN) bins shown in light grey,
    # clearly distinct from both the dark low-error and bright high-error ends
    error_cmap = plt.get_cmap("hot").copy()
    error_cmap.set_bad(color="lightgrey")

    fig, axes = plt.subplots(2, n_epochs,
                              figsize=(4 * n_epochs, 8),
                              sharex=True, sharey=True)
    if n_epochs == 1:
        axes = axes.reshape(2, 1)

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

    for col, epoch in enumerate(active_epochs):
        xy = np.concatenate(epoch_xy[epoch], axis=0)
        dist = np.concatenate(epoch_dist[epoch], axis=0)

        # Bin frame counts
        h_count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins)
        h_count = gaussian_filter(h_count.T, sigma=1.0)

        # Bin mean error: sum of distances / count per bin
        h_dist_sum = np.zeros((n_bins, n_bins))
        h_dist_count = np.zeros((n_bins, n_bins), dtype=int)
        xi = np.clip(np.digitize(xy[:, 0], bins[0]) - 1, 0, n_bins - 1)
        yi = np.clip(np.digitize(xy[:, 1], bins[1]) - 1, 0, n_bins - 1)
        np.add.at(h_dist_sum, (yi, xi), dist)
        np.add.at(h_dist_count, (yi, xi), 1)

        with np.errstate(invalid="ignore"):
            h_mean_error = np.where(h_dist_count >= min_frames_per_bin,
                                    h_dist_sum / h_dist_count, np.nan)

        # Occupancy map
        im0 = axes[0, col].imshow(h_count, origin="lower", extent=extent,
                                   aspect="auto", cmap="Blues")
        axes[0, col].set_title(f"Epoch: {epoch}\nOccupancy by CoM (n={len(xy)} frames)", fontsize=9)
        axes[0, col].set_xlabel("CoM X (calibration units)")
        if col == 0:
            axes[0, col].set_ylabel("CoM Y (calibration units)")
        plt.colorbar(im0, ax=axes[0, col], shrink=0.8, label="Frame count")

        # Mean error map
        im1 = axes[1, col].imshow(h_mean_error, origin="lower", extent=extent,
                                   aspect="auto", cmap=error_cmap,
                                   vmin=vmin, vmax=vmax)
        axes[1, col].set_title(
            f"Mean Snout distance\n(masked <{min_frames_per_bin} frames/bin)", fontsize=9
        )
        axes[1, col].set_xlabel("CoM X (calibration units)")
        if col == 0:
            axes[1, col].set_ylabel("CoM Y (calibration units)")
        plt.colorbar(im1, ax=axes[1, col], shrink=0.8, label="Mean distance (calibration units)")

    fig.suptitle(
        f"Snout keypoint error by spatial location (position = DANNCE centre-of-mass) "
        f"and calibration epoch",
        fontsize=11
    )
    plt.tight_layout()
    return fig


# ===========================================================================
# 5. QC Plots
# ===========================================================================
def _sort_qc_results_chronologically(qc_results):
    """
    Sort a list of qc_result dicts by session date (chronological order).
    Session IDs must follow the 'YYYY_MM_DD_N' format.
    Error entries (missing 'summary') are kept but sorted to the end.
    """
    def _sort_key(r):
        try:
            return session_to_datetime(r["session"])
        except Exception:
            return None

    valid = [r for r in qc_results if _sort_key(r) is not None]
    invalid = [r for r in qc_results if _sort_key(r) is None]
    return sorted(valid, key=_sort_key) + invalid


def plot_per_keypoint_distances(summary, ax=None, title=None):
    """Bar plot of mean per-keypoint distance with std error bars."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    means = summary["per_keypoint_mean"]
    stds = summary["per_keypoint_std"]
    names = summary["keypoint_names"]

    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Distance (calibration units)")
    ax.set_title(title or "SLEAP-DANNCE per-keypoint distance")
    ax.axhline(summary["overall_mean"], color="red", linestyle="--",
               alpha=0.5, label=f"Overall mean: {summary['overall_mean']:.1f}")
    ax.legend()
    return ax


def plot_distance_over_time(distances, keypoint_indices=None,
                            window=100, ax=None, title=None):
    """
    Plot rolling mean distance over frames for selected keypoints.

    Parameters
    ----------
    distances : (n_frames, 23)
    keypoint_indices : list of int, optional — defaults to all
    window : int — rolling window size
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    if keypoint_indices is None:
        keypoint_indices = list(range(N_KEYPOINTS))

    for ki in keypoint_indices:
        d = distances[:, ki]
        valid = ~np.isnan(d)
        if np.sum(valid) < window:
            continue
        # Simple rolling mean
        cumsum = np.nancumsum(d)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        counts = np.convolve(valid.astype(float), np.ones(window), mode="same")
        counts[counts == 0] = 1
        rolling = cumsum / counts
        ax.plot(rolling[:len(d)], alpha=0.5, label=NODES[ki])

    ax.set_xlabel("Frame")
    ax.set_ylabel("Distance (calibration units)")
    ax.set_title(title or "Keypoint distance over time")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    return ax


def plot_multi_session_distances(qc_results, calibration_dates=None, ax=None,
                                  outlier_threshold=None):
    """
    Plot overall mean SLEAP-DANNCE distance across sessions.

    Sessions are sorted chronologically. Optionally filters out sessions
    where overall_mean > outlier_threshold (flagged instead of plotted).

    Parameters
    ----------
    qc_results : list of dicts from track_distances_across_sessions
    calibration_dates : list of str — calibration folder date strings
    outlier_threshold : float or None — if set, sessions with mean distance
        above this value are excluded from the plot and their count is noted
        in the title. E.g. outlier_threshold=50.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Sort chronologically
    sorted_results = _sort_qc_results_chronologically(qc_results)

    sessions = []
    means = []
    medians = []
    n_filtered = 0
    for r in sorted_results:
        if "error" in r:
            continue
        mean_val = r["summary"]["overall_mean"]
        if outlier_threshold is not None and mean_val > outlier_threshold:
            n_filtered += 1
            continue
        sessions.append(f"{r['rat']}/{r['session']}")
        means.append(mean_val)
        medians.append(r["summary"]["overall_median"])

    x = np.arange(len(sessions))
    ax.plot(x, means, "o-", label="Mean distance")
    ax.plot(x, medians, "s--", alpha=0.7, label="Median distance")

    if calibration_dates:
        from datetime import datetime
        # Build list of (session_datetime, x_index) for all plotted sessions
        session_datetimes = []
        for i, s in enumerate(sessions):
            # s is "rat/session", extract the session part
            sess_id = s.split("/")[-1]
            try:
                session_datetimes.append((session_to_datetime(sess_id), i))
            except Exception:
                pass

        # For each calibration date, find the x position just before the first
        # session on or after that calibration date. Place line at x - 0.5.
        seen_positions = set()
        for cd in calibration_dates:
            try:
                cal_dt = datetime.strptime(cd, "%Y_%m_%d")
            except ValueError:
                continue
            # Find first session on or after this calibration date
            first_after = next(
                ((dt, xi) for dt, xi in session_datetimes if dt >= cal_dt),
                None,
            )
            if first_after is None:
                continue
            xpos = first_after[1] - 0.5
            if xpos in seen_positions:
                continue
            seen_positions.add(xpos)
            ax.axvline(xpos, color="red", linestyle=":", alpha=0.7)
            ax.text(xpos, ax.get_ylim()[1], f"Cal: {cd}", rotation=90,
                    va="top", fontsize=7, color="red")

    if outlier_threshold is not None and n_filtered > 0:
        ax.axhline(outlier_threshold, color="orange", linestyle="--", alpha=0.6,
                   label=f"Outlier threshold ({outlier_threshold} units)")

    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Distance (calibration units)")
    title = "SLEAP-DANNCE alignment quality across sessions"
    if n_filtered > 0:
        title += f" ({n_filtered} outlier sessions excluded)"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_temporal_alignment(temporal_stats, ax=None, title=None):
    """Plot temporal alignment quality metrics."""
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = [ax]

    if len(axes) >= 3:
        # Time difference histogram
        diffs = temporal_stats["sleap_dannce_time_diffs_ms"]
        valid_diffs = diffs[~np.isnan(diffs)]
        axes[0].hist(valid_diffs, bins=50, alpha=0.7, color="steelblue")
        axes[0].set_xlabel("Time diff (ms)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"SLEAP-DANNCE time offset\n"
                          f"mean={temporal_stats['mean_time_diff_ms']:.1f} ms, "
                          f"median={temporal_stats['median_time_diff_ms']:.1f} ms")

        # SLEAP frame intervals
        axes[1].hist(temporal_stats["sleap_frame_intervals_ms"], bins=50,
                     alpha=0.7, color="orange")
        axes[1].set_xlabel("Interval (ms)")
        axes[1].set_title("SLEAP frame intervals")

        # DANNCE frame intervals
        axes[2].hist(temporal_stats["dannce_frame_intervals_ms"], bins=50,
                     alpha=0.7, color="green")
        axes[2].set_xlabel("Interval (ms)")
        axes[2].set_title("DANNCE frame intervals")

    plt.suptitle(title or "Temporal alignment quality")
    plt.tight_layout()
    return axes


# ===========================================================================
# 6. Additional QC analyses
# ===========================================================================
def compute_keypoint_jitter(keys_3d, window=5):
    """
    Compute per-keypoint jitter (frame-to-frame displacement variability).
    High jitter may indicate tracking instability.

    Parameters
    ----------
    keys_3d : (n_frames, 23, 3)
    window : int — frames for rolling std

    Returns
    -------
    jitter : (n_frames, 23) — rolling std of frame-to-frame displacement
    """
    displacements = np.linalg.norm(np.diff(keys_3d, axis=0), axis=2)  # (n-1, 23)
    # Pad to match original length
    displacements = np.vstack([displacements[:1], displacements])

    jitter = np.zeros_like(displacements)
    for i in range(window, len(displacements)):
        jitter[i] = np.std(displacements[i - window : i], axis=0)

    return jitter


def detect_tracking_dropouts(keys_3d, max_displacement=100):
    """
    Detect frames where keypoints jump unrealistically far.

    Parameters
    ----------
    keys_3d : (n_frames, 23, 3)
    max_displacement : float — threshold for a single-frame jump

    Returns
    -------
    dropout_frames : (n_frames, 23) bool — True where dropout detected
    """
    displacements = np.linalg.norm(np.diff(keys_3d, axis=0), axis=2)
    dropout = displacements > max_displacement
    return np.vstack([np.zeros((1, N_KEYPOINTS), dtype=bool), dropout])


def compute_bone_length_consistency(keys_3d, edges=EDGES):
    """
    Compute bone lengths over time. Consistent bone lengths indicate
    stable tracking; high variance suggests errors.

    Parameters
    ----------
    keys_3d : (n_frames, 23, 3)

    Returns
    -------
    bone_lengths : (n_frames, n_edges) — per-edge lengths
    bone_stats : dict — mean, std, cv per edge
    """
    n_edges = len(edges)
    bone_lengths = np.zeros((len(keys_3d), n_edges))

    for ei, (i, j) in enumerate(edges):
        bone_lengths[:, ei] = np.linalg.norm(
            keys_3d[:, i, :] - keys_3d[:, j, :], axis=1
        )

    bone_mean = np.mean(bone_lengths, axis=0)
    bone_std = np.std(bone_lengths, axis=0)
    bone_cv = bone_std / (bone_mean + 1e-8)

    edge_labels = [f"{NODES[e[0]]}-{NODES[e[1]]}" for e in edges]

    return bone_lengths, {
        "mean": bone_mean,
        "std": bone_std,
        "cv": bone_cv,
        "edge_labels": edge_labels,
    }


# ===========================================================================
# 7. QC Video generation
# ===========================================================================
def _render_3d_panel(sleap_kp, dannce_kp, edges, panel_h, panel_w,
                     xyz_min, xyz_max, elev=25, azim=45):
    """
    Render a single 3D comparison frame to a numpy image using matplotlib.

    Returns (panel_h, panel_w, 3) uint8 BGR array.
    """
    dpi = 100
    fig_w = panel_w / dpi
    fig_h = panel_h / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # SLEAP (cyan)
    ax.scatter(sleap_kp[:, 0], sleap_kp[:, 1], sleap_kp[:, 2],
               c="cyan", s=20, alpha=0.8, depthshade=True)
    for e in edges:
        ax.plot(sleap_kp[e, 0], sleap_kp[e, 1], sleap_kp[e, 2],
                "c-", lw=1.5, alpha=0.6)

    # DANNCE (magenta)
    ax.scatter(dannce_kp[:, 0], dannce_kp[:, 1], dannce_kp[:, 2],
               c="magenta", s=20, alpha=0.8, depthshade=True)
    for e in edges:
        ax.plot(dannce_kp[e, 0], dannce_kp[e, 1], dannce_kp[e, 2],
                "m-", lw=1.5, alpha=0.6)

    # Connecting lines
    for ki in range(len(sleap_kp)):
        ax.plot([sleap_kp[ki, 0], dannce_kp[ki, 0]],
                [sleap_kp[ki, 1], dannce_kp[ki, 1]],
                [sleap_kp[ki, 2], dannce_kp[ki, 2]],
                "k--", alpha=0.25, lw=0.5)

    ax.set_xlim(xyz_min[0], xyz_max[0])
    ax.set_ylim(xyz_min[1], xyz_max[1])
    ax.set_zlim(xyz_min[2], xyz_max[2])
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("3D Aligned", fontsize=9)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    # Convert to numpy array (compatible with newer matplotlib)
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
    plt.close(fig)

    # Resize to exact panel size and convert to BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img_bgr.shape[0] != panel_h or img_bgr.shape[1] != panel_w:
        img_bgr = cv2.resize(img_bgr, (panel_w, panel_h))
    return img_bgr


def _draw_skeleton_2d(img, keypoints_2d, edges, color, radius=4, thickness=1):
    """Draw keypoints and skeleton edges on an image using OpenCV."""
    pts = keypoints_2d.astype(np.int32)
    for e in edges:
        p1 = tuple(pts[e[0]])
        p2 = tuple(pts[e[1]])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    for i in range(len(pts)):
        cv2.circle(img, tuple(pts[i]), radius, color, -1, cv2.LINE_AA)


def generate_qc_video(rat, session, output_path, alignment_result=None,
                      n_frames=500, start_frame=0, fps=20,
                      camera_idx=1, alignment_n_frames=1000,
                      render_3d=True):
    """
    Generate a QC video using OpenCV for speed.

      Left panel: SLEAP camera video with SLEAP (cyan) and DANNCE (magenta)
                  keypoints overlaid. SLEAP points are projected directly;
                  DANNCE points are inverse-transformed to SLEAP space first.
      Right panel: 3D keypoints in aligned (DANNCE) space (rendered via
                   matplotlib per-frame). Set render_3d=False to skip this
                   panel for maximum speed.

    Parameters
    ----------
    rat : str
    session : str
    output_path : str — output .mp4 path
    alignment_result : dict — from find_sleap_dannce_alignment
    n_frames : int — number of frames to render
    start_frame : int — starting SLEAP frame
    fps : int — output video fps
    camera_idx : int — which SLEAP camera (default 1 = Camera1)
    alignment_n_frames : int — frames for Procrustes fit (default 1000)
    render_3d : bool — if True, include 3D comparison panel (slower).
                       If False, only render the 2D overlay (much faster).
    """
    import os
    from data_io import load_sleap_dannce_keys, load_aligned_data
    from projection import project_3d_to_2d_for_camera
    from config import sleap_path, calibration_path as get_cal_path

    print(f"Loading data for {rat}/{session}...")
    keys = load_sleap_dannce_keys(rat, session)
    aligned = load_aligned_data(rat, session)

    sleap_3d = keys["sleap_keys_3D"]
    dannce_3d = keys["dannce_keys_3D"]
    if dannce_3d.ndim == 4:
        dannce_3d = dannce_3d.squeeze(axis=1).transpose(0, 2, 1)
    else:
        dannce_3d = dannce_3d.transpose(0, 2, 1)

    sleap_3d = median_filter(keys['sleap_keys_3D'], size=(11, 1, 1))
    #dannce_3d = median_filter(keys['dannce_keys_3D'], size=(5, 1, 1))

    dannce_3d = median_filter(dannce_3d, size=(25, 1, 1))

    aligned_indices = aligned["dannce_idx_for_sleap_cams"].astype(int).ravel()

    # Compute alignment if not provided
    if alignment_result is None:
        print(f"Computing alignment with {alignment_n_frames} frames...")
        alignment_result = find_sleap_dannce_alignment(
            sleap_3d, dannce_3d, aligned_indices,
            n_sample_frames=alignment_n_frames,
        )
    print(f"Alignment residual: {alignment_result['residual']:.2f}")

    # For 3D panel: SLEAP in DANNCE space
    sleap_aligned = alignment_result["apply"](sleap_3d)

    # Open source video
    sp = sleap_path(rat, session)
    cam_name = f"Camera{camera_idx}"
    video_file = os.path.join(sp, cam_name, "0.mp4")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_file}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cal_folder = get_cal_path(rat, session)

    end_frame = min(start_frame + n_frames, len(sleap_3d), len(aligned_indices))
    total_render = end_frame - start_frame

    # --- Pre-compute all 2D projections ---
    print("Projecting SLEAP keypoints to 2D...")
    sleap_2d = project_3d_to_2d_for_camera(
        sleap_3d[start_frame:end_frame], cal_folder, camera_idx=camera_idx
    )

    print("Transforming DANNCE to SLEAP space and projecting to 2D...")
    dannce_frames = []
    for i in range(start_frame, end_frame):
        di = aligned_indices[i] if i < len(aligned_indices) else 0
        if di < len(dannce_3d):
            dannce_frames.append(dannce_3d[di])
        else:
            dannce_frames.append(np.zeros((N_KEYPOINTS, 3)))
    dannce_frames_arr = np.array(dannce_frames)
    dannce_in_sleap = alignment_result["apply_inverse"](dannce_frames_arr)
    dannce_2d = project_3d_to_2d_for_camera(
        dannce_in_sleap, cal_folder, camera_idx=camera_idx
    )

    # --- Compute 3D axis limits for right panel ---
    if render_3d:
        sample_s = sleap_aligned[start_frame:end_frame]
        di_sample = aligned_indices[start_frame:end_frame]
        valid_di = di_sample[di_sample < len(dannce_3d)]
        sample_d = dannce_3d[valid_di] if len(valid_di) > 0 else sample_s
        all_pts = np.concatenate([sample_s.reshape(-1, 3), sample_d.reshape(-1, 3)])
        xyz_min = np.percentile(all_pts, 2, axis=0) - 20
        xyz_max = np.percentile(all_pts, 98, axis=0) + 20

    # --- Set up output video ---
    if render_3d:
        out_w = vid_w * 2
    else:
        out_w = vid_w
    out_h = vid_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_path}")

    # Colors (BGR for OpenCV)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)

    print(f"Rendering {total_render} frames ({'with' if render_3d else 'without'} 3D panel)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for fi in range(total_render):
        frame_idx = start_frame + fi

        # Read video frame
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

        # Draw SLEAP skeleton (cyan)
        _draw_skeleton_2d(frame, sleap_2d[fi], EDGES, CYAN, radius=4, thickness=1)

        # Draw DANNCE skeleton (magenta)
        _draw_skeleton_2d(frame, dannce_2d[fi], EDGES, MAGENTA, radius=4, thickness=1)

        # Add text overlay
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "SLEAP", (10, vid_h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "DANNCE", (10, vid_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, MAGENTA, 2, cv2.LINE_AA)

        if render_3d:
            # Render 3D panel
            s3d = sleap_aligned[frame_idx]
            di = aligned_indices[frame_idx] if frame_idx < len(aligned_indices) else 0
            d3d = dannce_3d[di] if di < len(dannce_3d) else s3d

            panel_3d = _render_3d_panel(
                s3d, d3d, EDGES, vid_h, vid_w, xyz_min, xyz_max
            )

            # Concatenate side by side
            combined = np.hstack([frame, panel_3d])
        else:
            combined = frame

        writer.write(combined)

        if (fi + 1) % 1000 == 0 or fi == total_render - 1:
            print(f"  {fi + 1}/{total_render} frames rendered")

    writer.release()
    cap.release()
    print(f"Saved to {output_path}")
