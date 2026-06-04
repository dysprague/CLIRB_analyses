"""
v4 — Track B sweeps that v1–v3 didn't cover.

Goal: address structured bias between SLEAP and DANNCE in PC space, optimizing
F1 (precision and recall equally), not recall alone.

Approach groups (same row format as v3 so the existing results notebook reads it):

    M_procrustes      Procrustes-fit-on-calibration-epoch SLEAP→DANNCE pre-alignment.
                      Sweep n_calibration_minutes × bounds_scalar × max_outside.

    N_joint_means     Recenter PCA on (SLEAP + DANNCE)/2 mean per session, then
                      project both. Sweep bounds_scalar × max_outside × distance_metric.

    O_pairwise_pooled Replace 69D xyz with 253D pairwise distances; fit a fresh
                      PCA on pooled (SLEAP+DANNCE) features; rebuild template in
                      that space from the SAME source frames as the original
                      template. Sweep n_pcs × bounds_scalar.

    P_kp_exclude      Drop the K worst-aligned keypoints (ranked from this
                      session's per-keypoint SLEAP-DANNCE distance) before
                      egocentric normalization. Sweep K = 0..6.

    Q_distance_metric Use the original template/PCs but switch the matching
                      metric: cosine (shape-only), correlation, MSE — at a
                      threshold percentile sweep.

Each row reports recall AND precision AND F1 at 100/300/500 ms tolerances
for both SLEAP-vs-GT and DANNCE-vs-GT (matches v3 schema).

Usage:
    python run_experiments_v4.py --rat R1 --config secondary
    python run_experiments_v4.py --rat all
"""
import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from data_io import (
    get_sessions,
    load_aligned_data,
    load_sleap_dannce_keys,
    load_template,
)
from exp_utils import (
    SLEAP_HZ,
    compute_alignment_multi_tol,
    compute_pairwise_distances,
    estimate_temporal_offset,
    load_session_data,
    run_template_matching,
    smooth_keypoints,
)
from skeleton import normalize_skeleton_batch, project_to_pcs
from qc_utils import find_sleap_dannce_alignment
from config import NODES, NODE_IDX

RESULTS_DIR = ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAT_CONFIG = {
    "R1": {
        "primary":   {"template_file": "R1_template_2.npz", "bounds": 1.0},
        "secondary": {"template_file": "R1_template_1.npz", "bounds": 1.5},
    },
    "R2": {"primary": {"template_file": "R2_template_1.npz", "bounds": 1.0}},
    "R3": {"primary": {"template_file": "R3_template_1.npz", "bounds": 1.0}},
}

TOLERANCES_MS = [100, 300, 500]
SL_SMOOTH = ("median", 11)
DN_SMOOTH = ("median", 11)
WIN = 30  # template window length, in SLEAP frames

# Sweep grids
BOUNDS_SCALAR_SWEEP = [0.75, 1.0, 1.25, 1.5]
MAX_OUT_SWEEP = [0, 1, 2, 3]
REFRAC = 30
CAL_MIN_SWEEP = [2.0, 5.0]                              # M: minutes for Procrustes calibration
DISTANCE_METRICS = ["uniform", "cosine", "correlation"]  # N, Q
N_PCS_SWEEP = [2, 3, 4, 6]                              # O
KP_EXCLUDE_SWEEP = [0, 2, 4, 6]                         # P
COSINE_PCT_SWEEP = [25, 50, 75, 90]                     # Q


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized matching primitives
# ─────────────────────────────────────────────────────────────────────────────

def _windows(feat, win):
    T = feat.shape[0]
    n = T - win + 1
    if n <= 0:
        return np.zeros((0, win, feat.shape[1]))
    shape = (n, win, feat.shape[1])
    strides = (feat.strides[0], feat.strides[0], feat.strides[1])
    return as_strided(feat, shape=shape, strides=strides)


def _apply_refractory(candidate_window_idx, win, refractory):
    if len(candidate_window_idx) == 0:
        return []
    frames = candidate_window_idx + win - 1
    matches = [int(frames[0])]
    for c in frames[1:]:
        if c - matches[-1] >= refractory:
            matches.append(int(c))
    return matches


def cosine_match(feat, tmpl, threshold, refractory=REFRAC):
    """Match where cosine similarity (flat windows vs flat template) >= threshold."""
    win = tmpl.shape[0]
    W = _windows(feat, win)
    if W.shape[0] == 0:
        return []
    a = W.reshape(W.shape[0], -1)        # (n, win*n_pcs)
    b = tmpl.ravel()                     # (win*n_pcs,)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b)
    denom = a_norm * b_norm + 1e-12
    cos = (a @ b) / denom
    cands = np.where(cos >= threshold)[0]
    return _apply_refractory(cands, win, refractory)


def corr_match(feat, tmpl, threshold, refractory=REFRAC):
    """Pearson correlation match."""
    win = tmpl.shape[0]
    W = _windows(feat, win)
    if W.shape[0] == 0:
        return []
    a = W.reshape(W.shape[0], -1)
    b = tmpl.ravel()
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean()
    num = a_c @ b_c
    denom = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum()) + 1e-12
    corr = num / denom
    cands = np.where(corr >= threshold)[0]
    return _apply_refractory(cands, win, refractory)


def mse_at_gt(feat, tmpl, gt_frames):
    """Per-event MSE using END-of-window convention. Returns 1D array."""
    out = []
    for f in gt_frames:
        s = f - tmpl.shape[0] + 1
        if s < 0 or f >= feat.shape[0]:
            continue
        out.append(np.mean((feat[s:f + 1] - tmpl) ** 2))
    return np.asarray(out)


def cosine_at_gt(feat, tmpl, gt_frames):
    out = []
    b = tmpl.ravel()
    bn = np.linalg.norm(b)
    for f in gt_frames:
        s = f - tmpl.shape[0] + 1
        if s < 0 or f >= feat.shape[0]:
            continue
        a = feat[s:f + 1].ravel()
        out.append(float(a @ b / (np.linalg.norm(a) * bn + 1e-12)))
    return np.asarray(out)


# ─────────────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_row(rat, session, group, exp_name, gt_matches, sl_matches, dn_matches,
             al_sl, al_dn, offset_ms, **extra):
    row = dict(
        rat=rat, session=session, group=group, exp_name=exp_name,
        n_gt=len(gt_matches), n_sleap=len(sl_matches), n_dannce=len(dn_matches),
        temporal_offset_ms=round(float(offset_ms), 1),
    )
    row.update(extra)
    for tol in TOLERANCES_MS:
        key = f"tol_{tol}ms"
        for which, al in [("sl", al_sl), ("dn", al_dn)]:
            r = al.get(key, {})
            row[f"{which}_recall_{tol}"] = round(r.get("recall", 0.0), 4)
            row[f"{which}_precision_{tol}"] = round(r.get("precision", 0.0), 4)
            row[f"{which}_f1_{tol}"] = round(r.get("f1", 0.0), 4)
            row[f"{which}_n_both_{tol}"] = r.get("n_both", 0)
    return row


def score(sl_m, dn_m, gt_m, st, offset_ms):
    al_sl = compute_alignment_multi_tol(sl_m, gt_m, TOLERANCES_MS, st, st, offset_ms)
    al_dn = compute_alignment_multi_tol(dn_m, gt_m, TOLERANCES_MS, st, st, 0.0)
    return al_sl, al_dn


def get_gt_and_offset(sl_pc, dn_pc, tmpl, xyz_stds, bounds_scalar, st):
    bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
    gt_m = run_template_matching(dn_pc, tmpl, bounds, max_outside=3,
                                 refractory_frames=REFRAC)
    sl_init = run_template_matching(sl_pc, tmpl, bounds, max_outside=3,
                                    refractory_frames=REFRAC)
    if len(sl_init) >= 2 and len(gt_m) >= 2 and st is not None:
        offset_ms = estimate_temporal_offset(sl_init, gt_m, st, st)
    else:
        offset_ms = 0.0
    return gt_m, offset_ms


# ─────────────────────────────────────────────────────────────────────────────
# Group M — Procrustes pre-alignment, fit on calibration epoch
# ─────────────────────────────────────────────────────────────────────────────

def _calibration_window(sleap_3d, dannce_3d, aligned, calibration_minutes):
    """Return the first `calibration_minutes` of frames as (sleap, dannce) arrays."""
    if aligned is None:
        return sleap_3d, dannce_3d
    times = np.array(aligned["sleap_times_ms"]).ravel()
    if len(times) < 10:
        return sleap_3d, dannce_3d
    cutoff_ms = times[0] + calibration_minutes * 60_000
    n_use = int(np.searchsorted(times, cutoff_ms))
    n_use = max(min(n_use, len(sleap_3d), len(dannce_3d)),
                int(0.5 * SLEAP_HZ * 60))  # at least 30 s
    return sleap_3d[:n_use], dannce_3d[:n_use]


def fit_procrustes(sleap_3d_cal, dannce_3d_cal):
    """
    Returns the apply() callable that maps SLEAP 3D → DANNCE space.
    Skips z-flip (we're using exp_utils z-flipped SLEAP throughout).
    """
    n = min(len(sleap_3d_cal), len(dannce_3d_cal))
    aligned_idx = np.arange(n, dtype=int)
    return find_sleap_dannce_alignment(
        sleap_3d_cal[:n], dannce_3d_cal[:n], aligned_idx,
        n_sample_frames=min(500, n), seed=42, try_z_flip=False)


def run_M_procrustes(rat, session, t, bounds_scalar, sleap_3d, dannce_3d,
                     aligned, st, sl_sm, dn_sm):
    pcu = t["pcs_to_use"].ravel().astype(int)
    pw = t["pc_weights"]
    fm = t["feature_means"]
    xyz_stds = t["feature_stds"][pcu]
    tmpl = t["template"][:, pcu]

    rows = []
    for cal_min in CAL_MIN_SWEEP:
        sl_cal, dn_cal = _calibration_window(sl_sm, dn_sm, aligned, cal_min)
        if len(sl_cal) < 30 or len(dn_cal) < 30:
            continue
        try:
            align = fit_procrustes(sl_cal, dn_cal)
        except Exception as e:
            print(f"    M procrustes fit failed (cal={cal_min}): {e}")
            continue

        sl_aligned = align["apply"](sl_sm)               # in DANNCE space now
        sl_rot, _, _ = normalize_skeleton_batch(sl_aligned)
        dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
        sl_pc = project_to_pcs(sl_rot, pw, fm)[:, pcu]
        dn_pc = project_to_pcs(dn_rot, pw, fm)[:, pcu]

        gt_m, offset_ms = get_gt_and_offset(sl_pc, dn_pc, tmpl, xyz_stds,
                                            bounds_scalar, st)
        if not gt_m:
            continue

        for scalar in BOUNDS_SCALAR_SWEEP:
            bounds = np.tile(xyz_stds * scalar, (WIN, 1))
            for mo in MAX_OUT_SWEEP:
                sl_m = run_template_matching(sl_pc, tmpl, bounds, max_outside=mo,
                                             refractory_frames=REFRAC)
                dn_m = run_template_matching(dn_pc, tmpl, bounds, max_outside=mo,
                                             refractory_frames=REFRAC)
                al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
                rows.append(make_row(
                    rat, session, "M_procrustes",
                    f"M_proc_cal{cal_min:.0f}_s{scalar:.2f}_mo{mo}",
                    gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                    procrustes_residual=round(float(align["residual"]), 3),
                    procrustes_scale=round(float(align["s"]), 4),
                    cal_minutes=cal_min, bounds_scalar=scalar, max_outside=mo,
                    refractory=REFRAC,
                ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Group N — Joint feature means (recenter PCA on pooled SLEAP+DANNCE mean)
# ─────────────────────────────────────────────────────────────────────────────

def run_N_joint_means(rat, session, t, bounds_scalar, st, sl_sm, dn_sm):
    pcu = t["pcs_to_use"].ravel().astype(int)
    pw = t["pc_weights"]
    fm = t["feature_means"]
    xyz_stds = t["feature_stds"][pcu]
    tmpl_pc = t["template"][:, pcu]                 # template in original PC space

    sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
    dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
    sl_flat = sl_rot.reshape(sl_rot.shape[0], -1)
    dn_flat = dn_rot.reshape(dn_rot.shape[0], -1)

    # Joint mean computed once, applied to both
    fm_joint = (sl_flat.mean(axis=0) + dn_flat.mean(axis=0)) / 2.0
    delta = fm_joint - fm                                       # shift in feature space
    # Project both with the new mean, original weights
    sl_pc = ((sl_flat - fm_joint) @ pw.T)[:, pcu]
    dn_pc = ((dn_flat - fm_joint) @ pw.T)[:, pcu]

    # Template shifts by -delta projected onto same PCs
    tmpl_shift = (delta @ pw.T)[pcu]                            # (n_pcs,)
    tmpl = tmpl_pc + tmpl_shift[None, :]

    gt_m, offset_ms = get_gt_and_offset(sl_pc, dn_pc, tmpl, xyz_stds,
                                        bounds_scalar, st)
    if not gt_m:
        return []

    rows = []
    for scalar in BOUNDS_SCALAR_SWEEP:
        bounds = np.tile(xyz_stds * scalar, (WIN, 1))
        for mo in MAX_OUT_SWEEP:
            sl_m = run_template_matching(sl_pc, tmpl, bounds, max_outside=mo,
                                         refractory_frames=REFRAC)
            dn_m = run_template_matching(dn_pc, tmpl, bounds, max_outside=mo,
                                         refractory_frames=REFRAC)
            al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
            rows.append(make_row(
                rat, session, "N_joint_means",
                f"N_jm_uniform_s{scalar:.2f}_mo{mo}",
                gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                bounds_scalar=scalar, max_outside=mo, refractory=REFRAC,
            ))

    # Cosine and correlation variants under joint means (calibrated thresholds)
    cos_at_gt = cosine_at_gt(dn_pc, tmpl, gt_m)
    corr_thresholds = []
    if cos_at_gt.size:
        for pct in COSINE_PCT_SWEEP:
            corr_thresholds.append(("cosine", pct, np.percentile(cos_at_gt, pct)))
    for metric, pct, thr in corr_thresholds:
        sl_m = cosine_match(sl_pc, tmpl, thr)
        dn_m = cosine_match(dn_pc, tmpl, thr)
        al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
        rows.append(make_row(
            rat, session, "N_joint_means_metric",
            f"N_jm_cos_p{pct}",
            gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
            metric=metric, threshold_pct=pct, threshold=round(float(thr), 4),
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Group O — Pairwise distance features with pooled (SLEAP+DANNCE) PCA
# ─────────────────────────────────────────────────────────────────────────────

def _load_template_window_3d(t):
    """
    Reload the 3D keypoint window that produced this template, so we can
    re-derive the template in a different feature space.

    Templates store either:
      - temp_origin_file = "Rx/YYYY_MM_DD_N"  + temp_origin_idx (frame index)
      - or no metadata at all (older templates).

    Returns the (WIN, 23, 3) SLEAP window post z-flip + median filter, or None
    if metadata is unavailable.
    """
    if "temp_origin_file" not in t:
        return None
    try:
        origin_str = str(t["temp_origin_file"]).strip().strip("'\"")
        if "/" not in origin_str:
            return None
        rat, session = origin_str.split("/", 1)
    except Exception:
        return None
    # Frame index lives under one of these keys depending on template version
    frame = None
    for k in ("temp_origin_idx", "temp_origin_frame"):
        if k in t:
            try:
                frame = int(np.asarray(t[k]).ravel()[0])
                break
            except Exception:
                continue
    if frame is None:
        return None
    try:
        keys = load_sleap_dannce_keys(rat, session)
        sl = keys["sleap_keys_3D"]
        from scipy.ndimage import median_filter
        sl = median_filter(sl, size=(SL_SMOOTH[1], 1, 1)).astype(np.float64)
        sl[:, :, 2] = -sl[:, :, 2]
        s, e = frame + 1, frame + 1 + WIN
        win = sl[s:e]
        if len(win) < WIN:
            return None
        win_rot, _, _ = normalize_skeleton_batch(win)
        return win_rot
    except Exception:
        return None


def _per_session_template_window(sl_sm_rot, dn_sm_rot, t, xyz_stds,
                                 bounds_scalar):
    """
    Fallback when template origin metadata is unavailable: locate this session's
    best-matching 30-frame window using the original PC-space template, and
    return that window in 3D (post normalization-aware feature extraction).

    Returns (window_normalized, found_frame) — window has shape (WIN, 23, 3)
    in the egocentric-rotated coordinate system (sl_sm_rot/dn_sm_rot input),
    or (None, None) if no GT match is found.
    """
    pcu = t["pcs_to_use"].ravel().astype(int)
    pw = t["pc_weights"]
    fm = t["feature_means"]
    tmpl_pc = t["template"][:, pcu]

    dn_flat = dn_sm_rot.reshape(dn_sm_rot.shape[0], -1)
    dn_pc = ((dn_flat - fm) @ pw.T)[:, pcu]
    bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
    gt = run_template_matching(dn_pc, tmpl_pc, bounds, max_outside=3,
                               refractory_frames=REFRAC)
    if not gt:
        return None, None
    # Pick the GT frame whose DANNCE window is *closest* (lowest MSE) to the template
    best_frame, best_err = None, np.inf
    for f in gt:
        s = f - WIN + 1
        if s < 0:
            continue
        err = np.mean((dn_pc[s:f + 1] - tmpl_pc) ** 2)
        if err < best_err:
            best_err = err
            best_frame = f
    if best_frame is None:
        return None, None
    s = best_frame - WIN + 1
    return dn_sm_rot[s:best_frame + 1], best_frame


def run_O_pairwise_pooled(rat, session, t, bounds_scalar, st, sl_sm, dn_sm):
    """
    Fit a fresh PCA on pooled (SLEAP+DANNCE) pairwise distance features.
    Re-derive the template from the same source frames in this new space.
    """
    pcu = t["pcs_to_use"].ravel().astype(int)
    xyz_stds = t["feature_stds"][pcu]

    sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
    dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
    sl_pw = compute_pairwise_distances(sl_rot)         # (T, 253)
    dn_pw = compute_pairwise_distances(dn_rot)
    pooled = np.vstack([sl_pw, dn_pw])
    pw_mean = pooled.mean(axis=0)
    centered = pooled - pw_mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components_full = Vt                              # (n_feat, n_feat)

    # Rebuild template: prefer original origin frames; else use this session's
    # best-matching DANNCE window as a per-session template
    win_rot = _load_template_window_3d(t)
    template_source = "origin"
    if win_rot is None:
        win_rot, _ = _per_session_template_window(sl_rot, dn_rot, t,
                                                  xyz_stds, bounds_scalar)
        template_source = "session_best_dannce"
    if win_rot is None:
        return [{"rat": rat, "session": session,
                 "error": "O: no template window (origin missing, no GT match)"}]
    win_pw = compute_pairwise_distances(win_rot)
    win_pc_full = (win_pw - pw_mean) @ components_full.T   # (WIN, n_feat)

    rows = []
    for n_pcs in N_PCS_SWEEP:
        comps = components_full[:n_pcs]                     # (n_pcs, n_feat)
        sl_pc = (sl_pw - pw_mean) @ comps.T                 # (T, n_pcs)
        dn_pc = (dn_pw - pw_mean) @ comps.T
        tmpl = win_pc_full[:, :n_pcs]
        # Use empirical std on this session as bounds reference
        feat_stds = np.std(np.vstack([sl_pc, dn_pc]), axis=0)

        gt_m, offset_ms = get_gt_and_offset(sl_pc, dn_pc, tmpl, feat_stds,
                                            bounds_scalar, st)
        if not gt_m:
            continue

        for scalar in BOUNDS_SCALAR_SWEEP:
            bounds = np.tile(feat_stds * scalar, (WIN, 1))
            for mo in MAX_OUT_SWEEP:
                sl_m = run_template_matching(sl_pc, tmpl, bounds, max_outside=mo,
                                             refractory_frames=REFRAC)
                dn_m = run_template_matching(dn_pc, tmpl, bounds, max_outside=mo,
                                             refractory_frames=REFRAC)
                al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
                rows.append(make_row(
                    rat, session, "O_pairwise_pooled",
                    f"O_pw_n{n_pcs}_s{scalar:.2f}_mo{mo}",
                    gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                    n_pcs=n_pcs, bounds_scalar=scalar, max_outside=mo,
                    refractory=REFRAC, template_source=template_source,
                ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Group P — Drop the worst-aligned keypoints, then re-PCA in original space
# ─────────────────────────────────────────────────────────────────────────────

def run_P_kp_exclude(rat, session, t, bounds_scalar, st, sl_sm, dn_sm):
    """
    Rank keypoints by SLEAP-vs-DANNCE mean distance after egocentric
    normalization, drop the K worst, refit PCA on pooled features, rebuild
    template in this restricted space.
    """
    pcu = t["pcs_to_use"].ravel().astype(int)
    xyz_stds = t["feature_stds"][pcu]

    sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
    dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
    kp_err = np.linalg.norm(sl_rot - dn_rot, axis=2).mean(axis=0)   # (23,)
    order = np.argsort(kp_err)[::-1]                                 # worst first

    win_rot = _load_template_window_3d(t)
    template_source = "origin"
    if win_rot is None:
        win_rot, _ = _per_session_template_window(sl_rot, dn_rot, t,
                                                  xyz_stds, bounds_scalar)
        template_source = "session_best_dannce"
    if win_rot is None:
        return [{"rat": rat, "session": session,
                 "error": "P: no template window available"}]

    rows = []
    for K in KP_EXCLUDE_SWEEP:
        keep = np.array([i for i in range(len(NODES)) if i not in order[:K]])
        sl_kept = sl_rot[:, keep, :].reshape(sl_rot.shape[0], -1)
        dn_kept = dn_rot[:, keep, :].reshape(dn_rot.shape[0], -1)
        win_kept = win_rot[:, keep, :].reshape(win_rot.shape[0], -1)

        pooled = np.vstack([sl_kept, dn_kept])
        mu = pooled.mean(axis=0)
        centered = pooled - mu
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        n_pcs = 4
        comps = Vt[:n_pcs]
        sl_pc = (sl_kept - mu) @ comps.T
        dn_pc = (dn_kept - mu) @ comps.T
        tmpl = (win_kept - mu) @ comps.T            # (WIN, n_pcs)
        feat_stds = np.std(np.vstack([sl_pc, dn_pc]), axis=0)

        gt_m, offset_ms = get_gt_and_offset(sl_pc, dn_pc, tmpl, feat_stds,
                                            bounds_scalar, st)
        if not gt_m:
            continue

        for scalar in BOUNDS_SCALAR_SWEEP:
            bounds = np.tile(feat_stds * scalar, (WIN, 1))
            for mo in MAX_OUT_SWEEP:
                sl_m = run_template_matching(sl_pc, tmpl, bounds, max_outside=mo,
                                             refractory_frames=REFRAC)
                dn_m = run_template_matching(dn_pc, tmpl, bounds, max_outside=mo,
                                             refractory_frames=REFRAC)
                al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
                rows.append(make_row(
                    rat, session, "P_kp_exclude",
                    f"P_K{K}_s{scalar:.2f}_mo{mo}",
                    gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                    K_excluded=K, bounds_scalar=scalar, max_outside=mo,
                    excluded_kps=";".join(NODES[i] for i in order[:K]),
                    template_source=template_source,
                ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Group Q — Distance-metric sweep on the original feature space
# ─────────────────────────────────────────────────────────────────────────────

def run_Q_distance_metric(rat, session, t, bounds_scalar, st, sl_sm, dn_sm):
    pcu = t["pcs_to_use"].ravel().astype(int)
    pw = t["pc_weights"]
    fm = t["feature_means"]
    xyz_stds = t["feature_stds"][pcu]
    tmpl = t["template"][:, pcu]

    sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
    dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
    sl_pc = project_to_pcs(sl_rot, pw, fm)[:, pcu]
    dn_pc = project_to_pcs(dn_rot, pw, fm)[:, pcu]

    gt_m, offset_ms = get_gt_and_offset(sl_pc, dn_pc, tmpl, xyz_stds,
                                        bounds_scalar, st)
    if not gt_m:
        return []

    rows = []
    # Calibrate thresholds on per-session DANNCE GT events
    cos_vals = cosine_at_gt(dn_pc, tmpl, gt_m)
    if cos_vals.size:
        for pct in COSINE_PCT_SWEEP:
            thr = float(np.percentile(cos_vals, pct))
            sl_m = cosine_match(sl_pc, tmpl, thr)
            dn_m = cosine_match(dn_pc, tmpl, thr)
            al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
            rows.append(make_row(
                rat, session, "Q_cosine",
                f"Q_cos_p{pct}",
                gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                metric="cosine", threshold_pct=pct, threshold=round(thr, 4),
            ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-session driver
# ─────────────────────────────────────────────────────────────────────────────

def process_session(rat, session, t, bounds_scalar, groups):
    try:
        sleap_3d, dannce_3d, aligned = load_session_data(rat, session)
    except Exception as e:
        return [{"rat": rat, "session": session, "error": f"load: {e}"}]

    st = np.array(aligned["sleap_times_ms"]).ravel() if aligned else None

    sl_sm = smooth_keypoints(sleap_3d, *SL_SMOOTH)
    dn_sm = smooth_keypoints(dannce_3d, *DN_SMOOTH)

    rows = []
    if "M" in groups:
        try: rows.extend(run_M_procrustes(rat, session, t, bounds_scalar,
                                          sleap_3d, dannce_3d, aligned, st,
                                          sl_sm, dn_sm))
        except Exception as e:
            traceback.print_exc(); rows.append({"rat": rat, "session": session,
                                                 "error": f"M: {e}"})
    if "N" in groups:
        try: rows.extend(run_N_joint_means(rat, session, t, bounds_scalar, st,
                                           sl_sm, dn_sm))
        except Exception as e:
            traceback.print_exc(); rows.append({"rat": rat, "session": session,
                                                 "error": f"N: {e}"})
    if "O" in groups:
        try: rows.extend(run_O_pairwise_pooled(rat, session, t, bounds_scalar,
                                               st, sl_sm, dn_sm))
        except Exception as e:
            traceback.print_exc(); rows.append({"rat": rat, "session": session,
                                                 "error": f"O: {e}"})
    if "P" in groups:
        try: rows.extend(run_P_kp_exclude(rat, session, t, bounds_scalar, st,
                                          sl_sm, dn_sm))
        except Exception as e:
            traceback.print_exc(); rows.append({"rat": rat, "session": session,
                                                 "error": f"P: {e}"})
    if "Q" in groups:
        try: rows.extend(run_Q_distance_metric(rat, session, t, bounds_scalar,
                                               st, sl_sm, dn_sm))
        except Exception as e:
            traceback.print_exc(); rows.append({"rat": rat, "session": session,
                                                 "error": f"Q: {e}"})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main with checkpointing (matches v3 pattern)
# ─────────────────────────────────────────────────────────────────────────────

def run_rat(rat, config_key="primary", max_sessions=None,
            groups=("M", "N", "O", "P", "Q")):
    cfg = RAT_CONFIG[rat][config_key]
    t = dict(load_template(rat, cfg["template_file"]))
    bounds_scalar = cfg["bounds"]

    sessions = get_sessions(rat=rat)["session"].tolist()
    if max_sessions:
        sessions = sessions[:max_sessions]

    out_path = RESULTS_DIR / f"{rat}_{config_key}_v4_results.csv"
    ckpt_path = RESULTS_DIR / f"{rat}_{config_key}_v4_checkpoint.csv"

    done = set()
    if ckpt_path.exists():
        prev = pd.read_csv(ckpt_path)
        done = set(prev["session"].unique())
        print(f"Resuming: {len(done)} sessions already processed.")

    todo = [s for s in sessions if s not in done]
    print(f"{rat} {config_key}: groups={groups}, "
          f"{len(todo)} sessions to run, {len(done)} done")

    t_start = time.time()
    for i, session in enumerate(todo):
        t1 = time.time()
        try:
            rows = process_session(rat, session, t, bounds_scalar, groups)
        except Exception as e:
            traceback.print_exc()
            rows = [{"rat": rat, "session": session, "error": str(e)}]
        valid = [r for r in rows if "error" not in r]
        invalid = [r for r in rows if "error" in r]
        print(f"  [{i+1}/{len(todo)}] {session}: "
              f"{len(valid)} rows ({len(invalid)} errors) "
              f"in {time.time()-t1:.1f}s")
        if valid:
            chunk = pd.DataFrame(valid)
            if ckpt_path.exists():
                chunk.to_csv(ckpt_path, mode="a", header=False, index=False)
            else:
                chunk.to_csv(ckpt_path, index=False)

    if ckpt_path.exists():
        final = pd.read_csv(ckpt_path)
        final.to_csv(out_path, index=False)
        print(f"\nDone. {len(final)} rows → {out_path}")
        # Quick summary
        if "sl_f1_300" in final.columns:
            good = final[final["temporal_offset_ms"] > 100]
            if len(good):
                summary = (good.groupby(["group", "exp_name"])
                           [["sl_f1_300", "sl_recall_300", "sl_precision_300",
                             "dn_f1_300"]]
                           .mean()
                           .sort_values("sl_f1_300", ascending=False)
                           .head(10))
                print("\nTop 10 by SLEAP F1 @300ms (good-offset sessions):")
                print(summary.round(3).to_string())
    print(f"Total: {time.time()-t_start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat", required=True,
                        choices=["R1", "R2", "R3", "all"])
    parser.add_argument("--config", default="primary",
                        choices=["primary", "secondary"])
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--groups", default="MNOPQ",
                        help="Subset of groups to run, e.g. 'MN' or 'OQ'")
    args = parser.parse_args()
    groups = tuple(args.groups)

    if args.rat == "all":
        for rat, cfg in [("R1", "primary"), ("R1", "secondary"),
                         ("R2", "primary"), ("R3", "primary")]:
            run_rat(rat, cfg, args.max_sessions, groups)
    else:
        run_rat(args.rat, args.config, args.max_sessions, groups)


if __name__ == "__main__":
    main()
