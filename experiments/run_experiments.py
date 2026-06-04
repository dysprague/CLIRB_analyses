"""
Template matching alignment experiment runner.
Session-centric: compute features once per session, run all experiments.

Usage:
    python run_experiments.py --rat R1 --config primary [--max_sessions N]
"""
import argparse
import numpy as np
import pandas as pd
import sys, time, traceback
from pathlib import Path
from collections import deque
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp_utils import (
    load_session_data, smooth_keypoints,
    compute_pairwise_distances, compute_velocity,
    run_template_matching, compute_alignment_multi_tol,
    estimate_temporal_offset, SLEAP_HZ, DANNCE_HZ
)
from data_io import load_template, get_sessions, load_sleap_dannce_keys
from skeleton import normalize_skeleton_batch, project_to_pcs
from config import NODE_IDX

RESULTS_DIR = ROOT / 'results' / 'metrics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = ROOT / 'results' / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RAT_CONFIG = {
    'R1': {
        'primary':   {'template_file': 'R1_template_2.npz', 'bounds': 1.0},
        'secondary': {'template_file': 'R1_template_1.npz', 'bounds': 1.5},
    },
    'R2': {'primary': {'template_file': 'R2_template_1.npz', 'bounds': 1.0}},
    'R3': {'primary': {'template_file': 'R3_template_1.npz', 'bounds': 1.0}},
}

TOLERANCES_MS = [100, 300, 500]
REFRACTORY_FRAMES = 30
SPINE_KPTS = ['Snout', 'EarL', 'EarR', 'SpineF', 'SpineM', 'SpineL', 'TailBase']

# Fixed smoothing configs for SLEAP and DANNCE (default)
# Both at 20 Hz (DANNCE is resampled to SLEAP frame rate in load_session_data)
SL_SMOOTH = ('median', 11)
DN_SMOOTH = ('median', 11)

# For preprocessing sweep experiments
SMOOTH_SWEEP = [
    ('median', 3), ('median', 5), ('median', 9), ('median', 11), ('median', 15),
    ('savgol', 5), ('savgol', 9),
    ('ema', 5), ('ema', 10),
]
MAX_OUT_SWEEP = [0, 1, 2, 3, 4, 5]

# ─────────────────────────────────────────────────────────────────────────────
# Template loading
# ─────────────────────────────────────────────────────────────────────────────

def load_template_info(rat, template_file):
    t = dict(load_template(rat, template_file))
    has_origin = (
        'temp_origin_file' in t and
        t['temp_origin_file'] is not None and
        str(t['temp_origin_file']).strip() not in ('', 'None', 'nan', '0')
    )
    return t, has_origin


def smooth_stored_template(t, window=7):
    from scipy.signal import savgol_filter
    tmpl = t['template'].copy()
    w = min(window, tmpl.shape[0]) | 1
    if w >= 3:
        for j in range(tmpl.shape[1]):
            tmpl[:, j] = savgol_filter(tmpl[:, j], w, 1)
    return tmpl


def extract_dannce_template_window(rat, t, smooth_method='median', smooth_window=11):
    origin_path = str(t['temp_origin_file'])
    origin_idx = int(t['temp_origin_idx'])
    session = origin_path.split('/')[-1]
    d = load_sleap_dannce_keys(rat, session)
    dn = d['dannce_keys_3D'].astype(np.float64)
    if dn.ndim == 3 and dn.shape[1] == 3:
        dn = dn.transpose(0, 2, 1)
    dn[:, :, 2] = -dn[:, :, 2]  # z-flip to match convention... actually DANNCE doesn't need flip
    # Revert: only SLEAP needs z-flip
    dn[:, :, 2] = -dn[:, :, 2]  # undo
    dn_sm = smooth_keypoints(dn, smooth_method, smooth_window)
    dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
    win = t['template'].shape[0]
    start = max(0, origin_idx - win)
    chunk = dn_rot[start:origin_idx]
    if len(chunk) < win:
        chunk = np.pad(chunk, ((win - len(chunk), 0), (0, 0), (0, 0)), mode='edge')
    return chunk  # (win, 23, 3) egocentric


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _combined_features(rot):
    dist = compute_pairwise_distances(rot)
    z = rot[:, :, 2] * 0.1
    vel = compute_velocity(rot) * 0.5
    return np.hstack([dist, z, vel])


def _normalize_subset(keys, SpineF_i, SpineM_i):
    SpineF = keys[:, SpineF_i, :]
    SpineM = keys[:, SpineM_i, :]
    angle = np.arctan2(-(SpineF[:, 1] - SpineM[:, 1]), SpineF[:, 0] - SpineM[:, 0])
    c, s = np.cos(angle), np.sin(angle)
    centered = keys - SpineM[:, None, :]
    out = centered.copy()
    out[:, :, 0] = c[:, None] * centered[:, :, 0] - s[:, None] * centered[:, :, 1]
    out[:, :, 1] = s[:, None] * centered[:, :, 0] + c[:, None] * centered[:, :, 1]
    return out


def get_offset(sl_feat, dn_feat, tmpl, bounds, sleap_times_ms):
    """Estimate temporal offset using quick preliminary match.
    Both SLEAP and DANNCE match indices now reference sleap_times_ms
    (DANNCE has been resampled to SLEAP frame rate).
    """
    if sleap_times_ms is None:
        return 0.0
    gt_m = run_template_matching(dn_feat, tmpl, bounds, max_outside=3)
    if len(gt_m) < 2:
        return 0.0
    sl_m = run_template_matching(sl_feat, tmpl, bounds, max_outside=3)
    if len(sl_m) < 2:
        return 0.0
    return estimate_temporal_offset(sl_m, gt_m, sleap_times_ms, sleap_times_ms)


def match_and_score(sl_feat, dn_feat, tmpl, bounds, max_out,
                    gt_matches, sleap_times_ms, offset_ms):
    sl_m = run_template_matching(sl_feat, tmpl, bounds, max_outside=max_out)
    dn_m = run_template_matching(dn_feat, tmpl, bounds, max_outside=max_out)
    al_sl = compute_alignment_multi_tol(sl_m, gt_matches, TOLERANCES_MS,
                                         sleap_times_ms, sleap_times_ms, offset_ms)
    al_dn = compute_alignment_multi_tol(dn_m, gt_matches, TOLERANCES_MS,
                                         sleap_times_ms, sleap_times_ms, 0.0)
    return sl_m, dn_m, al_sl, al_dn


def build_bounds(btype, tmpl, stds, scalar, sl_feat=None, dn_feat=None):
    win, n_pcs = tmpl.shape
    s = stds[:n_pcs] if len(stds) >= n_pcs else np.ones(n_pcs)
    if btype in ('uniform', 'pc_std'):
        return np.tile(s * scalar, (win, 1))
    elif btype == 'adaptive':
        return scalar * (0.25 * s[None, :] + 0.75 * np.abs(tmpl))
    elif btype == 'percentile':
        if sl_feat is not None and dn_feat is not None:
            min_len = min(len(sl_feat), len(dn_feat))
            step = max(1, min_len // 5000)
            n = min(len(sl_feat[::step]), len(dn_feat[::step]))
            diff = np.abs(sl_feat[:min_len:step][:n, :n_pcs] -
                          dn_feat[:min_len:step][:n, :n_pcs])
            p95 = np.percentile(diff, 95, axis=0)
            return np.tile(p95 * scalar, (win, 1))
        return np.tile(s * scalar, (win, 1))
    return np.tile(s * scalar, (win, 1))


def make_row(rat, session, group, exp_name, smooth_method, smooth_window,
             bounds_type, bounds_scalar, max_outside,
             gt_matches, sl_matches, dn_matches, al_sl, al_dn, offset_ms,
             online_compat=True, **extra):
    row = dict(
        rat=rat, session=session, group=group, exp_name=exp_name,
        smooth_method=smooth_method, smooth_window=smooth_window,
        bounds_type=bounds_type, bounds_scalar=bounds_scalar,
        max_outside=max_outside, online_compat=online_compat,
        n_gt=len(gt_matches), n_sleap=len(sl_matches), n_dannce=len(dn_matches),
        temporal_offset_ms=round(float(offset_ms), 1),
    )
    row.update(extra)
    for tol in TOLERANCES_MS:
        k = f'tol_{tol}ms'
        for src, al in [('sl', al_sl), ('dn', al_dn)]:
            row[f'{src}_recall_{tol}']    = round(al[k]['recall'], 4)
            row[f'{src}_precision_{tol}'] = round(al[k]['precision'], 4)
            row[f'{src}_f1_{tol}']        = round(al[k]['f1'], 4)
            row[f'{src}_n_both_{tol}']    = al[k]['n_both']
    return row


def mse_matching(feat, tmpl, threshold):
    win = tmpl.shape[0]
    buf = deque(maxlen=win)
    matches, last = [], -REFRACTORY_FRAMES
    for i, f in enumerate(feat):
        buf.append(f)
        if len(buf) < win:
            continue
        mse = np.mean((np.array(buf) - tmpl) ** 2)
        if mse <= threshold and (i - last) >= REFRACTORY_FRAMES:
            matches.append(i)
            last = i
    return matches


def corr_matching(feat, tmpl, threshold):
    win = tmpl.shape[0]
    buf = deque(maxlen=win)
    b = tmpl.ravel()
    matches, last = [], -REFRACTORY_FRAMES
    for i, f in enumerate(feat):
        buf.append(f)
        if len(buf) < win:
            continue
        a = np.array(buf).ravel()
        corr = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0
        if corr >= threshold and (i - last) >= REFRACTORY_FRAMES:
            matches.append(i)
            last = i
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Global PCA fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_all_pcas(rat, df, subsample=20, n_components=10):
    spine_idx = [NODE_IDX[k] for k in SPINE_KPTS]
    SpineF_i_sp = SPINE_KPTS.index('SpineF')
    SpineM_i_sp = SPINE_KPTS.index('SpineM')
    pool_pw, pool_zv, pool_sp_xyz, pool_sp_pw = [], [], [], []
    for _, row in df.iterrows():
        try:
            _, dn, _ = load_session_data(rat, row['session'])
            dn_sm = smooth_keypoints(dn[::subsample], 'median', 11)
            dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
            pool_pw.append(compute_pairwise_distances(dn_rot))
            pool_zv.append(_combined_features(dn_rot))
            sp = dn_sm[:, spine_idx, :]
            sp_rot = _normalize_subset(sp, SpineF_i_sp, SpineM_i_sp)
            pool_sp_xyz.append(sp_rot.reshape(sp_rot.shape[0], -1))
            pool_sp_pw.append(compute_pairwise_distances(sp_rot))
        except Exception:
            pass
    pcas = {}
    for name, pool in [('pairwise', pool_pw), ('pairwise_z_vel', pool_zv),
                        ('spine_xyz', pool_sp_xyz), ('spine_pairwise', pool_sp_pw)]:
        data = np.concatenate(pool)
        pcas[name] = PCA(n_components=min(n_components, data.shape[1])).fit(data)
    return pcas


# ─────────────────────────────────────────────────────────────────────────────
# Template building
# ─────────────────────────────────────────────────────────────────────────────

def build_templates(t, has_origin, pcas, dannce_tmpl_window):
    pcs_to_use = t['pcs_to_use'].ravel().astype(int)
    win = t['template'].shape[0]
    spine_idx = [NODE_IDX[k] for k in SPINE_KPTS]
    SpineF_i_sp = SPINE_KPTS.index('SpineF')
    SpineM_i_sp = SPINE_KPTS.index('SpineM')

    tmpls = {'xyz': t['template'][:, pcs_to_use]}

    if has_origin and dannce_tmpl_window is not None:
        pw_raw = compute_pairwise_distances(dannce_tmpl_window)
        tmpls['pairwise'] = pcas['pairwise'].transform(pw_raw)
        tmpls['pairwise_z_vel'] = pcas['pairwise_z_vel'].transform(
            _combined_features(dannce_tmpl_window))
        sp = dannce_tmpl_window[:, spine_idx, :]
        sp_rot = _normalize_subset(sp, SpineF_i_sp, SpineM_i_sp)
        tmpls['spine_xyz'] = pcas['spine_xyz'].transform(sp_rot.reshape(win, -1))
        tmpls['spine_pairwise'] = pcas['spine_pairwise'].transform(
            compute_pairwise_distances(sp_rot))
    else:
        for k, pca in [('pairwise', pcas['pairwise']), ('pairwise_z_vel', pcas['pairwise_z_vel']),
                        ('spine_xyz', pcas['spine_xyz']), ('spine_pairwise', pcas['spine_pairwise'])]:
            tmpls[k] = np.zeros((win, pca.n_components))

    return tmpls


# ─────────────────────────────────────────────────────────────────────────────
# Session processing (main)
# ─────────────────────────────────────────────────────────────────────────────

def process_session(rat, session, t, has_origin, pcas, tmpls, bounds_scalar):
    """
    Compute features for one session and run all experiments.
    Returns list of result rows.
    """
    pcs_to_use = t['pcs_to_use'].ravel().astype(int)
    xyz_stds = t['feature_stds'][pcs_to_use]
    spine_idx = [NODE_IDX[k] for k in SPINE_KPTS]
    SpineF_i_sp = SPINE_KPTS.index('SpineF')
    SpineM_i_sp = SPINE_KPTS.index('SpineM')

    try:
        sleap_3d, dannce_3d, aligned = load_session_data(rat, session)
    except Exception as e:
        return [{'rat': rat, 'session': session, 'error': f'load: {e}'}]

    # Both SLEAP and DANNCE match indices now reference sleap_times_ms
    # (DANNCE was resampled to SLEAP frame rate in load_session_data)
    st = np.array(aligned['sleap_times_ms']).ravel() if aligned else None
    dt = st  # same time axis for both

    # ── Pre-compute features at default smoothing ─────────────────────────
    try:
        sl_sm = smooth_keypoints(sleap_3d, SL_SMOOTH[0], SL_SMOOTH[1])
        dn_sm = smooth_keypoints(dannce_3d, DN_SMOOTH[0], DN_SMOOTH[1])
        sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
        dn_rot, _, _ = normalize_skeleton_batch(dn_sm)

        # XYZ PCA
        sl_xyz = project_to_pcs(sl_rot, t['pc_weights'], t['feature_means'])[:, pcs_to_use]
        dn_xyz = project_to_pcs(dn_rot, t['pc_weights'], t['feature_means'])[:, pcs_to_use]

        # Pairwise distances
        sl_pw_raw = compute_pairwise_distances(sl_rot)
        dn_pw_raw = compute_pairwise_distances(dn_rot)
        sl_pw = pcas['pairwise'].transform(sl_pw_raw)
        dn_pw = pcas['pairwise'].transform(dn_pw_raw)

        # Pairwise + Z + Vel
        sl_zv = pcas['pairwise_z_vel'].transform(_combined_features(sl_rot))
        dn_zv = pcas['pairwise_z_vel'].transform(_combined_features(dn_rot))

        # Spine
        sl_sp = sl_sm[:, spine_idx, :]
        dn_sp = dn_sm[:, spine_idx, :]
        sl_sp_rot = _normalize_subset(sl_sp, SpineF_i_sp, SpineM_i_sp)
        dn_sp_rot = _normalize_subset(dn_sp, SpineF_i_sp, SpineM_i_sp)
        sl_sp_xyz = pcas['spine_xyz'].transform(sl_sp_rot.reshape(sl_sp_rot.shape[0], -1))
        dn_sp_xyz = pcas['spine_xyz'].transform(dn_sp_rot.reshape(dn_sp_rot.shape[0], -1))
        sl_sp_pw = pcas['spine_pairwise'].transform(compute_pairwise_distances(sl_sp_rot))
        dn_sp_pw = pcas['spine_pairwise'].transform(compute_pairwise_distances(dn_sp_rot))
    except Exception as e:
        return [{'rat': rat, 'session': session, 'error': f'features: {e}'}]

    rows = []

    # ═══════════════════════════════════════════════════════════════════════
    # A. XYZ PCA — bounds type × max_outside sweep (fixed smoothing)
    # ═══════════════════════════════════════════════════════════════════════
    tmpl_xyz = tmpls['xyz']
    gt_bounds_xyz = build_bounds('uniform', tmpl_xyz, xyz_stds, bounds_scalar)
    gt_m_xyz = run_template_matching(dn_xyz, tmpl_xyz, gt_bounds_xyz, max_outside=3)
    if gt_m_xyz:
        off_xyz = get_offset(sl_xyz, dn_xyz, tmpl_xyz, gt_bounds_xyz, st)
        dn_xyz_stds = np.std(dn_xyz, axis=0)

        for btype, scalar in [
            ('uniform', bounds_scalar),
            ('adaptive', bounds_scalar * 0.75),
            ('adaptive', bounds_scalar),
            ('adaptive', bounds_scalar * 1.25),
            ('percentile', 1.0),
            ('percentile', 1.5),
        ]:
            b = build_bounds(btype, tmpl_xyz, xyz_stds, scalar, sl_xyz, dn_xyz)
            for mo in MAX_OUT_SWEEP:
                sl_m, dn_m, al_sl, al_dn = match_and_score(
                    sl_xyz, dn_xyz, tmpl_xyz, b, mo, gt_m_xyz, st, off_xyz)
                rows.append(make_row(rat, session, 'A_xyz', f'A_xyz_{btype}',
                                     SL_SMOOTH[0], SL_SMOOTH[1], btype, scalar, mo,
                                     gt_m_xyz, sl_m, dn_m, al_sl, al_dn, off_xyz))

        # A2: MSE matching
        from scipy.signal import savgol_filter as _sg
        mse_vals = []
        for m in gt_m_xyz:
            s0 = max(0, m - 30)
            chunk = dn_xyz[s0:m]
            if len(chunk) == 30:
                mse_vals.append(np.mean((chunk - tmpl_xyz)**2))
        if mse_vals:
            for pct in [75, 90, 95]:
                thresh = np.percentile(mse_vals, pct) * 1.5
                sl_m = mse_matching(sl_xyz, tmpl_xyz, thresh)
                dn_m = mse_matching(dn_xyz, tmpl_xyz, thresh)
                al_sl = compute_alignment_multi_tol(sl_m, gt_m_xyz, TOLERANCES_MS, st, st, off_xyz)
                al_dn = compute_alignment_multi_tol(dn_m, gt_m_xyz, TOLERANCES_MS, st, st, 0.0)
                rows.append(make_row(rat, session, 'A_mse', f'A_mse_pct{pct}',
                                     SL_SMOOTH[0], SL_SMOOTH[1], 'mse', bounds_scalar, 0,
                                     gt_m_xyz, sl_m, dn_m, al_sl, al_dn, off_xyz))

        # A3: Correlation matching
        b_flat = tmpl_xyz.ravel()
        corr_vals = []
        for m in gt_m_xyz:
            s0 = max(0, m - 30)
            chunk = dn_xyz[s0:m]
            if len(chunk) == 30:
                a = chunk.ravel()
                if np.std(a) > 0:
                    corr_vals.append(np.corrcoef(a, b_flat)[0, 1])
        if corr_vals:
            thresh = np.percentile(corr_vals, 10) * 0.9
            sl_m = corr_matching(sl_xyz, tmpl_xyz, thresh)
            dn_m = corr_matching(dn_xyz, tmpl_xyz, thresh)
            al_sl = compute_alignment_multi_tol(sl_m, gt_m_xyz, TOLERANCES_MS, st, st, off_xyz)
            al_dn = compute_alignment_multi_tol(dn_m, gt_m_xyz, TOLERANCES_MS, st, st, 0.0)
            rows.append(make_row(rat, session, 'A_corr', 'A_correlation',
                                 SL_SMOOTH[0], SL_SMOOTH[1], 'correlation', bounds_scalar, 0,
                                 gt_m_xyz, sl_m, dn_m, al_sl, al_dn, off_xyz))

    # ═══════════════════════════════════════════════════════════════════════
    # B. XYZ PCA — smoothing sweep (fixed bounds/max_outside)
    # ═══════════════════════════════════════════════════════════════════════
    for method, window in SMOOTH_SWEEP:
        try:
            sl_s = smooth_keypoints(sleap_3d, method, window)
            dn_s = smooth_keypoints(dannce_3d, method, window)
            sl_r, _, _ = normalize_skeleton_batch(sl_s)
            dn_r, _, _ = normalize_skeleton_batch(dn_s)
            sl_f = project_to_pcs(sl_r, t['pc_weights'], t['feature_means'])[:, pcs_to_use]
            dn_f = project_to_pcs(dn_r, t['pc_weights'], t['feature_means'])[:, pcs_to_use]

            gt_b = build_bounds('uniform', tmpl_xyz, xyz_stds, bounds_scalar)
            gt_m = run_template_matching(dn_f, tmpl_xyz, gt_b, max_outside=3)
            if not gt_m:
                continue
            off = get_offset(sl_f, dn_f, tmpl_xyz, gt_b, st)
            for mo in [0, 1, 2, 3, 4, 5]:
                sl_m = run_template_matching(sl_f, tmpl_xyz, gt_b, max_outside=mo)
                dn_m = run_template_matching(dn_f, tmpl_xyz, gt_b, max_outside=mo)
                al_sl = compute_alignment_multi_tol(sl_m, gt_m, TOLERANCES_MS, st, st, off)
                al_dn = compute_alignment_multi_tol(dn_m, gt_m, TOLERANCES_MS, st, st, 0.0)
                rows.append(make_row(rat, session, 'B_smooth',
                                     f'B_smooth_{method}{window}',
                                     method, window, 'uniform', bounds_scalar, mo,
                                     gt_m, sl_m, dn_m, al_sl, al_dn, off))
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # C. Pairwise distances → DANNCE PCA (n_pcs sweep)
    # ═══════════════════════════════════════════════════════════════════════
    for n_pcs in [1, 2, 3, 4, 6, 8, 10]:
        tmpl_pw = tmpls['pairwise'][:, :n_pcs]
        sl_f = sl_pw[:, :n_pcs]
        dn_f = dn_pw[:, :n_pcs]
        dn_stds = pcas['pairwise'].explained_variance_[:n_pcs] ** 0.5

        gt_b = build_bounds('pc_std', tmpl_pw, dn_stds, bounds_scalar)
        gt_m = run_template_matching(dn_f, tmpl_pw, gt_b, max_outside=3)
        if not gt_m:
            continue
        off = get_offset(sl_f, dn_f, tmpl_pw, gt_b, st)

        for btype, scalar in [
            ('pc_std', bounds_scalar),
            ('adaptive', bounds_scalar),
            ('percentile', 1.0),
            ('percentile', 1.5),
        ]:
            b = build_bounds(btype, tmpl_pw, dn_stds, scalar, sl_f, dn_f)
            for mo in MAX_OUT_SWEEP:
                sl_m, dn_m, al_sl, al_dn = match_and_score(
                    sl_f, dn_f, tmpl_pw, b, mo, gt_m, st, off)
                rows.append(make_row(rat, session, 'C_pairwise',
                                     f'C_pairwise_n{n_pcs}_{btype}',
                                     SL_SMOOTH[0], SL_SMOOTH[1], btype, scalar, mo,
                                     gt_m, sl_m, dn_m, al_sl, al_dn, off,
                                     n_pcs=n_pcs))

    # ═══════════════════════════════════════════════════════════════════════
    # D. Pairwise + Z + Velocity
    # ═══════════════════════════════════════════════════════════════════════
    for n_pcs in [2, 4, 6]:
        tmpl_zv = tmpls['pairwise_z_vel'][:, :n_pcs]
        sl_f = sl_zv[:, :n_pcs]
        dn_f = dn_zv[:, :n_pcs]
        dn_stds = pcas['pairwise_z_vel'].explained_variance_[:n_pcs] ** 0.5

        gt_b = build_bounds('pc_std', tmpl_zv, dn_stds, bounds_scalar)
        gt_m = run_template_matching(dn_f, tmpl_zv, gt_b, max_outside=3)
        if not gt_m:
            continue
        off = get_offset(sl_f, dn_f, tmpl_zv, gt_b, st)
        b = gt_b
        for mo in MAX_OUT_SWEEP:
            sl_m, dn_m, al_sl, al_dn = match_and_score(
                sl_f, dn_f, tmpl_zv, b, mo, gt_m, st, off)
            rows.append(make_row(rat, session, 'D_zvel',
                                 f'D_zvel_n{n_pcs}',
                                 SL_SMOOTH[0], SL_SMOOTH[1], 'pc_std', bounds_scalar, mo,
                                 gt_m, sl_m, dn_m, al_sl, al_dn, off,
                                 n_pcs=n_pcs))

    # ═══════════════════════════════════════════════════════════════════════
    # E. Spine-only features
    # ═══════════════════════════════════════════════════════════════════════
    for feat_name, sl_f_all, dn_f_all, tmpl_key, group in [
        ('spine_xyz', sl_sp_xyz, dn_sp_xyz, 'spine_xyz', 'E_spine_xyz'),
        ('spine_pairwise', sl_sp_pw, dn_sp_pw, 'spine_pairwise', 'E_spine_pw'),
    ]:
        for n_pcs in [2, 4, 6]:
            tmpl_sp = tmpls[tmpl_key][:, :n_pcs]
            sl_f = sl_f_all[:, :n_pcs]
            dn_f = dn_f_all[:, :n_pcs]
            dn_stds = pcas[tmpl_key].explained_variance_[:n_pcs] ** 0.5

            gt_b = build_bounds('pc_std', tmpl_sp, dn_stds, bounds_scalar)
            gt_m = run_template_matching(dn_f, tmpl_sp, gt_b, max_outside=3)
            if not gt_m:
                continue
            off = get_offset(sl_f, dn_f, tmpl_sp, gt_b, st)
            for mo in MAX_OUT_SWEEP:
                sl_m, dn_m, al_sl, al_dn = match_and_score(
                    sl_f, dn_f, tmpl_sp, gt_b, mo, gt_m, st, off)
                rows.append(make_row(rat, session, group, f'{group}_n{n_pcs}',
                                     SL_SMOOTH[0], SL_SMOOTH[1], 'pc_std', bounds_scalar, mo,
                                     gt_m, sl_m, dn_m, al_sl, al_dn, off,
                                     n_pcs=n_pcs))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_rat(rat, config_key='primary', max_sessions=None):
    cfg = RAT_CONFIG[rat][config_key]
    template_file = cfg['template_file']
    bounds_scalar = cfg['bounds']

    print(f"\n{'='*60}")
    print(f"RAT={rat}  config={config_key}  template={template_file}  bounds={bounds_scalar}")
    print(f"{'='*60}")

    t, has_origin = load_template_info(rat, template_file)
    print(f"has_origin: {has_origin}")

    df = get_sessions(rat=rat)
    if max_sessions:
        df = df.head(max_sessions)
    sessions = df['session'].tolist()
    print(f"Sessions: {len(sessions)}")

    print("Fitting PCAs...")
    t0 = time.time()
    pcas = fit_all_pcas(rat, df)
    print(f"  {time.time()-t0:.1f}s")

    print("Building templates...")
    dannce_tmpl_window = None
    if has_origin:
        try:
            dannce_tmpl_window = extract_dannce_template_window(rat, t)
            print(f"  DANNCE window: {dannce_tmpl_window.shape}")
        except Exception as e:
            print(f"  [warn] {e}")
            has_origin = False
    tmpls = build_templates(t, has_origin, pcas, dannce_tmpl_window)

    all_rows = []
    log_lines = []

    for i, session in enumerate(sessions):
        t0 = time.time()
        try:
            rows = process_session(rat, session, t, has_origin, pcas, tmpls, bounds_scalar)
            ok = [r for r in rows if 'error' not in r]
            err = [r for r in rows if 'error' in r]
            all_rows.extend(ok)
            for r in err:
                log_lines.append(f"ERR {session}: {r.get('error','')[:120]}")
            print(f"  [{i+1:3d}/{len(sessions)}] {session}  {len(ok):5d} rows  {time.time()-t0:.1f}s")
        except Exception as e:
            log_lines.append(f"FATAL {session}: {e}")
            print(f"  [{i+1:3d}/{len(sessions)}] {session}  FATAL {e}")

        # Save checkpoint every 10 sessions
        if (i + 1) % 10 == 0:
            pd.DataFrame(all_rows).to_csv(
                RESULTS_DIR / f'{rat}_{config_key}_checkpoint.csv', index=False)

    out_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / f'{rat}_{config_key}_results.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows → {out_path}")

    with open(LOGS_DIR / f'{rat}_{config_key}.log', 'w') as f:
        f.write('\n'.join(log_lines))

    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rat', default='R1', choices=['R1', 'R2', 'R3'])
    parser.add_argument('--config', default='primary', choices=['primary', 'secondary'])
    parser.add_argument('--max_sessions', type=int, default=None)
    args = parser.parse_args()
    run_rat(args.rat, args.config, max_sessions=args.max_sessions)
