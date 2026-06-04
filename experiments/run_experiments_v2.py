"""
Targeted follow-up experiments for template matching (v2).

New approaches:
F. SLEAP-calibrated template: build template from SLEAP data at DANNCE event times
   (using alignment index) — avoids SLEAP/DANNCE PC-space mismatch
G. Refractory period sweep: 10, 15, 20, 25, 30 frames at default SLEAP refractory
H. Cross-session MSE calibration: pool MSE values from held-out sessions

Usage:
    python run_experiments_v2.py --rat R1 --config primary
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
    estimate_temporal_offset, compute_alignment, SLEAP_HZ, DANNCE_HZ
)
from data_io import load_template, get_sessions, load_sleap_dannce_keys, load_aligned_data
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
SL_SMOOTH = ('median', 11)
DN_SMOOTH = ('median', 11)
REFRACTORY_SWEEP = [5, 10, 15, 20, 25, 30]
WIN = 30  # template window length in SLEAP frames


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_template_info(rat, template_file):
    t = dict(load_template(rat, template_file))
    has_origin = (
        'temp_origin_file' in t and
        t['temp_origin_file'] is not None and
        str(t['temp_origin_file']).strip() not in ('', 'None', 'nan', '0')
    )
    return t, has_origin


def get_good_sessions(rat):
    """Return sessions where temporal offset is positive (system delay properly estimated)."""
    f = RESULTS_DIR / f'{rat}_primary_checkpoint.csv'
    if not f.exists():
        f = RESULTS_DIR / f'{rat}_primary_results.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    mse = df[df.exp_name == 'A_mse_pct90']
    good = mse[mse.temporal_offset_ms > 100]['session'].unique()
    return list(good)


def mse_match_refractory(feat, tmpl, threshold, refractory):
    """MSE-based matching with variable refractory period. Vectorized with stride tricks."""
    from numpy.lib.stride_tricks import as_strided
    T, n_pcs = feat.shape
    win = tmpl.shape[0]
    if T < win:
        return []
    # Build sliding window view (zero-copy)
    n_windows = T - win + 1
    shape = (n_windows, win, n_pcs)
    strides = (feat.strides[0], feat.strides[0], feat.strides[1])
    windows = as_strided(feat, shape=shape, strides=strides)
    # MSE per window
    mse = np.mean((windows - tmpl[None, :, :]) ** 2, axis=(1, 2))
    candidates = np.where(mse <= threshold)[0] + win - 1
    if len(candidates) == 0:
        return []
    matches = [candidates[0]]
    for c in candidates[1:]:
        if c - matches[-1] >= refractory:
            matches.append(c)
    return matches


def corr_match_refractory(feat, tmpl, threshold, refractory):
    """Correlation-based matching with variable refractory period. Vectorized."""
    from numpy.lib.stride_tricks import as_strided
    T, n_pcs = feat.shape
    win = tmpl.shape[0]
    if T < win:
        return []
    n_windows = T - win + 1
    shape = (n_windows, win, n_pcs)
    strides = (feat.strides[0], feat.strides[0], feat.strides[1])
    windows = as_strided(feat, shape=shape, strides=strides)
    # Flatten for correlation computation
    b = tmpl.ravel()  # (win*n_pcs,)
    a = windows.reshape(n_windows, -1)  # (n_windows, win*n_pcs)
    # Vectorized Pearson correlation
    a_mean = a.mean(axis=1, keepdims=True)
    b_mean = b.mean()
    a_c = a - a_mean
    b_c = b - b_mean
    num = (a_c * b_c).sum(axis=1)
    denom = np.sqrt((a_c**2).sum(axis=1) * (b_c**2).sum())
    corr = np.where(denom > 0, num / denom, 0.0)
    candidates = np.where(corr >= threshold)[0] + win - 1
    if len(candidates) == 0:
        return []
    matches = [candidates[0]]
    for c in candidates[1:]:
        if c - matches[-1] >= refractory:
            matches.append(c)
    return matches


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
        temporal_offset_ms=offset_ms,
    )
    for tol in TOLERANCES_MS:
        key = f'tol_{tol}ms'
        for which, al in [('sl', al_sl), ('dn', al_dn)]:
            r = al.get(key, {})
            row[f'{which}_recall_{tol}'] = r.get('recall', 0.0)
            row[f'{which}_precision_{tol}'] = r.get('precision', 0.0)
            row[f'{which}_f1_{tol}'] = r.get('f1', 0.0)
            row[f'{which}_n_both_{tol}'] = r.get('n_both', 0)
    row.update(extra)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Build SLEAP-calibrated template
# ─────────────────────────────────────────────────────────────────────────────

def build_sleap_calibrated_template(rat, t, calibration_sessions, bounds_scalar):
    """
    Build a SLEAP-based template by extracting SLEAP keypoints at DANNCE event times
    (using the alignment index) across calibration sessions.

    Returns:
        sl_tmpl  : (WIN, n_pcs) — mean SLEAP PC trajectory at event times
        sl_mse_thresh : float — MSE threshold from calibration data
        sl_corr_thresh : float — correlation threshold from calibration data
        calibration_data : list of (win, n_pcs) windows from calibration sessions
    """
    pw = t['pc_weights']
    fm = t['feature_means']
    pcu = t['pcs_to_use'].ravel().astype(int)
    xyz_stds = t['feature_stds'][pcu]
    tmpl_stored = t['template'][:, pcu]
    from scipy.signal import savgol_filter
    win_size = min(7, tmpl_stored.shape[0]) | 1
    tmpl_smooth = tmpl_stored.copy()
    if win_size >= 3:
        for j in range(tmpl_smooth.shape[1]):
            tmpl_smooth[:, j] = savgol_filter(tmpl_smooth[:, j], win_size, 1)

    # Step 1: Use DANNCE to find GT events, then find corresponding SLEAP windows
    event_sl_windows = []  # SLEAP PC windows at true events
    event_dn_windows = []  # DANNCE PC windows at true events

    for sess in calibration_sessions:
        try:
            sleap_3d, dannce_3d, aligned = load_session_data(rat, sess)
            if aligned is None:
                continue
            sl_t = np.array(aligned['sleap_times_ms']).ravel()
            dn_t = np.array(aligned['sleap_times_ms']).ravel()
            si = np.array(aligned.get('sleap_idx_for_dannce_cams', [])).ravel()
            if len(si) == 0:
                continue

            sl_sm = smooth_keypoints(sleap_3d, SL_SMOOTH[0], SL_SMOOTH[1])
            dn_sm = smooth_keypoints(dannce_3d, DN_SMOOTH[0], DN_SMOOTH[1])
            sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
            dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
            sl_xyz = project_to_pcs(sl_rot, pw, fm)[:, pcu]
            dn_xyz = project_to_pcs(dn_rot, pw, fm)[:, pcu]

            # GT events from DANNCE
            gt_bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
            gt_m = run_template_matching(dn_xyz, tmpl_smooth, gt_bounds, max_outside=3)

            for m in gt_m:
                # SLEAP frame corresponding to this DANNCE frame
                if m >= len(si):
                    continue
                sl_end = si[m]
                sl_start = sl_end - WIN
                if sl_start < 0:
                    continue
                sl_window = sl_xyz[sl_start:sl_end]
                if len(sl_window) == WIN:
                    event_sl_windows.append(sl_window)
                    event_dn_windows.append(dn_xyz[max(0, m - WIN):m])
        except Exception:
            continue

    if len(event_sl_windows) < 3:
        return None, None, None, []

    # Build mean SLEAP template from event windows
    sl_windows = np.array(event_sl_windows)  # (n_events, WIN, n_pcs)
    sl_tmpl = sl_windows.mean(axis=0)  # (WIN, n_pcs)

    # MSE values of event windows from the mean SLEAP template
    mse_vals = [np.mean((w - sl_tmpl)**2) for w in event_sl_windows]
    sl_mse_thresh = np.percentile(mse_vals, 95) * 1.5

    # Correlation values
    b = sl_tmpl.ravel()
    corr_vals = []
    for w in event_sl_windows:
        a = w.ravel()
        if np.std(a) > 0 and np.std(b) > 0:
            corr_vals.append(np.corrcoef(a, b)[0, 1])
    sl_corr_thresh = np.percentile(corr_vals, 5) * 0.9 if corr_vals else 0.0

    print(f"    Calibrated SLEAP template from {len(event_sl_windows)} events "
          f"(MSE_thresh={sl_mse_thresh:.1f}, corr_thresh={sl_corr_thresh:.3f})")

    return sl_tmpl, sl_mse_thresh, sl_corr_thresh, event_sl_windows


# ─────────────────────────────────────────────────────────────────────────────
# Cross-session MSE calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_mse_cross_session(rat, t, calibration_sessions, bounds_scalar):
    """
    Pool MSE values from calibration sessions for threshold estimation.
    Uses leave-one-out: calibrate on all other sessions, test on this session.

    Returns dict: {session: mse_thresh}
    """
    pw = t['pc_weights']
    fm = t['feature_means']
    pcu = t['pcs_to_use'].ravel().astype(int)
    xyz_stds = t['feature_stds'][pcu]
    tmpl_stored = t['template'][:, pcu]
    from scipy.signal import savgol_filter
    win_size = min(7, tmpl_stored.shape[0]) | 1
    tmpl_smooth = tmpl_stored.copy()
    if win_size >= 3:
        for j in range(tmpl_smooth.shape[1]):
            tmpl_smooth[:, j] = savgol_filter(tmpl_smooth[:, j], win_size, 1)

    # Collect MSE values from DANNCE GT events for each session
    session_mse = {}
    for sess in calibration_sessions:
        try:
            _, dannce_3d, aligned = load_session_data(rat, sess)
            if aligned is None:
                continue
            dn_sm = smooth_keypoints(dannce_3d, DN_SMOOTH[0], DN_SMOOTH[1])
            dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
            dn_xyz = project_to_pcs(dn_rot, pw, fm)[:, pcu]
            gt_bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
            gt_m = run_template_matching(dn_xyz, tmpl_smooth, gt_bounds, max_outside=3)
            mse_vals = []
            for m in gt_m:
                chunk = dn_xyz[max(0, m - WIN):m]
                if len(chunk) == WIN:
                    mse_vals.append(np.mean((chunk - tmpl_smooth)**2))
            if mse_vals:
                session_mse[sess] = mse_vals
        except Exception:
            continue

    if len(session_mse) < 2:
        all_mse = [v for vals in session_mse.values() for v in vals]
        global_thresh = np.percentile(all_mse, 95) * 1.5 if all_mse else 1e9
        return {}, global_thresh

    # Leave-one-out calibration
    loo_thresh = {}
    all_sessions = list(session_mse.keys())
    for test_sess in all_sessions:
        pool = []
        for s, vals in session_mse.items():
            if s != test_sess:
                pool.extend(vals)
        if pool:
            loo_thresh[test_sess] = np.percentile(pool, 95) * 1.5
        else:
            loo_thresh[test_sess] = np.percentile(session_mse[test_sess], 95) * 1.5

    # Global threshold from all sessions
    all_mse = [v for vals in session_mse.values() for v in vals]
    global_thresh = np.percentile(all_mse, 95) * 1.5

    return loo_thresh, global_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Session processing
# ─────────────────────────────────────────────────────────────────────────────

def process_session(rat, session, t, bounds_scalar,
                    sl_tmpl_calib, sl_mse_thresh, sl_corr_thresh,
                    loo_mse_thresh, global_mse_thresh):
    """
    Process one session with the new approaches.
    """
    pw = t['pc_weights']
    fm = t['feature_means']
    pcu = t['pcs_to_use'].ravel().astype(int)
    xyz_stds = t['feature_stds'][pcu]
    tmpl_stored = t['template'][:, pcu]
    from scipy.signal import savgol_filter
    win_size = min(7, tmpl_stored.shape[0]) | 1
    tmpl_smooth = tmpl_stored.copy()
    if win_size >= 3:
        for j in range(tmpl_smooth.shape[1]):
            tmpl_smooth[:, j] = savgol_filter(tmpl_smooth[:, j], win_size, 1)

    try:
        sleap_3d, dannce_3d, aligned = load_session_data(rat, session)
    except Exception as e:
        return [{'rat': rat, 'session': session, 'error': f'load: {e}'}]

    st = np.array(aligned['sleap_times_ms']).ravel() if aligned else None
    dn_t = st  # DANNCE resampled to SLEAP frame rate

    try:
        sl_sm = smooth_keypoints(sleap_3d, SL_SMOOTH[0], SL_SMOOTH[1])
        dn_sm = smooth_keypoints(dannce_3d, DN_SMOOTH[0], DN_SMOOTH[1])
        sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
        dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
        sl_xyz = project_to_pcs(sl_rot, pw, fm)[:, pcu]
        dn_xyz = project_to_pcs(dn_rot, pw, fm)[:, pcu]
    except Exception as e:
        return [{'rat': rat, 'session': session, 'error': f'features: {e}'}]

    # GT matches (DANNCE)
    gt_bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
    gt_m = run_template_matching(dn_xyz, tmpl_smooth, gt_bounds, max_outside=3)
    if not gt_m:
        return []

    # Initial offset estimate
    sl_m_init = run_template_matching(sl_xyz, tmpl_smooth, gt_bounds, max_outside=3)
    if len(sl_m_init) >= 2 and len(gt_m) >= 2:
        offset_ms = estimate_temporal_offset(sl_m_init, gt_m, st, st)
    else:
        offset_ms = 0.0

    # Per-session MSE calibration (baseline)
    sess_mse_vals = []
    for m in gt_m:
        chunk = dn_xyz[max(0, m - WIN):m]
        if len(chunk) == WIN:
            sess_mse_vals.append(np.mean((chunk - tmpl_smooth)**2))
    sess_thresh_95 = np.percentile(sess_mse_vals, 95) * 1.5 if sess_mse_vals else 1e9

    rows = []

    # ═══════════════════════════════════════════════════════════════════════
    # F. SLEAP-calibrated template (if available)
    # ═══════════════════════════════════════════════════════════════════════
    if sl_tmpl_calib is not None:
        # F1: MSE matching with SLEAP calibrated template
        for refractory in REFRACTORY_SWEEP:
            sl_m = mse_match_refractory(sl_xyz, sl_tmpl_calib, sl_mse_thresh, refractory)
            dn_m = mse_match_refractory(dn_xyz, sl_tmpl_calib, sl_mse_thresh, refractory)
            al_sl = compute_alignment_multi_tol(sl_m, gt_m, TOLERANCES_MS, st, st, offset_ms)
            al_dn = compute_alignment_multi_tol(dn_m, gt_m, TOLERANCES_MS, st, st, 0.0)
            rows.append(make_row(rat, session, 'F_sl_calib', 'F_sl_mse',
                                 SL_SMOOTH[0], SL_SMOOTH[1], 'sleap_calib', bounds_scalar,
                                 refractory, gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                                 n_pcs=len(pcu), refractory=refractory))

        # F2: Correlation matching with SLEAP calibrated template
        if sl_corr_thresh is not None:
            for refractory in REFRACTORY_SWEEP:
                sl_m = corr_match_refractory(sl_xyz, sl_tmpl_calib, sl_corr_thresh, refractory)
                dn_m = corr_match_refractory(dn_xyz, sl_tmpl_calib, sl_corr_thresh, refractory)
                al_sl = compute_alignment_multi_tol(sl_m, gt_m, TOLERANCES_MS, st, st, offset_ms)
                al_dn = compute_alignment_multi_tol(dn_m, gt_m, TOLERANCES_MS, st, st, 0.0)
                rows.append(make_row(rat, session, 'F_sl_calib', 'F_sl_corr',
                                     SL_SMOOTH[0], SL_SMOOTH[1], 'sleap_calib_corr', bounds_scalar,
                                     refractory, gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                                     n_pcs=len(pcu), refractory=refractory))

    # ═══════════════════════════════════════════════════════════════════════
    # G. Refractory period sweep (MSE and correlation with original template)
    # ═══════════════════════════════════════════════════════════════════════
    for refractory in REFRACTORY_SWEEP:
        # G1: MSE (per-session calibration)
        sl_m = mse_match_refractory(sl_xyz, tmpl_smooth, sess_thresh_95, refractory)
        dn_m = mse_match_refractory(dn_xyz, tmpl_smooth, sess_thresh_95, refractory)
        al_sl = compute_alignment_multi_tol(sl_m, gt_m, TOLERANCES_MS, st, st, offset_ms)
        al_dn = compute_alignment_multi_tol(dn_m, gt_m, TOLERANCES_MS, st, st, 0.0)
        rows.append(make_row(rat, session, 'G_refractory', 'G_mse_sess',
                             SL_SMOOTH[0], SL_SMOOTH[1], 'mse_sess95', bounds_scalar,
                             refractory, gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                             n_pcs=len(pcu), refractory=refractory))

        # G2: Correlation (per-session calibration)
        dn_corr_vals = []
        b = tmpl_smooth.ravel()
        for m in gt_m:
            chunk = dn_xyz[max(0, m - WIN):m]
            if len(chunk) == WIN:
                a = chunk.ravel()
                if np.std(a) > 0 and np.std(b) > 0:
                    dn_corr_vals.append(np.corrcoef(a, b)[0, 1])
        corr_thresh = np.percentile(dn_corr_vals, 10) * 0.9 if dn_corr_vals else 0.0
        sl_m_c = corr_match_refractory(sl_xyz, tmpl_smooth, corr_thresh, refractory)
        dn_m_c = corr_match_refractory(dn_xyz, tmpl_smooth, corr_thresh, refractory)
        al_sl = compute_alignment_multi_tol(sl_m_c, gt_m, TOLERANCES_MS, st, st, offset_ms)
        al_dn = compute_alignment_multi_tol(dn_m_c, gt_m, TOLERANCES_MS, st, st, 0.0)
        rows.append(make_row(rat, session, 'G_refractory', 'G_corr_sess',
                             SL_SMOOTH[0], SL_SMOOTH[1], 'corr_sess', bounds_scalar,
                             refractory, gt_m, sl_m_c, dn_m_c, al_sl, al_dn, offset_ms,
                             n_pcs=len(pcu), refractory=refractory))

    # ═══════════════════════════════════════════════════════════════════════
    # H. Cross-session MSE calibration
    # ═══════════════════════════════════════════════════════════════════════
    if loo_mse_thresh is not None:
        for thresh_name, thresh in [
            ('loo', loo_mse_thresh.get(session, global_mse_thresh)),
            ('global', global_mse_thresh),
        ]:
            for refractory in [20, 30]:
                sl_m = mse_match_refractory(sl_xyz, tmpl_smooth, thresh, refractory)
                dn_m = mse_match_refractory(dn_xyz, tmpl_smooth, thresh, refractory)
                al_sl = compute_alignment_multi_tol(sl_m, gt_m, TOLERANCES_MS, st, st, offset_ms)
                al_dn = compute_alignment_multi_tol(dn_m, gt_m, TOLERANCES_MS, st, st, 0.0)
                rows.append(make_row(rat, session, 'H_cross_sess', f'H_mse_{thresh_name}',
                                     SL_SMOOTH[0], SL_SMOOTH[1], f'mse_{thresh_name}', bounds_scalar,
                                     refractory, gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                                     n_pcs=len(pcu), refractory=refractory))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_rat(rat, config_key='primary', max_sessions=None):
    cfg = RAT_CONFIG[rat][config_key]
    template_file = cfg['template_file']
    bounds_scalar = cfg['bounds']

    print(f"\n{'='*60}")
    print(f"RAT={rat}  config={config_key}  template={template_file}  bounds={bounds_scalar}")
    print(f"{'='*60}")

    t, has_origin = load_template_info(rat, template_file)
    df = get_sessions(rat=rat)
    if max_sessions:
        df = df.head(max_sessions)
    sessions = df['session'].tolist()
    print(f"Sessions: {len(sessions)}")

    # Identify calibration sessions (good-offset from prior run)
    good_sess = get_good_sessions(rat)
    if good_sess is None:
        good_sess = sessions[:min(10, len(sessions))]
    calib_sessions = [s for s in good_sess if s in sessions]
    print(f"Calibration sessions: {len(calib_sessions)}")

    # Build SLEAP-calibrated template
    print("Building SLEAP-calibrated template...")
    t0 = time.time()
    sl_tmpl, sl_mse_thresh, sl_corr_thresh, calib_windows = build_sleap_calibrated_template(
        rat, t, calib_sessions, bounds_scalar
    )
    print(f"  {time.time()-t0:.1f}s  (windows: {len(calib_windows)})")

    # Cross-session MSE calibration
    print("Cross-session MSE calibration...")
    t0 = time.time()
    loo_thresh, global_thresh = calibrate_mse_cross_session(rat, t, calib_sessions, bounds_scalar)
    g_str = f'{global_thresh:.1f}' if global_thresh and global_thresh < 1e8 else 'N/A'
    print(f"  {time.time()-t0:.1f}s  global_thresh={g_str}")

    all_rows = []
    log_lines = []

    for i, session in enumerate(sessions):
        t0 = time.time()
        try:
            rows = process_session(
                rat, session, t, bounds_scalar,
                sl_tmpl, sl_mse_thresh, sl_corr_thresh,
                loo_thresh, global_thresh
            )
            ok = [r for r in rows if 'error' not in r]
            err = [r for r in rows if 'error' in r]
            all_rows.extend(ok)
            for r in err:
                log_lines.append(f"ERR {session}: {r.get('error','')[:120]}")
            print(f"  [{i+1:3d}/{len(sessions)}] {session}  {len(ok):5d} rows  {time.time()-t0:.1f}s")
        except Exception as e:
            log_lines.append(f"FATAL {session}: {traceback.format_exc()[:200]}")
            print(f"  [{i+1:3d}/{len(sessions)}] {session}  FATAL {e}")

        if (i + 1) % 10 == 0:
            pd.DataFrame(all_rows).to_csv(
                RESULTS_DIR / f'{rat}_{config_key}_v2_checkpoint.csv', index=False)

    out_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / f'{rat}_{config_key}_v2_results.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows → {out_path}")

    with open(LOGS_DIR / f'{rat}_{config_key}_v2.log', 'w') as f:
        f.write('\n'.join(log_lines))

    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rat', default='R1', choices=['R1', 'R2', 'R3'])
    parser.add_argument('--config', default='primary', choices=['primary', 'secondary'])
    parser.add_argument('--max_sessions', type=int, default=None)
    args = parser.parse_args()
    run_rat(args.rat, args.config, max_sessions=args.max_sessions)
