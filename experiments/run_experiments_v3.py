"""
F1-optimized template matching experiments (v3).

Goal: maximize F1 (equal weight on precision and recall), rather than recall alone.

New approaches:
I. MSE threshold sweep  — sweep threshold percentile (50th–99th) × refractory (10,20,30,40,50)
   to map out the full F1 surface and find the operating point that maximizes it.
   Calibration source: per-session DANNCE GT events.
   Thresholds probed: pct25, pct50, pct75, pct90, pct95 of GT MSE values (no ×1.5 inflation).

J. Bounds-based strict matching — sweep max_outside (0–3) × bounds_scalar (0.5–2.0)
   using uniform bounds. Also tries percentile-derived bounds (95th pct of
   SLEAP-DANNCE aligned differences). Target: find region where precision ~ recall.

K. F1-optimized correlation — sweep correlation threshold percentile (10th–90th of GT)
   × refractory (10,20,30,40,50). Identifies tighter thresholds that cut false positives.

L. Two-stage filter — first pass: loose MSE to find candidates; second pass: require
   correlation ≥ per-session threshold. Combines MSE recall with correlation precision.

M. SLEAP-calibrated MSE (extended) — reuse build_sleap_calibrated_template from v2 but
   sweep threshold percentile and refractory to find F1-optimal operating point.

Usage:
    python run_experiments_v3.py --rat R1 --config secondary
    python run_experiments_v3.py --rat R2 --config primary
    python run_experiments_v3.py --rat R1 --config primary
"""
import argparse
import numpy as np
import pandas as pd
import sys, time, traceback
from pathlib import Path
from numpy.lib.stride_tricks import as_strided
import warnings
warnings.filterwarnings('ignore')

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp_utils import (
    load_session_data, smooth_keypoints,
    compute_pairwise_distances, run_template_matching,
    compute_alignment_multi_tol, estimate_temporal_offset,
    SLEAP_HZ, DANNCE_HZ
)
from data_io import load_template, get_sessions
from skeleton import normalize_skeleton_batch, project_to_pcs

RESULTS_DIR = ROOT / 'results' / 'metrics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
WIN = 30

# Sweep parameters
MSE_PCT_SWEEP    = [25, 50, 75, 90, 95, 99]   # percentile of GT MSE (no inflation)
MSE_SCALE_SWEEP  = [1.0, 1.5, 2.0]            # multiplier on top of percentile
REFRAC_SWEEP     = [10, 20, 30, 40, 50]        # frames
CORR_PCT_SWEEP   = [5, 10, 20, 30, 50, 70, 90] # percentile of GT correlations (lower = tighter)
BOUNDS_SCALAR_SWEEP = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
MAX_OUT_SWEEP    = [0, 1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized matching primitives
# ─────────────────────────────────────────────────────────────────────────────

def _sliding_mse(feat, tmpl):
    """Return (n_windows,) array of MSE values for every window."""
    T, n_pcs = feat.shape
    win = tmpl.shape[0]
    n_windows = T - win + 1
    if n_windows <= 0:
        return np.array([])
    shape = (n_windows, win, n_pcs)
    strides = (feat.strides[0], feat.strides[0], feat.strides[1])
    windows = as_strided(feat, shape=shape, strides=strides)
    return np.mean((windows - tmpl[None, :, :]) ** 2, axis=(1, 2))


def _sliding_corr(feat, tmpl):
    """Return (n_windows,) array of Pearson correlation values."""
    T, n_pcs = feat.shape
    win = tmpl.shape[0]
    n_windows = T - win + 1
    if n_windows <= 0:
        return np.array([])
    shape = (n_windows, win, n_pcs)
    strides = (feat.strides[0], feat.strides[0], feat.strides[1])
    windows = as_strided(feat, shape=shape, strides=strides)
    b = tmpl.ravel()
    a = windows.reshape(n_windows, -1)
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean()
    num = (a_c * b_c).sum(axis=1)
    denom = np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum())
    return np.where(denom > 0, num / denom, 0.0)


def apply_refractory(candidates, refractory, win):
    """Apply refractory period to candidate window indices, return match frame indices."""
    if len(candidates) == 0:
        return []
    frame_candidates = candidates + win - 1
    matches = [frame_candidates[0]]
    for c in frame_candidates[1:]:
        if c - matches[-1] >= refractory:
            matches.append(c)
    return matches


def mse_match(feat, tmpl, threshold, refractory):
    mse = _sliding_mse(feat, tmpl)
    if len(mse) == 0:
        return []
    return apply_refractory(np.where(mse <= threshold)[0], refractory, tmpl.shape[0])


def corr_match(feat, tmpl, threshold, refractory):
    corr = _sliding_corr(feat, tmpl)
    if len(corr) == 0:
        return []
    return apply_refractory(np.where(corr >= threshold)[0], refractory, tmpl.shape[0])


def two_stage_match(feat, tmpl, mse_thresh, corr_thresh, refractory):
    """
    Two-stage filter: MSE gates candidates, correlation filters them.
    Returns matches where both MSE <= mse_thresh AND corr >= corr_thresh,
    then applies refractory.
    """
    mse = _sliding_mse(feat, tmpl)
    corr = _sliding_corr(feat, tmpl)
    if len(mse) == 0:
        return []
    candidates = np.where((mse <= mse_thresh) & (corr >= corr_thresh))[0]
    return apply_refractory(candidates, refractory, tmpl.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# Template helpers
# ─────────────────────────────────────────────────────────────────────────────

def smooth_template(tmpl, window=7):
    from scipy.signal import savgol_filter
    w = min(window, tmpl.shape[0]) | 1
    out = tmpl.copy()
    if w >= 3:
        for j in range(tmpl.shape[1]):
            out[:, j] = savgol_filter(out[:, j], w, 1)
    return out


def get_gt_matches(dn_xyz, tmpl, xyz_stds, bounds_scalar):
    """Ground-truth matches from DANNCE using uniform bounds, max_outside=3."""
    gt_bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
    return run_template_matching(dn_xyz, tmpl, gt_bounds, max_outside=3)


def estimate_offset(sl_xyz, dn_xyz, tmpl, xyz_stds, bounds_scalar, st, gt_m):
    gt_bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
    sl_init = run_template_matching(sl_xyz, tmpl, gt_bounds, max_outside=3)
    if len(sl_init) >= 2 and len(gt_m) >= 2 and st is not None:
        return estimate_temporal_offset(sl_init, gt_m, st, st)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-session calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_cross_session(rat, t, calibration_sessions, bounds_scalar):
    """
    Collect per-session MSE and correlation distributions from DANNCE GT events.
    Returns:
        session_mse_vals  : {session: [mse values at GT events]}
        session_corr_vals : {session: [corr values at GT events]}
        global_mse_pcts   : {pct: threshold} from pooled MSE values
        global_corr_pcts  : {pct: threshold} from pooled corr values
    """
    pw = t['pc_weights']
    fm = t['feature_means']
    pcu = t['pcs_to_use'].ravel().astype(int)
    xyz_stds = t['feature_stds'][pcu]
    tmpl = smooth_template(t['template'][:, pcu])
    b = tmpl.ravel()

    session_mse_vals = {}
    session_corr_vals = {}

    for sess in calibration_sessions:
        try:
            _, dannce_3d, aligned = load_session_data(rat, sess)
            if aligned is None:
                continue
            dn_sm = smooth_keypoints(dannce_3d, DN_SMOOTH[0], DN_SMOOTH[1])
            dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
            dn_xyz = project_to_pcs(dn_rot, pw, fm)[:, pcu]
            gt_m = get_gt_matches(dn_xyz, tmpl, xyz_stds, bounds_scalar)
            mse_vals, corr_vals = [], []
            for m in gt_m:
                chunk = dn_xyz[max(0, m - WIN):m]
                if len(chunk) == WIN:
                    mse_vals.append(np.mean((chunk - tmpl) ** 2))
                    a = chunk.ravel()
                    if np.std(a) > 0 and np.std(b) > 0:
                        corr_vals.append(np.corrcoef(a, b)[0, 1])
            if mse_vals:
                session_mse_vals[sess] = mse_vals
            if corr_vals:
                session_corr_vals[sess] = corr_vals
        except Exception:
            continue

    all_mse = [v for vals in session_mse_vals.values() for v in vals]
    all_corr = [v for vals in session_corr_vals.values() for v in vals]

    global_mse_pcts = {p: np.percentile(all_mse, p) if all_mse else 1e9
                       for p in MSE_PCT_SWEEP}
    global_corr_pcts = {p: np.percentile(all_corr, p) if all_corr else 0.0
                        for p in CORR_PCT_SWEEP}

    return session_mse_vals, session_corr_vals, global_mse_pcts, global_corr_pcts


# ─────────────────────────────────────────────────────────────────────────────
# Row building
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
        key = f'tol_{tol}ms'
        for which, al in [('sl', al_sl), ('dn', al_dn)]:
            r = al.get(key, {})
            row[f'{which}_recall_{tol}']    = round(r.get('recall', 0.0), 4)
            row[f'{which}_precision_{tol}'] = round(r.get('precision', 0.0), 4)
            row[f'{which}_f1_{tol}']        = round(r.get('f1', 0.0), 4)
            row[f'{which}_n_both_{tol}']    = r.get('n_both', 0)
    return row


def score(sl_m, dn_m, gt_m, times_ms, offset_ms):
    al_sl = compute_alignment_multi_tol(sl_m, gt_m, TOLERANCES_MS, times_ms, times_ms, offset_ms)
    al_dn = compute_alignment_multi_tol(dn_m, gt_m, TOLERANCES_MS, times_ms, times_ms, 0.0)
    return al_sl, al_dn


# ─────────────────────────────────────────────────────────────────────────────
# Session processing
# ─────────────────────────────────────────────────────────────────────────────

def process_session(rat, session, t, bounds_scalar,
                    global_mse_pcts, global_corr_pcts,
                    session_mse_vals, session_corr_vals):
    pw = t['pc_weights']
    fm = t['feature_means']
    pcu = t['pcs_to_use'].ravel().astype(int)
    xyz_stds = t['feature_stds'][pcu]
    tmpl = smooth_template(t['template'][:, pcu])
    b = tmpl.ravel()

    try:
        sleap_3d, dannce_3d, aligned = load_session_data(rat, session)
    except Exception as e:
        return [{'rat': rat, 'session': session, 'error': f'load: {e}'}]

    st   = np.array(aligned['sleap_times_ms']).ravel() if aligned else None
    dn_t = st

    try:
        sl_sm = smooth_keypoints(sleap_3d, SL_SMOOTH[0], SL_SMOOTH[1])
        dn_sm = smooth_keypoints(dannce_3d, DN_SMOOTH[0], DN_SMOOTH[1])
        sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
        dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
        sl_xyz = project_to_pcs(sl_rot, pw, fm)[:, pcu]
        dn_xyz = project_to_pcs(dn_rot, pw, fm)[:, pcu]
    except Exception as e:
        return [{'rat': rat, 'session': session, 'error': f'features: {e}'}]

    gt_m = get_gt_matches(dn_xyz, tmpl, xyz_stds, bounds_scalar)
    if not gt_m:
        return []

    offset_ms = estimate_offset(sl_xyz, dn_xyz, tmpl, xyz_stds, bounds_scalar, st, gt_m)

    # Pre-compute sliding MSE and correlation for both SLEAP and DANNCE (fast, reused below)
    sl_mse_arr  = _sliding_mse(sl_xyz, tmpl)
    dn_mse_arr  = _sliding_mse(dn_xyz, tmpl)
    sl_corr_arr = _sliding_corr(sl_xyz, tmpl)
    dn_corr_arr = _sliding_corr(dn_xyz, tmpl)

    # Per-session calibration from DANNCE GT events
    sess_mse_vals, sess_corr_vals = [], []
    for m in gt_m:
        chunk = dn_xyz[max(0, m - WIN):m]
        if len(chunk) == WIN:
            sess_mse_vals.append(np.mean((chunk - tmpl) ** 2))
            a = chunk.ravel()
            if np.std(a) > 0 and np.std(b) > 0:
                sess_corr_vals.append(np.corrcoef(a, b)[0, 1])

    sess_mse_pcts  = {p: np.percentile(sess_mse_vals, p)  if sess_mse_vals  else 1e9
                      for p in MSE_PCT_SWEEP}
    sess_corr_pcts = {p: np.percentile(sess_corr_vals, p) if sess_corr_vals else 0.0
                      for p in CORR_PCT_SWEEP}

    rows = []

    # ═══════════════════════════════════════════════════════════════════════
    # I. MSE threshold sweep × refractory sweep
    # ═══════════════════════════════════════════════════════════════════════
    for calib, mse_pcts in [('sess', sess_mse_pcts), ('global', global_mse_pcts)]:
        for pct in MSE_PCT_SWEEP:
            for scale in MSE_SCALE_SWEEP:
                thresh = mse_pcts[pct] * scale
                for ref in REFRAC_SWEEP:
                    sl_m = apply_refractory(np.where(sl_mse_arr <= thresh)[0], ref, WIN)
                    dn_m = apply_refractory(np.where(dn_mse_arr <= thresh)[0], ref, WIN)
                    al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
                    rows.append(make_row(
                        rat, session, 'I_mse_sweep',
                        f'I_mse_{calib}_p{pct}_s{scale:.0f}',
                        gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                        mse_pct=pct, mse_scale=scale, calib=calib, refractory=ref
                    ))

    # ═══════════════════════════════════════════════════════════════════════
    # J. Bounds-based sweep (uniform × bounds_scalar × max_outside)
    # ═══════════════════════════════════════════════════════════════════════
    for scalar in BOUNDS_SCALAR_SWEEP:
        bounds = np.tile(xyz_stds * scalar, (WIN, 1))
        for mo in MAX_OUT_SWEEP:
            sl_m = run_template_matching(sl_xyz, tmpl, bounds, max_outside=mo,
                                         refractory_frames=30)
            dn_m = run_template_matching(dn_xyz, tmpl, bounds, max_outside=mo,
                                         refractory_frames=30)
            al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
            rows.append(make_row(
                rat, session, 'J_bounds_sweep',
                f'J_bounds_s{scalar:.2f}_mo{mo}',
                gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                bounds_scalar=scalar, max_outside=mo, refractory=30
            ))

    # Also sweep bounds_scalar × refractory at fixed max_outside=1 (found to be near-optimal in v1)
    for scalar in BOUNDS_SCALAR_SWEEP:
        bounds = np.tile(xyz_stds * scalar, (WIN, 1))
        for ref in REFRAC_SWEEP:
            sl_m = run_template_matching(sl_xyz, tmpl, bounds, max_outside=1,
                                         refractory_frames=ref)
            dn_m = run_template_matching(dn_xyz, tmpl, bounds, max_outside=1,
                                         refractory_frames=ref)
            al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
            rows.append(make_row(
                rat, session, 'J_bounds_refrac',
                f'J_brefrac_s{scalar:.2f}_r{ref}',
                gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                bounds_scalar=scalar, max_outside=1, refractory=ref
            ))

    # ═══════════════════════════════════════════════════════════════════════
    # K. Correlation threshold sweep × refractory sweep
    # ═══════════════════════════════════════════════════════════════════════
    for calib, corr_pcts in [('sess', sess_corr_pcts), ('global', global_corr_pcts)]:
        for pct in CORR_PCT_SWEEP:
            thresh = corr_pcts[pct]
            for ref in REFRAC_SWEEP:
                sl_m = apply_refractory(np.where(sl_corr_arr >= thresh)[0], ref, WIN)
                dn_m = apply_refractory(np.where(dn_corr_arr >= thresh)[0], ref, WIN)
                al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
                rows.append(make_row(
                    rat, session, 'K_corr_sweep',
                    f'K_corr_{calib}_p{pct}',
                    gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                    corr_pct=pct, calib=calib, refractory=ref
                ))

    # ═══════════════════════════════════════════════════════════════════════
    # L. Two-stage filter: MSE gating + correlation filtering
    # ═══════════════════════════════════════════════════════════════════════
    # Use loose MSE (sess p95×2) as first gate, then sweep correlation threshold
    mse_gate = sess_mse_pcts.get(95, 1e9) * 2.0
    for corr_pct in CORR_PCT_SWEEP:
        corr_thresh = sess_corr_pcts.get(corr_pct, 0.0)
        for ref in REFRAC_SWEEP:
            sl_m = two_stage_match(sl_xyz, tmpl, mse_gate, corr_thresh, ref)
            dn_m = two_stage_match(dn_xyz, tmpl, mse_gate, corr_thresh, ref)
            al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
            rows.append(make_row(
                rat, session, 'L_two_stage',
                f'L_two_stage_cp{corr_pct}',
                gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                corr_pct=corr_pct, refractory=ref
            ))

    # Also sweep the MSE gate level alongside correlation threshold
    for mse_pct in [75, 90, 95]:
        for corr_pct in [10, 30, 50]:
            mse_gate = sess_mse_pcts.get(mse_pct, 1e9) * 1.5
            corr_thresh = sess_corr_pcts.get(corr_pct, 0.0)
            for ref in [20, 30]:
                sl_m = two_stage_match(sl_xyz, tmpl, mse_gate, corr_thresh, ref)
                dn_m = two_stage_match(dn_xyz, tmpl, mse_gate, corr_thresh, ref)
                al_sl, al_dn = score(sl_m, dn_m, gt_m, st, offset_ms)
                rows.append(make_row(
                    rat, session, 'L_two_stage_grid',
                    f'L_2s_m{mse_pct}_c{corr_pct}_r{ref}',
                    gt_m, sl_m, dn_m, al_sl, al_dn, offset_ms,
                    mse_pct=mse_pct, corr_pct=corr_pct, refractory=ref
                ))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_rat(rat, config_key='primary', max_sessions=None):
    cfg = RAT_CONFIG[rat][config_key]
    template_file = cfg['template_file']
    bounds_scalar = cfg['bounds']

    t = dict(load_template(rat, template_file))
    df_sess = get_sessions(rat=rat)
    if max_sessions:
        df_sess = df_sess.head(max_sessions)
    sessions = df_sess['session'].tolist()

    out_path = RESULTS_DIR / f'{rat}_{config_key}_v3_results.csv'
    ckpt_path = RESULTS_DIR / f'{rat}_{config_key}_v3_checkpoint.csv'

    # Resume from checkpoint
    done_sessions = set()
    if ckpt_path.exists():
        done_df = pd.read_csv(ckpt_path)
        done_sessions = set(done_df['session'].unique())
        print(f"Resuming: {len(done_sessions)} sessions already done.")

    todo = [s for s in sessions if s not in done_sessions]
    print(f"{rat} {config_key}: {len(todo)} sessions to run ({len(done_sessions)} already done)")

    # Cross-session calibration on ALL sessions upfront
    print("Running cross-session calibration...")
    t0 = time.time()
    session_mse_vals, session_corr_vals, global_mse_pcts, global_corr_pcts = \
        calibrate_cross_session(rat, t, sessions, bounds_scalar)
    print(f"  Calibration done in {time.time()-t0:.1f}s on {len(session_mse_vals)} sessions")
    print(f"  Global MSE p95={global_mse_pcts.get(95, 'N/A'):.3f}, "
          f"corr p10={global_corr_pcts.get(10, 'N/A'):.3f}")

    all_rows = []
    t_start = time.time()

    for i, session in enumerate(todo):
        t1 = time.time()
        try:
            rows = process_session(
                rat, session, t, bounds_scalar,
                global_mse_pcts, global_corr_pcts,
                session_mse_vals, session_corr_vals
            )
        except Exception as e:
            traceback.print_exc()
            rows = [{'rat': rat, 'session': session, 'error': str(e)}]

        elapsed = time.time() - t1
        valid = [r for r in rows if 'error' not in r]
        print(f"  [{i+1}/{len(todo)}] {session}: {len(valid)} rows in {elapsed:.1f}s")

        if valid:
            all_rows.extend(valid)
            chunk = pd.DataFrame(valid)
            if ckpt_path.exists():
                chunk.to_csv(ckpt_path, mode='a', header=False, index=False)
            else:
                chunk.to_csv(ckpt_path, index=False)

    # Merge checkpoint with any previously completed sessions
    if ckpt_path.exists():
        final_df = pd.read_csv(ckpt_path)
    else:
        final_df = pd.DataFrame(all_rows)

    final_df.to_csv(out_path, index=False)
    print(f"\nDone. {len(final_df)} rows saved to {out_path}")
    print(f"Total time: {time.time()-t_start:.1f}s")

    # Quick summary of best F1
    if len(final_df) > 0 and 'sl_f1_300' in final_df.columns:
        good = final_df[final_df['temporal_offset_ms'] > 100]
        if len(good) > 0:
            grp = good.groupby(['group', 'exp_name'])['sl_f1_300'].mean()
            best = grp.sort_values(ascending=False).head(5)
            print("\nTop 5 by F1 (good-offset sessions):")
            print(best.round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rat', required=True, choices=['R1', 'R2', 'R3'])
    parser.add_argument('--config', default='primary', choices=['primary', 'secondary'])
    parser.add_argument('--max_sessions', type=int, default=None)
    args = parser.parse_args()
    run_rat(args.rat, args.config, args.max_sessions)
