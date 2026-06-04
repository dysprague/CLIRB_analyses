"""
Analysis: SLEAP vs DANNCE keypoint/PC error vs template matching F1.

For each session across R1, R2, R3:
  1. Compute keypoint MSE (egocentric) between SLEAP and DANNCE
  2. Compute PC MSE (2-PC template space) between SLEAP and DANNCE
  3. Get best SLEAP F1@300ms from results CSVs
  4. Correlate (1), (2) with (3)
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '/home/yutaka-sprague/CLIRB_analyses')

from data_io import load_template, get_sessions
from experiments.exp_utils import load_session_data, smooth_keypoints
from skeleton import normalize_skeleton_batch, project_to_pcs

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = '/home/yutaka-sprague/CLIRB_analyses/results/metrics'
RESULTS_CSVS = {
    'R1': os.path.join(RESULTS_DIR, 'R1_secondary_results.csv'),
    'R2': os.path.join(RESULTS_DIR, 'R2_primary_results.csv'),
    'R3': os.path.join(RESULTS_DIR, 'R3_primary_results.csv'),
}
CANONICAL_TEMPLATES = {
    'R1': 'R1_template_1.npz',
    'R2': 'R2_template_1.npz',
    'R3': 'R3_template_1.npz',
}
SMOOTH_METHOD = 'median'
SMOOTH_WINDOW = 11
N_NODES = 23

# ─────────────────────────────────────────────────────────────────────────────
# Load best F1 per session from results CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_best_f1(rat):
    """Return DataFrame indexed by session with best sl_f1_300 and max n_gt."""
    df = pd.read_csv(RESULTS_CSVS[rat])
    # Best F1 = max across all exp_name rows for that session
    grp = df.groupby('session').agg(
        best_sl_f1=('sl_f1_300', 'max'),
        n_gt=('n_gt', 'max'),   # n_gt should be constant per session, but take max to be safe
    ).reset_index()
    grp['rat'] = rat
    return grp

# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

records = []

for rat in ['R1', 'R2', 'R3']:
    print(f"\n{'='*60}")
    print(f"Processing {rat}")
    print(f"{'='*60}")

    # Load canonical template
    tmpl = load_template(rat, CANONICAL_TEMPLATES[rat])
    pc_weights   = tmpl['pc_weights']    # (n_pcs, 69)
    feature_means = tmpl['feature_means']  # (69,)
    pcs_to_use   = tmpl['pcs_to_use']    # (2,) indices into pc_weights rows

    # Build the 2-PC weight matrix and means
    # project_to_pcs returns (T, n_pcs_total) — we use columns pcs_to_use[0] and pcs_to_use[1]
    pc0, pc1 = int(pcs_to_use[0]), int(pcs_to_use[1])
    print(f"  Template PCs to use: {pc0}, {pc1}")
    print(f"  pc_weights shape: {pc_weights.shape}, feature_means shape: {feature_means.shape}")

    # Load F1 results
    f1_df = load_best_f1(rat)
    f1_lookup = dict(zip(f1_df['session'], f1_df['best_sl_f1']))
    ngt_lookup = dict(zip(f1_df['session'], f1_df['n_gt']))

    # Get sessions
    session_df = get_sessions(rat=rat)

    n_sessions = len(session_df)
    for i, row in session_df.iterrows():
        session = row['session']
        task = row['task']

        # Skip if no F1 data
        if session not in f1_lookup:
            print(f"  [{session}] SKIP — not in results CSV")
            continue

        best_f1 = f1_lookup[session]
        n_gt = ngt_lookup.get(session, np.nan)

        try:
            # Load data
            sl_raw, dn_raw, aligned = load_session_data(rat, session)

            # Smooth both with median-11
            sl_smooth = smooth_keypoints(sl_raw, method=SMOOTH_METHOD, window=SMOOTH_WINDOW)
            dn_smooth = smooth_keypoints(dn_raw, method=SMOOTH_METHOD, window=SMOOTH_WINDOW)

            # Egocentric normalization
            sl_rot, _, _ = normalize_skeleton_batch(sl_smooth)   # (T, 23, 3)
            dn_rot, _, _ = normalize_skeleton_batch(dn_smooth)   # (T, 23, 3)

            # 1. Keypoint MSE (all frames, all 23 keypoints, all 3 dims)
            keypoint_mse = float(np.mean((sl_rot - dn_rot) ** 2))

            # Per-keypoint MSE: mean over frames and 3 dims, per node
            per_kp_mse = np.mean((sl_rot - dn_rot) ** 2, axis=(0, 2))  # (23,)

            # 2. PC MSE: project both to PC space, use pcs_to_use columns
            sl_pcs_all = project_to_pcs(sl_rot, pc_weights, feature_means)  # (T, n_pcs)
            dn_pcs_all = project_to_pcs(dn_rot, pc_weights, feature_means)

            sl_pcs = sl_pcs_all[:, [pc0, pc1]]   # (T, 2)
            dn_pcs = dn_pcs_all[:, [pc0, pc1]]

            pc_mse = float(np.mean((sl_pcs - dn_pcs) ** 2))

            records.append({
                'rat': rat,
                'session': session,
                'task': task,
                'keypoint_mse': keypoint_mse,
                'pc_mse': pc_mse,
                'best_sl_f1': best_f1,
                'n_gt': n_gt,
                'per_kp_mse': per_kp_mse,  # store as array for later
                'n_frames': sl_rot.shape[0],
            })

            print(f"  [{session}] kp_mse={keypoint_mse:.4f}  pc_mse={pc_mse:.6f}  "
                  f"best_f1={best_f1:.3f}  n_gt={n_gt}  T={sl_rot.shape[0]}")

        except Exception as e:
            print(f"  [{session}] ERROR: {e}")
            import traceback
            traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# Build results DataFrame
# ─────────────────────────────────────────────────────────────────────────────

# Extract per-keypoint MSE as separate columns (for inspection)
per_kp_arrays = [r.pop('per_kp_mse') for r in records]

results_df = pd.DataFrame(records)

# Also store per-keypoint breakdown separately
from config import NODES
per_kp_df = pd.DataFrame(per_kp_arrays, columns=[f'kp_mse_{n}' for n in NODES])
per_kp_df.insert(0, 'session', results_df['session'].values)
per_kp_df.insert(0, 'rat', results_df['rat'].values)

print(f"\n\nTotal sessions processed: {len(results_df)}")
print(f"  R1: {(results_df.rat=='R1').sum()}")
print(f"  R2: {(results_df.rat=='R2').sum()}")
print(f"  R3: {(results_df.rat=='R3').sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY STATISTICS (all sessions)")
print("="*70)
for col in ['keypoint_mse', 'pc_mse', 'best_sl_f1']:
    vals = results_df[col].dropna()
    print(f"\n{col}:")
    print(f"  mean={vals.mean():.6f}  median={vals.median():.6f}  "
          f"std={vals.std():.6f}  min={vals.min():.6f}  max={vals.max():.6f}")

print("\nPer-rat summary:")
print(results_df.groupby('rat')[['keypoint_mse','pc_mse','best_sl_f1']].agg(['mean','std']).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("CORRELATION: keypoint_mse / pc_mse  vs  best_sl_f1@300ms")
print("="*70)

valid = results_df.dropna(subset=['keypoint_mse', 'pc_mse', 'best_sl_f1'])

for error_col in ['keypoint_mse', 'pc_mse']:
    x = valid[error_col].values
    y = valid['best_sl_f1'].values

    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    print(f"\n{error_col} vs best_sl_f1 (n={len(x)}):")
    print(f"  Pearson  r = {r_pearson:+.4f}   p = {p_pearson:.4e}")
    print(f"  Spearman r = {r_spearman:+.4f}   p = {p_spearman:.4e}")

    # Per-rat
    for rat in ['R1', 'R2', 'R3']:
        sub = valid[valid.rat == rat]
        if len(sub) < 3:
            continue
        xr = sub[error_col].values
        yr = sub['best_sl_f1'].values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rp, pp = stats.pearsonr(xr, yr)
            rs, ps = stats.spearmanr(xr, yr)
        print(f"  {rat} (n={len(sub)}): Pearson r={rp:+.4f} p={pp:.3f}  "
              f"Spearman r={rs:+.4f} p={ps:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# F1 quartile analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("SESSIONS GROUPED BY BEST_SL_F1 QUARTILE")
print("="*70)

valid2 = valid.copy()
valid2['f1_quartile'] = pd.qcut(valid2['best_sl_f1'], q=4, labels=['Q1_low','Q2','Q3','Q4_high'])

print(valid2.groupby('f1_quartile')[['keypoint_mse','pc_mse','best_sl_f1']].agg(['mean','std','count']).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Failed vs working sessions
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("FAILED vs WORKING template matching")
print("(F1=0 OR n_gt=0  vs  F1>0 AND n_gt>0)")
print("="*70)

failed = results_df[(results_df['best_sl_f1'] == 0) | (results_df['n_gt'] == 0)]
working = results_df[(results_df['best_sl_f1'] > 0) & (results_df['n_gt'] > 0)]

print(f"\nFailed sessions: {len(failed)}")
print(f"Working sessions: {len(working)}")

for col in ['keypoint_mse', 'pc_mse']:
    f_vals = failed[col].dropna()
    w_vals = working[col].dropna()
    if len(f_vals) == 0 or len(w_vals) == 0:
        print(f"\n{col}: insufficient data for comparison")
        continue
    stat, p = stats.mannwhitneyu(f_vals, w_vals, alternative='two-sided')
    print(f"\n{col}:")
    print(f"  Failed:  mean={f_vals.mean():.6f}  median={f_vals.median():.6f}  n={len(f_vals)}")
    print(f"  Working: mean={w_vals.mean():.6f}  median={w_vals.median():.6f}  n={len(w_vals)}")
    print(f"  Mann-Whitney U={stat:.1f}  p={p:.4e}")

# Per-rat breakdown of failed vs working
print("\nPer-rat failed/working counts:")
for rat in ['R1', 'R2', 'R3']:
    sub = results_df[results_df.rat == rat]
    nf = ((sub['best_sl_f1'] == 0) | (sub['n_gt'] == 0)).sum()
    nw = ((sub['best_sl_f1'] > 0) & (sub['n_gt'] > 0)).sum()
    print(f"  {rat}: {nf} failed, {nw} working")

# ─────────────────────────────────────────────────────────────────────────────
# Per-keypoint MSE ranking (which body parts have highest error?)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("PER-KEYPOINT MSE RANKING (mean across all sessions)")
print("="*70)

mean_per_kp = per_kp_df.drop(columns=['rat','session']).mean()
sorted_kp = mean_per_kp.sort_values(ascending=False)
print(sorted_kp.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Save output CSV
# ─────────────────────────────────────────────────────────────────────────────

out_path = '/home/yutaka-sprague/CLIRB_analyses/results/sleap_dannce_error_analysis.csv'
save_cols = ['rat', 'session', 'keypoint_mse', 'pc_mse', 'best_sl_f1', 'n_gt', 'task', 'n_frames']
results_df[save_cols].to_csv(out_path, index=False)
print(f"\n\nResults saved to: {out_path}")
print(f"Total rows: {len(results_df)}")

# Also print the full table
print("\n" + "="*70)
print("FULL RESULTS TABLE (sorted by rat, session)")
print("="*70)
display_df = results_df[save_cols].sort_values(['rat','session'])
with pd.option_context('display.max_rows', 300, 'display.width', 120, 'display.float_format', '{:.6f}'.format):
    print(display_df.to_string(index=False))

print("\n\nDone.")
