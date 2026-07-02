"""Diagnose 'other' matches — offline-detected matches that didn't fire online
and weren't excluded by the port-distance gate.

The authoritative source for what the online rule actually saw is
behavior_log.csv: it records the PC values the rule computed at every
processed frame. So for each "other" match (offline replay said: this should
have fired), we look up the *online-computed* PC values around the
match-completion frame and re-run the rule's match-checking logic on those
values. The result tells us, frame-by-frame, why the online rule did or
didn't fire on that template window:

  - Recomputed_num_outside <= accepted_errors and not in port radius
    → the online rule SHOULD have fired here; if it didn't, blame refractory.
  - Recomputed_num_outside <= accepted_errors but in port radius
    → online check_time=False suppressed it (would have been port_excluded
      under the offline classifier; the offline replay misclassified it
      because its PC values disagreed with the online rule).
  - Recomputed_num_outside > accepted_errors
    → the online rule never saw this window as a match. The offline replay
      flagged it because its PC values diverged slightly from the online
      rule's. This is the "offline replay artifact" category.

Usage:
    python scripts/diagnose_offline_only.py --rat R4 --session 2026_06_18_1
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from scripts.render_rear_compilation import (load_and_analyze, SLEAP_HZ,    # noqa: E402
                                              TEMPLATE_LEN_FRAMES)


def diagnose(rat, session):
    info = load_and_analyze(rat, session)
    matches = info['matches_cam0']
    classes = info['match_classes']
    other_idx = np.where(classes == 'other')[0]
    if len(other_idx) == 0:
        print(f'{rat}/{session}: no \"other\" matches to diagnose.')
        return
    print(f'\n=== {rat}/{session} — {len(other_idx)} \"other\" matches ===')

    sleap_dir = info['sleap_dir']
    log = pd.read_csv(os.path.join(sleap_dir, 'behavior_log.csv'),
                       skipinitialspace=True)
    fm = pd.read_csv(os.path.join(sleap_dir, 'frame_mapping.csv'),
                       skipinitialspace=True)
    pf_to_cam0 = fm['cam0_frame'].to_numpy(dtype=int)
    cam0_to_pf = {int(c): i for i, c in enumerate(pf_to_cam0)}

    # Load template for bounds + accepted_errors.
    import glob as _g
    cands = sorted(_g.glob(os.path.join(sleap_dir, f'{rat}_template_*.npz')))
    tmpl_path = cands[0]
    tmpl = np.load(tmpl_path, allow_pickle=True)
    pcs_to_use = np.asarray(tmpl['pcs_to_use']).ravel().astype(int)
    template_pc = tmpl['template'][:, :len(pcs_to_use)]
    if 'bounds_to_use' in tmpl.files:
        bounds_arr = tmpl['bounds_to_use']
    else:
        scalar = float(tmpl['bounds'])
        bounds_arr = np.tile(tmpl['feature_stds'][pcs_to_use] * scalar,
                              (template_pc.shape[0], 1))
    accepted_errors = 3   # matches the rig's campy_config

    online_pf = log.loc[log['triggered'] == 1, 'frame'].to_numpy(dtype=int)
    online_pf = online_pf[online_pf < len(pf_to_cam0)]
    online_cam0_sorted = np.sort(pf_to_cam0[online_pf])

    # Per-processed-frame PC1/PC2 from the online rule. We assume PC1 in the
    # log = pcs_to_use[0], PC2 = pcs_to_use[1] (the TemplateLogger writes
    # exactly these two and the rule uses pcs_to_use to index them).
    log_pc = log[['PC1', 'PC2']].to_numpy(dtype=float)   # (n_proc, 2)

    port_dist = info['port_dist']
    min_port = info['min_port']
    refractory_frames = TEMPLATE_LEN_FRAMES

    rows = []
    for k in other_idx:
        match_end = int(matches[k])
        win_start = match_end - (TEMPLATE_LEN_FRAMES - 1)
        # Build the 30-frame template window in ONLINE PC space.
        online_window_pcs = np.full((TEMPLATE_LEN_FRAMES, 2), np.nan)
        online_window_present = 0
        online_in_port = 0
        for j, cam0_f in enumerate(range(win_start, match_end + 1)):
            pf = cam0_to_pf.get(int(cam0_f))
            if pf is not None and pf < len(log_pc):
                online_window_pcs[j] = log_pc[pf]
                online_window_present += 1
            d = port_dist[cam0_f] if 0 <= cam0_f < len(port_dist) else np.nan
            if np.isfinite(d) and d < min_port:
                online_in_port += 1
        # Compute num_outside using ONLINE PCs (only over frames present;
        # missing frames are an automatic 'outside' since we have no data).
        if online_window_present > 0:
            above = online_window_pcs >= (template_pc + bounds_arr)
            below = online_window_pcs <= (template_pc - bounds_arr)
            # Treat missing frames (NaN) as 'outside' to be conservative.
            missing = ~np.isfinite(online_window_pcs)
            outside_mask = (above | below | missing)
            online_num_outside = int(outside_mask.sum())
        else:
            online_num_outside = TEMPLATE_LEN_FRAMES * 2

        # Check refractory clash with online triggers.
        lo = np.searchsorted(online_cam0_sorted, match_end - refractory_frames, side='left')
        hi = np.searchsorted(online_cam0_sorted, match_end, side='left')
        clashing = online_cam0_sorted[lo:hi]

        # Port distance at match-end frame (controls the live check_time gate).
        dist_at_end = port_dist[match_end] if match_end < len(port_dist) else np.nan
        port_excluded_at_end = np.isfinite(dist_at_end) and dist_at_end < min_port

        # Classify.
        if online_num_outside > accepted_errors:
            reason = (f'offline-replay artifact: online rule saw '
                       f'num_outside={online_num_outside} (> {accepted_errors}); '
                       f'this window was never a match online')
        elif port_excluded_at_end:
            reason = (f'port-excluded online: would have matched '
                       f'(online num_outside={online_num_outside}) but '
                       f'dist_at_end={dist_at_end:.0f}mm < {min_port:.0f}mm')
        elif len(clashing) > 0:
            reason = (f'refractory clash: would have matched online '
                       f'(num_outside={online_num_outside}) but online '
                       f'triggered at cam0={int(clashing[-1])} '
                       f'({match_end - int(clashing[-1])} frames before)')
        else:
            reason = (f'unexplained: online num_outside={online_num_outside}, '
                       f'no port suppression, no refractory clash')

        rows.append({
            'match_end_cam0': match_end,
            'session_time': f'{match_end / SLEAP_HZ / 60:.0f}m'
                              f'{(match_end / SLEAP_HZ) % 60:.1f}s',
            'win_frames_in_log': f'{online_window_present}/{TEMPLATE_LEN_FRAMES}',
            'online_num_outside': online_num_outside,
            'dist_at_end_mm': round(float(dist_at_end), 1) if np.isfinite(dist_at_end) else np.nan,
            'win_frames_in_port': online_in_port,
            'reason': reason,
        })

    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.width', 240)
    print(df.to_string(index=False))
    print()
    from collections import Counter
    tally = Counter()
    for row in rows:
        key = row['reason'].split(':', 1)[0]
        tally[key] += 1
    print('Reason tally:')
    for r, n in sorted(tally.items(), key=lambda x: -x[1]):
        print(f'  {r}: {n}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rat', action='append', required=True)
    ap.add_argument('--session', required=True)
    args = ap.parse_args()
    for rat in args.rat:
        diagnose(rat, args.session)


if __name__ == '__main__':
    main()
