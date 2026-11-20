"""Is the post-recalibration SLEAP failure a BAD CALIBRATION or a BAD DETECTOR?

Decisive test, per keypoint, using only SLEAP's own data (no DANNCE):

Triangulate each keypoint from each PAIR of SLEAP cameras (0-1, 0-2, 1-2) and
measure how far the three pairwise estimates fall from one another, restricted
to frames where ALL THREE cameras report confidence >= 0.5 for that keypoint.

  * If the camera CALIBRATION were wrong, every keypoint's pairwise estimates
    would disagree -- the geometry is shared by all of them.
  * If the 2D DETECTOR is placing a keypoint at a different body location in
    different views, only that keypoint's pairwise estimates disagree.

Writes results/newcal/triangulation_consistency.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import quality as ql
from config import NODE_IDX
from scripts.newcal_eras import ERAS, ERA_SHORT

OUT = 'results/newcal'
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
CONF_T = 0.5
STRIDE = 20



def pair_triangulate(und, P_list, pair):
    """DLT from two cameras. und (T,3,23,2) normalized undistorted."""
    T = und.shape[0]; N = T * 23
    x = und[..., 0].transpose(0, 2, 1).reshape(N, 3)
    y = und[..., 1].transpose(0, 2, 1).reshape(N, 3)
    P = np.stack(P_list)
    A = np.empty((N, 4, 4))
    for k, c in enumerate(pair):
        A[:, 2 * k]     = x[:, c, None] * P[c, 2][None, :] - P[c, 0][None, :]
        A[:, 2 * k + 1] = y[:, c, None] * P[c, 2][None, :] - P[c, 1][None, :]
    _, _, Vt = np.linalg.svd(A)
    h = Vt[:, -1, :]
    w = np.where(np.abs(h[:, 3]) < 1e-12, 1.0, h[:, 3])
    return (h[:, :3] / w[:, None]).reshape(T, 23, 3)


def session_rows(rat, session, era):
    raw = sc.get_raw_session(rat, session, want_dual=False)
    k2 = np.asarray(raw['sleap_keys_2D'])
    fm = np.asarray(raw['frame_mapping_cam0'], dtype=int)
    n = min(len(fm), len(k2))
    k2 = k2[:n].astype(np.float64)
    sub = np.arange(0, n, STRIDE)
    k2s = k2[sub]
    conf = k2s[..., 2]
    P, K, D, R, T = ql.load_calib(ql.calib_dir(rat, session))
    und = ql.undistort_all(k2s[..., :2], K, D)
    pts = [pair_triangulate(und, P, p) for p in [(0, 1), (0, 2), (1, 2)]]
    spread = np.nanmax(np.stack([
        np.linalg.norm(pts[0] - pts[1], axis=-1),
        np.linalg.norm(pts[0] - pts[2], axis=-1),
        np.linalg.norm(pts[1] - pts[2], axis=-1)]), axis=0)      # (T, 23)
    good = (conf >= CONF_T).all(axis=1)                          # (T, 23)
    rows = []
    for k, name in enumerate(KP_NAMES):
        g = good[:, k]
        rows.append(dict(
            rat=rat, session=session, era=era, keypoint=name,
            n_all3_conf=int(g.sum()), n_eval=int(len(sub)),
            frac_all3_conf=float(g.mean()),
            pair_spread_mm=float(np.nanmedian(spread[g, k])) if g.sum() > 200 else np.nan,
            pair_spread_p90_mm=float(np.nanpercentile(spread[g, k], 90)) if g.sum() > 200 else np.nan,
        ))
    return rows


def pick_sessions():
    """Session subset: the last 6 clean September sessions plus every
    post-recalibration session (read from the era inventory, so new days are
    picked up automatically)."""
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    sept = inv[inv.era == 'sept_baseline'].sort_values('session').tail(6)
    post = inv[inv.era != 'sept_baseline'].sort_values(['era', 'rat', 'session'])
    return [(r.rat, r.session, r.era) for r in sept.itertuples()] + \
           [(r.rat, r.session, r.era) for r in post.itertuples()]

def main():
    rows = []
    for rat, ses, era in pick_sessions():
        try:
            rows += session_rows(rat, ses, era)
            print(f'  {rat}/{ses} ({era}) done', flush=True)
        except Exception as e:
            print(f'  {rat}/{ses}: ERROR {e!r}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/triangulation_consistency.csv', index=False)
    t = (d.groupby(['era', 'keypoint']).pair_spread_mm.median()
          .unstack(0).reindex(columns=ERAS)
          .rename(columns=ERA_SHORT))
    base = ERA_SHORT['sept_baseline']
    for e in ERAS[1:]:
        t[f'x{ERA_SHORT[e]}'] = t[ERA_SHORT[e]] / t[base]
    with pd.option_context('display.float_format', '{:.1f}'.format,
                           'display.width', 200):
        print('\nMedian max-pairwise-triangulation disagreement (mm), '
              'frames with all 3 cameras confident:')
        print(t.sort_values(ERA_SHORT[ERAS[-1]], ascending=False).to_string())
    print(f'\nwrote {OUT}/triangulation_consistency.csv')


if __name__ == '__main__':
    main()
