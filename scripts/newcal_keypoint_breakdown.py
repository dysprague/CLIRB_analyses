"""Per-keypoint breakdown of the September-vs-post-recalibration comparison.

The headline session numbers are RMSE, which a handful of fly-away frames can
dominate.  This pass recomputes, for EVERY session and EVERY keypoint, both an
outlier-sensitive and an outlier-robust measure, in the two spaces that matter:

  world space    -- DANNCE mapped into SLEAP's frame by the same session-level
                    Procrustes fit used everywhere else (stage20 / section 8q).
                    This is "how far apart are the two systems' estimates of
                    this body part".
  egocentric     -- after z-flip (SLEAP) + normalize_skeleton + 5-frame causal
                    median, i.e. the representation the online rule consumes.
                    SpineM is 0 by construction here and is reported as NaN.

Per keypoint, per session:
    med_mm   median Euclidean error over frames   (robust: "typical" error)
    p90_mm   90th percentile                      (how bad the bad frames get)
    rmse_mm  root-mean-square                     (what the headline uses)
    fly_frac fraction of frames more than 100 mm apart

Writes results/newcal/keypoint_breakdown.csv (one row per session x keypoint).
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb.spatial import dual as du
from config import NODE_IDX
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample, COMMON_TEMPLATE

OUT = 'results/newcal'
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
SPINEM = NODE_IDX['SpineM']


def stats(dist):
    """dist (n_frames, 23) Euclidean error -> dict of per-keypoint arrays."""
    return dict(
        med=np.median(dist, axis=0),
        p90=np.percentile(dist, 90, axis=0),
        rmse=np.sqrt((dist ** 2).mean(axis=0)),
        fly=(dist > 100.0).mean(axis=0),
    )


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    common = dict(np.load(COMMON_TEMPLATE, allow_pickle=True))
    rng = np.random.default_rng(0)

    rows = []
    for i, r in enumerate(inv.itertuples(), 1):
        raw = sc.get_raw_session(r.rat, r.session, want_dual=True)
        if raw is None or not raw.get('has_dual', False):
            continue
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        n = min(len(raw['triang_3d']), len(d3))
        sl = np.asarray(raw['triang_3d'][:n], dtype=np.float32)
        dn = np.asarray(d3[:n], dtype=np.float32)
        samp, _ = balanced_sample(sl.mean(axis=1)[:, :2], rng=rng)
        if len(samp) < 500:
            continue
        fit = robust_proc(dn[samp], sl[samp])
        w = stats(np.linalg.norm(sl - fit['apply'](dn), axis=-1))

        _, s_rot = ev.project_session_to_pcs(sl, common, flip_z=True)
        _, d_rot = ev.project_session_to_pcs(dn, common, flip_z=False)
        e = stats(np.linalg.norm(s_rot - d_rot, axis=-1))

        for k, name in enumerate(KP_NAMES):
            rows.append(dict(
                rat=r.rat, session=r.session, era=r.era, keypoint=name,
                n_frames=int(n),
                world_med_mm=float(w['med'][k]), world_p90_mm=float(w['p90'][k]),
                world_rmse_mm=float(w['rmse'][k]), world_fly_frac=float(w['fly'][k]),
                ego_med_mm=np.nan if k == SPINEM else float(e['med'][k]),
                ego_rmse_mm=np.nan if k == SPINEM else float(e['rmse'][k]),
                ego_fly_frac=np.nan if k == SPINEM else float(e['fly'][k]),
            ))
        print(f'[{i}/{len(inv)}] {r.rat}/{r.session}', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/keypoint_breakdown.csv', index=False)
    print(f'\nwrote {OUT}/keypoint_breakdown.csv '
          f'({d.session.nunique()} sessions x {len(KP_NAMES)} keypoints)')


if __name__ == '__main__':
    main()
