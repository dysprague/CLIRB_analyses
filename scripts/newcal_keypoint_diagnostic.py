"""Why did SLEAP get worse? Per-keypoint 2D-level diagnostic.

The world-space ranking (F3) shows every keypoint roughly 5x worse after the
environment change, and SpineM ~20x worse with a ~38% fly-away rate.  SpineM is
also the origin of the egocentric normalisation the online rule uses, so if it
fails the entire PC representation fails with it -- which is exactly what F4
shows.

This script asks, per keypoint, whether the failure is already present in
SLEAP's 2D detections or appears only in the 3D triangulation:

  conf_mean / conf_lo_frac : SLEAP's own per-camera 2D detection confidence
  reproj_px                : median pixel distance between the stored 3D pose
                             reprojected into each camera and the 2D detection
                             the network produced for that camera
  n_cam_conf               : how many of the 3 cameras clear conf >= 0.5

and, for the fly-away frames specifically, how the above differ from the rest.

Runs on a matched subset: every post-recalibration session plus the last few
clean September sessions per rat.
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import dual as du
from clirb.spatial import quality as ql
from config import NODE_IDX
from scripts.newcal_eras import ERAS, ERA_SHORT
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample

OUT = 'results/newcal'
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
CONF_T = 0.5



def session_rows(rat, session, era):
    raw = sc.get_raw_session(rat, session, want_dual=True)
    d3 = du.dannce_aligned(raw)
    n = min(len(raw['triang_3d']), len(d3))
    sl = np.asarray(raw['triang_3d'][:n], dtype=np.float32)
    dn = np.asarray(d3[:n], dtype=np.float32)
    samp, _ = balanced_sample(sl.mean(axis=1)[:, :2], rng=np.random.default_rng(0))
    fit = robust_proc(dn[samp], sl[samp])
    dist = np.linalg.norm(sl - fit['apply'](dn), axis=-1)       # (n, 23) mm
    fly = dist > 100.0

    k2 = np.asarray(raw['sleap_keys_2D'])
    fm = np.asarray(raw['frame_mapping_cam0'], dtype=int)
    nproc = min(len(fm), len(k2))
    cal_dir = ql.calib_dir(rat, session)

    # Per-keypoint, per-camera reprojection error of the STORED 3D against the
    # 2D detections, on the processed-frame basis (quality.py's frame bridge).
    from projection import project_3d_to_2d
    P, K, D, R, T = ql.load_calib(cal_dir)
    fmk = fm[:nproc]
    ok = (fmk >= 0) & (fmk < n)
    X = np.full((nproc, 23, 3), np.nan)
    X[ok] = sl[fmk[ok]]
    det = k2[:nproc, :, :, :2].astype(np.float64)
    conf = k2[:nproc, :, :, 2].astype(np.float64)
    err = np.full((nproc, 3, 23), np.nan)
    flat = X.reshape(-1, 3); fin = np.isfinite(flat).all(axis=1)
    for cam in range(3):
        pr = np.full((flat.shape[0], 2), np.nan)
        pr[fin] = project_3d_to_2d(flat[fin], K[cam], R[cam], T[cam], D[cam])
        pr = pr.reshape(nproc, 23, 2)
        e = np.linalg.norm(pr - det[:, cam], axis=-1)
        e[conf[:, cam] < CONF_T] = np.nan
        err[:, cam] = e

    # fly-away flag on the processed-frame basis
    fly_proc = np.zeros((nproc, 23), dtype=bool)
    fly_proc[ok] = fly[fmk[ok]]

    rows = []
    with np.errstate(all='ignore'):
        for k, name in enumerate(KP_NAMES):
            c = conf[:, :, k]                      # (nproc, 3)
            e = err[:, :, k]
            n_ok = (c >= CONF_T).sum(axis=1)
            f = fly_proc[:, k]
            rows.append(dict(
                rat=rat, session=session, era=era, keypoint=name,
                rmse_mm=float(np.sqrt((dist[:, k] ** 2).mean())),
                med_mm=float(np.median(dist[:, k])),
                fly_frac=float(fly[:, k].mean()),
                conf_mean=float(np.nanmean(c)),
                conf_lo_frac=float(np.nanmean(c < CONF_T)),
                n_cam_conf_mean=float(n_ok.mean()),
                frac_lt2_cam=float((n_ok < 2).mean()),
                reproj_px=float(np.nanmedian(e)),
                conf_mean_fly=float(np.nanmean(c[f])) if f.any() else np.nan,
                conf_mean_ok=float(np.nanmean(c[~f])) if (~f).any() else np.nan,
                n_cam_conf_fly=float(n_ok[f].mean()) if f.any() else np.nan,
                n_cam_conf_ok=float(n_ok[~f].mean()) if (~f).any() else np.nan,
                reproj_px_fly=float(np.nanmedian(e[f])) if f.any() else np.nan,
                reproj_px_ok=float(np.nanmedian(e[~f])) if (~f).any() else np.nan,
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
    d.to_csv(f'{OUT}/keypoint_2d_diagnostic.csv', index=False)

    agg = d.groupby(['era', 'keypoint']).median(numeric_only=True).reset_index()
    piv = agg.pivot(index='keypoint', columns='era')
    cols = ['rmse_mm', 'fly_frac', 'conf_mean', 'conf_lo_frac',
            'n_cam_conf_mean', 'reproj_px']
    t = pd.DataFrame(index=piv.index)
    for c in cols:
        for e in ERAS:
            if (c, e) in piv.columns:
                t[f'{c}_{ERA_SHORT[e]}'] = piv[(c, e)]
    last = f'rmse_mm_{ERA_SHORT[ERAS[-1]]}'
    t = t.sort_values(last if last in t else t.columns[0], ascending=False)
    with pd.option_context('display.width', 260, 'display.max_columns', 40,
                           'display.float_format', '{:.3f}'.format):
        n = d.groupby('era').apply(lambda g: g.groupby(['rat', 'session']).ngroups,
                                   include_groups=False).to_dict()
        print(f'\nPer-keypoint, median across sessions {n}:')
        print(t.to_string())

    print(f'\nFly-away vs non-fly-away frames, {ERA_SHORT[ERAS[-1]]} sessions only:')
    o = d[d.era == ERAS[-1]].groupby('keypoint').median(numeric_only=True)
    o = o[['fly_frac', 'conf_mean_ok', 'conf_mean_fly',
           'n_cam_conf_ok', 'n_cam_conf_fly', 'reproj_px_ok', 'reproj_px_fly']]
    with pd.option_context('display.width', 220, 'display.float_format', '{:.3f}'.format):
        print(o.sort_values('fly_frac', ascending=False).to_string())
    print(f'\nwrote {OUT}/keypoint_2d_diagnostic.csv')


if __name__ == '__main__':
    main()
