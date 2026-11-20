"""How much does SLEAP's OWN tracking accuracy vary across arena sectors?

This deliberately avoids DANNCE for the primary measures.  The previous
appendices established that DANNCE collapses in the water-port sector, so any
per-sector SLEAP-vs-DANNCE disagreement there would blame SLEAP for DANNCE's
failure.  Every headline number here is truth-free -- computed from SLEAP's own
2D detections, its own 3D, and the camera geometry:

  reproj_px      median pixel distance between the stored 3D pose reprojected
                 into each camera and SLEAP's own 2D detection there
  pair_spread    disagreement between the three two-camera triangulations of
                 the same keypoint, on frames where all three cameras are
                 confident -- the sharpest available probe of whether the views
                 agree with each other
  conf           mean 2D detection confidence, and the mean number of cameras
                 clearing 0.5
  bone_cv        per-frame bone-length coefficient of variation (rigidity)
  offimage_frac  how often the highest keypoint projects outside an image

The SLEAP-vs-DANNCE world RMSE is carried along for contrast ONLY, explicitly
flagged, so the port sector's split between the two systems is visible.

FRAME BASES.  `sleap_keys_2D.npy` and everything derived from it are on the
PROCESSED-frame basis; `triang_keys_3D.npy`, the COM and therefore the sector
assignment are on the CAM0 ACQUISITION basis.  frame_mapping.csv's cam0_frame
bridges them (see clirb/spatial/quality.py).  Sectors are computed once on the
acquisition grid and indexed through that bridge, never recomputed.

ARENA FRAME.  The arena is a physical object, so the frame it defines is fitted
from whichever COM track traces the perimeter best -- DANNCE's, pooled per era,
exactly as scripts/newcal_sector_scale_bias.py does.  That choice does not
contaminate the SLEAP-only measures: it fixes a coordinate system, nothing
more.  Two fits had to be guarded.  SLEAP's own COM gives a bad circle because
fly-away keypoints drag the mean (and the median is not enough), and the Oct-26
sessions park the animal at the port for 61 % of the time, so their own COM
barely spans the perimeter: a fit with residual over 15 mm is replaced by the
arena of the nearest well-fitted era, which for Oct 26 is Oct 22-23 -- the same
room, since only the calibration and the task changed between them.  The port
sector is read from each session's water_port_location.npy.

Writes results/newcal/sleap_sector_quality.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import dual as du
from clirb.spatial import quality as ql
from clirb.spatial.geometry import fit_arena, sector_index
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample
from scripts.newcal_triangulation_consistency import pair_triangulate
from config import NODE_IDX

OUT = 'results/newcal'
SECTOR_W = 60
CONF_T = 0.5
STRIDE = 40          # for the pairwise-triangulation probe
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]


def era_arena(coms):
    P = np.concatenate(coms)
    P = P[np.isfinite(P).all(axis=1)]
    c0 = np.median(P, axis=0)
    r = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
    a, cost = fit_arena(P[r <= np.percentile(r, 99.5)])
    return a, float(cost)


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    sept = inv[inv.era == 'sept_baseline'].sort_values('session').tail(9)
    use = pd.concat([sept, inv[inv.era.isin(['oct_b', 'oct_26'])]])

    coms = {}
    for r in use.itertuples():
        raw = sc.get_raw_session(r.rat, r.session, want_dual=True)
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        coms.setdefault(r.era, []).append(
            np.asarray(d3, np.float32).mean(axis=1)[::10, :2])
    arenas, costs = {}, {}
    for e, v in coms.items():
        a, c = era_arena(v)
        arenas[e], costs[e] = a, c
        print(f'{e}: arena ({a.cx:.0f},{a.cy:.0f}) r={a.radius:.0f} cost={c:.1f}mm',
              flush=True)
    GUARD = 15.0
    good = [e for e in arenas if costs[e] <= GUARD]
    for e in list(arenas):
        if costs[e] > GUARD and good:
            sub = 'oct_b' if ('oct_b' in good and e == 'oct_26') else good[0]
            print(f'  {e}: fit residual {costs[e]:.1f}mm > {GUARD}mm '
                  f'(the animal does not span the perimeter) -- using {sub}\'s arena',
                  flush=True)
            arenas[e] = arenas[sub]

    rows = []
    for i, r in enumerate(use.itertuples(), 1):
        raw = sc.get_raw_session(r.rat, r.session, want_dual=True)
        sl = np.asarray(raw['triang_3d'], np.float32)
        n = len(sl)
        arena = arenas[r.era]
        com = np.median(sl, axis=1)[:, :2]
        sec_acq = sector_index(arena.theta(com), SECTOR_W)
        port = np.asarray(raw['port_xy']).ravel()[:2]
        port_sec = (int(sector_index(np.atleast_1d(arena.theta(port[None, :]))[0], SECTOR_W))
                    if np.isfinite(port).all() else -1)

        bcv = ql.bone_length_cv(sl)                       # acquisition basis
        cal_dir = ql.calib_dir(r.rat, r.session)
        k2 = np.asarray(raw['sleap_keys_2D'])
        fm = np.asarray(raw['frame_mapping_cam0'], dtype=int)
        nproc = min(len(fm), len(k2))
        fmp = fm[:nproc]
        ok = (fmp >= 0) & (fmp < n)
        sec_proc = np.full(nproc, -1, int)
        sec_proc[ok] = sec_acq[fmp[ok]]

        resid, per_cam, ncam, fov = ql.reprojection_residual_aligned(
            k2[:nproc], sl, fmp, cal_dir)
        conf = k2[:nproc, :, :, 2].astype(np.float64)

        # pairwise two-camera triangulation spread, subsampled
        sub = np.arange(0, nproc, STRIDE)
        P, K, D, R, T = ql.load_calib(cal_dir)
        und = ql.undistort_all(k2[sub][..., :2].astype(np.float64), K, D)
        pts = [pair_triangulate(und, P, p) for p in [(0, 1), (0, 2), (1, 2)]]
        spread = np.nanmax(np.stack([np.linalg.norm(pts[0] - pts[1], axis=-1),
                                     np.linalg.norm(pts[0] - pts[2], axis=-1),
                                     np.linalg.norm(pts[1] - pts[2], axis=-1)]), axis=0)
        allconf = (conf[sub] >= CONF_T).all(axis=1)       # (len(sub), 23)
        sec_sub = sec_proc[sub]

        # DANNCE contrast only
        d3 = du.dannce_aligned(raw)
        frmse = np.full(n, np.nan)
        if d3 is not None:
            m = min(n, len(d3))
            dn = np.asarray(d3[:m], np.float32)
            samp, _ = balanced_sample(sl[:m].mean(axis=1)[:, :2],
                                      rng=np.random.default_rng(0))
            if len(samp) >= 500:
                fit = robust_proc(dn[samp], sl[samp])
                frmse[:m] = np.sqrt((((sl[:m] - fit['apply'](dn)) ** 2)
                                     .sum(axis=-1)).mean(axis=1))

        for k in range(int(360 // SECTOR_W)):
            ma, mp = sec_acq == k, sec_proc == k
            if ma.sum() < 300 or mp.sum() < 100:
                continue
            ms = sec_sub == k
            sp = spread[ms][allconf[ms]] if ms.any() else np.array([])
            with np.errstate(all='ignore'):
                rows.append(dict(
                    rat=r.rat, session=r.session, era=r.era, sector=k,
                    centre_deg=-180 + SECTOR_W * (k + 0.5),
                    is_port_sector=(k == port_sec),
                    occ=float(ma.mean()), n_frames=int(ma.sum()),
                    reproj_px=float(np.nanmedian(resid[mp])),
                    reproj_px_p90=float(np.nanpercentile(resid[mp], 90)),
                    conf_mean=float(np.nanmean(conf[mp])),
                    conf_lo_frac=float(np.nanmean(conf[mp] < CONF_T)),
                    ncam_conf=float(np.nanmean(ncam[mp])),
                    offimage_frac=float(np.nanmean(np.nanmin(fov[mp], axis=1) < 0)),
                    bone_cv=float(np.nanmedian(bcv[ma])),
                    pair_spread_mm=float(np.nanmedian(sp)) if len(sp) > 200 else np.nan,
                    dannce_rmse_mm=float(np.nanmedian(frmse[ma]))))
        print(f'[{i}/{len(use)}] {r.rat}/{r.session} ({r.era}) port sector={port_sec}',
              flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/sleap_sector_quality.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['reproj_px', 'pair_spread_mm', 'conf_mean', 'ncam_conf', 'bone_cv',
            'offimage_frac', 'dannce_rmse_mm']
    for e in ['sept_baseline', 'oct_b', 'oct_26']:
        s = d[d.era == e]
        if not len(s):
            continue
        g = s.groupby('sector').agg(centre=('centre_deg', 'first'),
                                    port=('is_port_sector', 'max'),
                                    occ=('occ', 'median'),
                                    **{c: (c, 'median') for c in cols}).reset_index()
        print(f'\n=== {e} — SLEAP quality by {SECTOR_W}° sector '
              f'({s.groupby(["rat", "session"]).ngroups} sessions) ===')
        print(g.round(3).to_string(index=False))
        print('   spread across sectors (max/min): ' + '  '.join(
            f'{c}={g[c].max()/g[c].min():.2f}x' for c in cols if g[c].min() > 0))
        pm, om = s[s.is_port_sector], s[~s.is_port_sector]
        if len(pm) and len(om):
            print('   port sector vs elsewhere: ' + '  '.join(
                f'{c} {pm[c].median():.3f} vs {om[c].median():.3f}' for c in cols))
    print(f'\nwrote {OUT}/sleap_sector_quality.csv')


if __name__ == '__main__':
    main()
