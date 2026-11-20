"""Is fine-tuning a useful lever, or is the information simply not in the
images near the port?

Those two cases call for different responses and look identical in any
confidence statistic on its own.  They separate with a LEAVE-ONE-CAMERA-OUT
test, which needs no ground truth:

  for each keypoint and frame, where the OTHER two cameras are both confident,
  triangulate from those two alone, reproject into the held-out camera, and
  compare with what SLEAP actually detected there.

That gives a truth-free accuracy for the held-out detection, which can then be
binned by the held-out camera's OWN confidence:

  low confidence, SMALL leave-one-out error
      the detector found the right pixel and simply reported low confidence.
      The information IS in the image; a fine-tune would recover it, and the
      confidence-weighted triangulation is already down-weighting good data.
  low confidence, LARGE leave-one-out error
      the detection really is wrong -- the keypoint is occluded or outside the
      view.  No amount of training recovers what the camera cannot see; only a
      camera move does.

Reported separately near and far from the water port, so the obstruction's
contribution is visible.

Writes results/newcal/finetune_headroom.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import quality as ql
from projection import project_3d_to_2d
from config import NODE_IDX

OUT = 'results/newcal'
CONF_T = 0.5
NEAR, FAR = 150.0, 300.0
STRIDE = 12
BINS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01]
GOOD_PX = 15.0            # "the detector found the right pixel"
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
SESSIONS = [('R5', '2026_09_29_1'), ('R6', '2026_09_24_1'),
            ('R7', '2026_10_23_1'), ('R8', '2026_10_23_1'),
            ('R7', '2026_10_26_1'), ('R8', '2026_10_26_1'), ('R9', '2026_10_26_1')]


def pair_tri(und, P_list, pair):
    T = und.shape[0]; N = T * 23
    x = und[..., 0].transpose(0, 2, 1).reshape(N, 3)
    y = und[..., 1].transpose(0, 2, 1).reshape(N, 3)
    P = np.stack(P_list)
    A = np.empty((N, 4, 4))
    for k, c in enumerate(pair):
        A[:, 2 * k] = x[:, c, None] * P[c, 2][None, :] - P[c, 0][None, :]
        A[:, 2 * k + 1] = y[:, c, None] * P[c, 2][None, :] - P[c, 1][None, :]
    _, _, Vt = np.linalg.svd(A)
    h = Vt[:, -1, :]
    w = np.where(np.abs(h[:, 3]) < 1e-12, 1.0, h[:, 3])
    return (h[:, :3] / w[:, None]).reshape(T, 23, 3)


def main():
    rows = []
    for rat, session in SESSIONS:
        raw = sc.get_raw_session(rat, session, want_dual=True)
        sl = np.asarray(raw['triang_3d'], np.float32)
        k2 = np.asarray(raw['sleap_keys_2D'])
        fm = np.asarray(raw['frame_mapping_cam0'], int)
        nproc = min(len(fm), len(k2))
        port = np.asarray(raw['port_xy']).ravel()[:2]
        com = np.median(sl, axis=1)[:, :2]
        dist_acq = np.hypot(com[:, 0] - port[0], com[:, 1] - port[1])
        fmp = fm[:nproc]
        okf = (fmp >= 0) & (fmp < len(sl))
        dproc = np.full(nproc, np.nan)
        dproc[okf] = dist_acq[fmp[okf]]

        sub = np.arange(0, nproc, STRIDE)
        P, K, D, R, T = ql.load_calib(ql.calib_dir(rat, session))
        xy = k2[sub][..., :2].astype(np.float64)
        conf = k2[sub][..., 2].astype(np.float64)
        und = ql.undistort_all(xy, K, D)
        dsub = dproc[sub]

        for held in range(3):
            others = [c for c in range(3) if c != held]
            X = pair_tri(und, P, tuple(others))                 # (T,23,3)
            flat = X.reshape(-1, 3)
            fin = np.isfinite(flat).all(axis=1)
            pr = np.full((flat.shape[0], 2), np.nan)
            pr[fin] = project_3d_to_2d(flat[fin], K[held], R[held], T[held], D[held])
            pr = pr.reshape(X.shape[0], 23, 2)
            err = np.linalg.norm(pr - xy[:, held], axis=-1)     # (T,23)
            usable = (conf[:, others[0]] >= CONF_T) & (conf[:, others[1]] >= CONF_T)
            for lab, m0 in [('near port', dsub < NEAR), ('far from port', dsub > FAR)]:
                m0 = m0 & np.isfinite(dsub)
                for b in range(len(BINS) - 1):
                    sel = (usable & m0[:, None]
                           & (conf[:, held] >= BINS[b]) & (conf[:, held] < BINS[b + 1]))
                    if sel.sum() < 200:
                        continue
                    e = err[sel]
                    rows.append(dict(
                        rat=rat, session=session,
                        era=('Sep' if session.startswith('2026_09') else
                             ('Oct26' if '10_26' in session else 'Oct23')),
                        held_cam=held, stratum=lab,
                        conf_lo=BINS[b], conf_hi=BINS[b + 1], n=int(sel.sum()),
                        loo_err_med=float(np.nanmedian(e)),
                        loo_err_p90=float(np.nanpercentile(e, 90)),
                        frac_good=float(np.nanmean(e < GOOD_PX))))
        print(f'{rat}/{session} done', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/finetune_headroom.csv', index=False)
    pd.set_option('display.width', 220)
    print('\n=== leave-one-camera-out error of the held-out detection, '
          'by its own confidence ===')
    g = (d.groupby(['era', 'stratum', 'conf_lo'])
          .apply(lambda x: pd.Series({
              'n': x.n.sum(),
              'loo_err_med': np.average(x.loo_err_med, weights=x.n),
              'frac_good_<15px': np.average(x.frac_good, weights=x.n)}),
              include_groups=False)
          .reset_index())
    print(g.round(3).to_string(index=False))
    print('\n=== recoverable vs genuinely lost, for detections below conf 0.5 ===')
    lo = d[d.conf_hi <= CONF_T]
    for (era, stratum), x in lo.groupby(['era', 'stratum']):
        fg = np.average(x.frac_good, weights=x.n)
        print(f'  {era:6s} {stratum:14s}: {x.n.sum():7d} low-confidence '
              f'keypoint-views, {100*fg:5.1f}% land within {GOOD_PX:.0f} px of '
              f'where the other two cameras say they should be')
    print(f'\nwrote {OUT}/finetune_headroom.csv')


if __name__ == '__main__':
    main()
