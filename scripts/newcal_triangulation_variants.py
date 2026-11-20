"""Can the port-sector damage be reduced in software alone?

The per-sector result showed that near the port SLEAP typically has only ~1.15
of 3 cameras reporting a confident detection, yet `triang_keys_3D` is produced
by a plain 3-camera DLT that weights every view equally.  That throws away the
confidence SLEAP already outputs.  This tests whether using it helps, and --
just as important -- measures the floor below which no triangulation scheme can
help, because fewer than two usable views leaves the point unconstrained.

Variants, all re-triangulated from the saved `sleap_keys_2D`:
  equal        plain 3-camera DLT -- reproduces what the rig saved
  weighted     DLT with each camera's two rows scaled by its confidence
  best2        DLT from the two highest-confidence cameras only
  gated        `weighted`, but NaN wherever fewer than two cameras clear 0.5

Scored with truth-free measures, because DANNCE is itself unreliable in the
region under test:
  bone_cv   per-frame bone-length coefficient of variation (rigidity)
  jitter    mean |2nd time-difference| across keypoints (mm/frame^2)
  reproj    median pixel residual against the 2D detections -- reported but
            partly circular for `equal`/`weighted`, so not used to rank

Writes results/newcal/triangulation_variants.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import quality as ql
from config import EDGES

OUT = 'results/newcal'
CONF_T = 0.5
NEAR, FAR = 150.0, 300.0
N_CAMS, N_KP = 3, 23
SESSIONS = [('R7', '2026_10_26_1'), ('R8', '2026_10_26_1'), ('R9', '2026_10_26_1'),
            ('R7', '2026_10_23_1'), ('R8', '2026_10_23_1'), ('R9', '2026_10_23_1')]


def dlt(und, P_list, w):
    """Weighted DLT. und (T,3,23,2) normalised; w (T,3,23) row weights."""
    T = und.shape[0]
    N = T * N_KP
    x = und[..., 0].transpose(0, 2, 1).reshape(N, N_CAMS)
    y = und[..., 1].transpose(0, 2, 1).reshape(N, N_CAMS)
    ww = w.transpose(0, 2, 1).reshape(N, N_CAMS)
    P = np.stack(P_list)
    A = np.empty((N, 2 * N_CAMS, 4))
    for i in range(N_CAMS):
        A[:, 2 * i] = (x[:, i, None] * P[i, 2][None, :] - P[i, 0][None, :]) * ww[:, i, None]
        A[:, 2 * i + 1] = (y[:, i, None] * P[i, 2][None, :] - P[i, 1][None, :]) * ww[:, i, None]
    out = np.full((N, 3), np.nan)
    ok = (ww > 0).sum(axis=1) >= 2
    if ok.any():
        _, _, Vt = np.linalg.svd(A[ok])
        h = Vt[:, -1, :]
        d = np.where(np.abs(h[:, 3]) < 1e-12, 1.0, h[:, 3])
        out[ok] = h[:, :3] / d[:, None]
    return out.reshape(T, N_KP, 3)


def jitter(kp):
    d2 = np.diff(kp, n=2, axis=0)
    return np.concatenate([[np.nan], np.linalg.norm(d2, axis=-1).mean(axis=1), [np.nan]])


def bone_cv(kp):
    return ql.bone_length_cv(kp)


def main():
    rows = []
    for rat, session in SESSIONS:
        raw = sc.get_raw_session(rat, session, want_dual=True)
        sl = np.asarray(raw['triang_3d'], np.float32)
        k2 = np.asarray(raw['sleap_keys_2D'])
        fm = np.asarray(raw['frame_mapping_cam0'], dtype=int)
        nproc = min(len(fm), len(k2))
        fmp = fm[:nproc]
        port = np.asarray(raw['port_xy']).ravel()[:2]
        com = np.median(sl, axis=1)[:, :2]
        dist = np.hypot(com[:, 0] - port[0], com[:, 1] - port[1])
        ok = (fmp >= 0) & (fmp < len(sl))
        dproc = np.full(nproc, np.nan)
        dproc[ok] = dist[fmp[ok]]

        cal = ql.calib_dir(rat, session)
        P, K, D, R, T = ql.load_calib(cal)
        xy = k2[:nproc, :, :, :2].astype(np.float64)
        conf = k2[:nproc, :, :, 2].astype(np.float64)
        und = ql.undistort_all(xy, K, D)

        n_conf = (conf >= CONF_T).sum(axis=1)                 # (nproc, 23)
        w_eq = np.ones_like(conf)
        w_wt = np.clip(conf, 0.0, None)
        order = np.argsort(-conf, axis=1)
        w_b2 = np.zeros_like(conf)
        np.put_along_axis(w_b2, order[:, :2, :], 1.0, axis=1)

        variants = {'equal': dlt(und, P, w_eq), 'weighted': dlt(und, P, w_wt),
                    'best2': dlt(und, P, w_b2)}
        g = variants['weighted'].copy()
        g[n_conf < 2] = np.nan
        variants['gated'] = g

        for lab, m in [('near port', dproc < NEAR), ('far from port', dproc > FAR)]:
            m = m & np.isfinite(dproc)
            if m.sum() < 500:
                continue
            rec = dict(rat=rat, session=session,
                       era='Oct26' if '10_26' in session else 'Oct23',
                       stratum=lab, n=int(m.sum()),
                       mean_conf_cams=float(n_conf[m].mean()),
                       frac_kp_lt2_cams=float((n_conf[m] < 2).mean()))
            for name, kp in variants.items():
                with np.errstate(all='ignore'):
                    rec[f'{name}_bonecv'] = float(np.nanmedian(bone_cv(kp)[m]))
                    rec[f'{name}_jitter'] = float(np.nanmedian(jitter(kp)[m]))
                    rec[f'{name}_nan_frac'] = float(np.isnan(kp[m]).any(axis=-1).mean())
            rows.append(rec)
        print(f'{rat}/{session} done', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/triangulation_variants.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n=== how constrained is the triangulation? (fraction of keypoint-frames '
          'with fewer than 2 confident cameras) ===')
    print(d.groupby(['era', 'stratum'])[['mean_conf_cams', 'frac_kp_lt2_cams']]
          .median().round(3).to_string())
    for metric in ['bonecv', 'jitter']:
        cols = [f'{v}_{metric}' for v in ['equal', 'weighted', 'best2', 'gated']]
        print(f'\n=== {metric} by variant (lower is better) ===')
        print(d.groupby(['era', 'stratum'])[cols].median().round(4).to_string())
    print('\n=== how much of the session `gated` would discard ===')
    print(d.groupby(['era', 'stratum'])[['gated_nan_frac']].median().round(3).to_string())
    print(f'\nwrote {OUT}/triangulation_variants.csv')


if __name__ == '__main__':
    main()
