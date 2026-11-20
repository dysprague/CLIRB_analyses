"""Sweep the weighting function for confidence-weighted triangulation.

The 12-22% improvement reported in REPORT.md was measured with ONE choice --
raw confidence as the weight, no floor, no ceiling.  That choice was never
compared against alternatives, and it is about to become an online default, so
this sweeps it over the same 22 sessions before it is fixed.

Variants (w is the per-camera, per-keypoint DLT row weight):
  equal      w = 1                       the rig's current behaviour
  linear     w = conf                    what the headline numbers used
  sq         w = conf^2                  sharper -- trusts the best view more
  sqrt       w = sqrt(conf)              gentler -- closer to equal
  floor10    w = conf if conf >= 0.10    drop views the detector has written off
  floor35    w = conf if conf >= 0.35    drop more aggressively
  cap80      w = min(conf, 0.80)         stop one very confident view dominating
  binary50   w = 1 if conf >= 0.50       hard gate, no graded weighting

UNDER-DETERMINED FALLBACK.  Where fewer than two cameras have w > 0 the point is
unconstrained, so it falls back to equal weights -- the `on_underdetermined:
equal` behaviour the online plan proposes, which is never worse than today.
Without it the floor and binary variants would be mostly NaN: 27-55% of
keypoint-frames have fewer than two cameras above 0.5 confidence.  The rate at
which the fallback fires is recorded per variant, because a variant that is
only good by falling back to `equal` is not actually good.

Scored with the same metrics as newcal_weighted_reanalysis.py, on the same
contiguous full-length series (gaps filled as the rig fills them).

Writes results/newcal/weight_function_sweep2.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import dual as du
from clirb.spatial import quality as ql
from scripts.newcal_sleap_dannce_eval import COMMON_TEMPLATE
from scripts.newcal_weighted_reanalysis import (dlt_chunked, fill_gaps, metrics,
                                                N_PER_ERA, N_KP)

OUT = 'results/newcal'


def weight_variants(conf):
    """name -> (n_cams, n_kp) weights, same shape as conf."""
    c = np.clip(conf, 0.0, None)
    return {
        'equal':    np.ones_like(c),
        'linear':   c,
        'sq':       c ** 2,
        'sqrt':     np.sqrt(c),
        'floor10':  np.where(c >= 0.10, c, 0.0),
        'floor35':  np.where(c >= 0.35, c, 0.0),
        'cap80':    np.minimum(c, 0.80),
        'binary50': (c >= 0.50).astype(float),
        # exponent extension: conf^2 beat conf in the first pass, so check
        # whether the trend continues or turns over
        'p1_5':     c ** 1.5,
        'p3':       c ** 3,
        'p4':       c ** 4,
        'p6':       c ** 6,
    }


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    use = pd.concat([inv[inv.era == e].sort_values('session').tail(n)
                     for e, n in N_PER_ERA.items()])
    common = dict(np.load(COMMON_TEMPLATE, allow_pickle=True))

    rows = []
    for i, r in enumerate(use.itertuples(), 1):
        raw = sc.get_raw_session(r.rat, r.session, want_dual=True)
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        sl_rig = np.asarray(raw['triang_3d'], np.float32)
        n = min(len(sl_rig), len(d3))
        dn = np.asarray(d3[:n], np.float32)
        k2 = np.asarray(raw['sleap_keys_2D'])
        fm = np.asarray(raw['frame_mapping_cam0'], int)
        nproc = min(len(fm), len(k2))
        fmp = fm[:nproc]
        P, K, D, T_, _t = ql.load_calib(ql.calib_dir(r.rat, r.session))[0:5]
        und = ql.undistort_all(k2[:nproc, :, :, :2].astype(np.float64), K, D)
        conf = k2[:nproc, :, :, 2].astype(np.float64)

        for name, w in weight_variants(conf).items():
            under = (w > 0).sum(axis=1) < 2                   # (nproc, 23)
            w_eff = w.copy()
            w_eff[np.broadcast_to(under[:, None, :], w.shape)] = 1.0
            kp = dlt_chunked(und, P, w_eff)
            sl = np.full((n, N_KP, 3), np.nan, np.float32)
            ok = ((fmp >= 0) & (fmp < n)
                  & np.isfinite(kp).all(axis=(1, 2))[:len(fmp)])
            sl[fmp[ok]] = kp[ok]
            known = np.zeros(n, bool); known[fmp[ok]] = True
            sl = fill_gaps(sl, known)
            m = metrics(sl, dn, common, 'v')
            if m is None:
                continue
            rec = dict(rat=r.rat, session=r.session, era=r.era, variant=name,
                       fallback_frac=float(under.mean()))
            rec.update({k[2:]: v for k, v in m.items()
                        if not k.startswith('v_kpmed_')})
            rows.append(rec)
        print(f'[{i}/{len(use)}] {r.rat}/{r.session} ({r.era}) done', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/weight_function_sweep2.csv', index=False)

    pd.set_option('display.width', 250)
    order = ['equal', 'sqrt', 'linear', 'p1_5', 'sq', 'p3', 'p4', 'p6',
             'cap80', 'floor10', 'floor35', 'binary50']
    cols = ['kp_med', 'bone_cv', 'jitter', 'fit_residual', 'ear_rmse',
            'turn_rmse', 'pc1_r', 'rear_f1', 'fallback_frac']
    print('\n=== pooled over all 22 sessions (medians) ===')
    g = d.groupby('variant')[cols].median().reindex(order)
    base = g.loc['equal']
    pct = pd.DataFrame({c: 100 * (g[c] / base[c] - 1) for c in
                        ['kp_med', 'bone_cv', 'jitter', 'fit_residual',
                         'ear_rmse', 'turn_rmse']})
    print(g.round(4).to_string())
    print('\n  change vs equal (%), negative = better:')
    print(pct.round(1).to_string())
    print('\n=== by era, median keypoint error (mm) ===')
    print(d.pivot_table(index='variant', columns='era', values='kp_med',
                        aggfunc='median').reindex(order).round(3).to_string())
    print('\n=== by era, bone-length CV ===')
    print(d.pivot_table(index='variant', columns='era', values='bone_cv',
                        aggfunc='median').reindex(order).round(4).to_string())
    print(f'\nwrote {OUT}/weight_function_sweep2.csv')


if __name__ == '__main__':
    main()
