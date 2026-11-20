"""ear_mean_z and cum_heading_change from three pose sources, compared pairwise.

Sources, all on the same contiguous acquisition grid and all carrying the
online rule's own filtering:

  eq   SLEAP, equal-weight DLT -- exactly what `triang_keys_3D` contains
  wt   SLEAP, conf^2-weighted DLT -- the change proposed in
       WEIGHTED_TRIANGULATION_PLAN.md, whose exponent was fixed by the sweep in
       REPORT.md ("which weighting function?")
  dn   DANNCE, Procrustes-mapped into SLEAP's world frame

Pipeline per source: world keypoints -> `ev.causal_median_5_kp(warmup='online')`
-> z negated to up-positive -> `clirb.features.compute_features`.  That is the
same 5-frame causal median the online rule applies and every offline feature
script applies through `ev.project_session_to_pcs`, with the first four frames
left unfiltered as the rule leaves them while its buffer fills.

Two conventions worth stating because they are easy to get wrong:

  * The Procrustes fit maps DANNCE onto the EQUAL-weight SLEAP, and that single
    transform is reused for the weighted variant.  Fitting a separate transform
    per variant would give each one an alignment tuned to itself and make
    `eq vs dn` and `wt vs dn` incomparable.
  * The weighted variant is re-triangulated on PROCESSED frames only; its gaps
    are then filled by the same linear interpolation the rig applies to
    `triang_keys_3D`.  Both features are temporal -- `cum_heading_change` is a
    30-frame rolling sum -- so a non-contiguous series would be meaningless.

`cum_heading_change` is a cumulative quantity, not an angle on (-180, 180], so
its differences are NOT wrapped (it is in `cf.SIGNED_FEATURES`, not
`cf.ANGLE_FEATURES`).  Turn-sign agreement conditioned on a real turn is
reported alongside RMSE and correlation, because for that feature the sign is
the gate the project actually cares about.

Everything is also split by distance from the water port, using the SLEAP
centre of mass (median over keypoints, which fly-aways cannot drag) against
each session's own `water_port_location.npy`.  The port mechanism obstructs
both rigs in that sector -- see the REPORT appendix "the size bias is the water
port mechanism" -- so a pooled number mixes two regimes with very different
reliability.  Strata partition the session: near <150 mm, mid 150-300 mm,
far >300 mm, plus `all`.

Writes results/newcal/feature_three_way.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, features as cf
from clirb.spatial import dual as du
from clirb.spatial import quality as ql
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample
from scripts.newcal_weighted_reanalysis import dlt_chunked, fill_gaps

OUT = 'results/newcal'
EXPONENT = 2.0
FEATURES = ['ear_mean_z', 'cum_heading_change']
UNITS = {'ear_mean_z': 'mm', 'cum_heading_change': 'deg'}
TURN_DEG = 45.0
PAIRS = [('eq', 'dn'), ('wt', 'dn'), ('eq', 'wt')]
NEAR, FAR = 150.0, 300.0
STRATA = ['all', 'near port (<150mm)', 'mid (150-300mm)', 'far (>300mm)']

SESSIONS = (
    [('R7', s) for s in ['2026_10_22_1', '2026_10_23_1', '2026_10_26_1',
                         '2026_10_27_1', '2026_10_28_1']] +
    [('R8', s) for s in ['2026_10_22_1', '2026_10_23_1', '2026_10_26_1',
                         '2026_10_27_1', '2026_10_28_1']] +
    [('R9', s) for s in ['2026_10_26_1', '2026_10_27_1', '2026_10_28_1']])


def build(rat, session):
    """(features per source, n_frames) or None."""
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw is None or not raw.get('has_dual', False):
        return None
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        return None
    sl_eq = np.asarray(raw['triang_3d'], np.float32)
    n = min(len(sl_eq), len(d3))
    sl_eq, dn = sl_eq[:n], np.asarray(d3[:n], np.float32)

    k2 = np.asarray(raw['sleap_keys_2D'])
    fm = np.asarray(raw['frame_mapping_cam0'], int)
    nproc = min(len(fm), len(k2))
    P, K, D, R, T = ql.load_calib(ql.calib_dir(rat, session))
    und = ql.undistort_all(k2[:nproc, :, :, :2].astype(np.float64), K, D)
    conf = np.clip(k2[:nproc, :, :, 2].astype(np.float64), 0, None)
    w = conf ** EXPONENT
    under = (w > 0).sum(axis=1) < 2
    w[np.broadcast_to(under[:, None, :], w.shape)] = 1.0     # on_underdetermined: equal
    kp = dlt_chunked(und, P, w)

    sl_wt = np.full_like(sl_eq, np.nan)
    fmp = fm[:nproc]
    ok = (fmp >= 0) & (fmp < n) & np.isfinite(kp).all(axis=(1, 2))[:len(fmp)]
    sl_wt[fmp[ok]] = kp[ok]
    known = np.zeros(n, bool); known[fmp[ok]] = True
    sl_wt = fill_gaps(sl_wt, known)

    samp, _ = balanced_sample(np.median(sl_eq, axis=1)[:, :2],
                              rng=np.random.default_rng(0))
    if len(samp) < 500:
        return None
    fit = robust_proc(dn[samp], sl_eq[samp])
    dn_w = fit['apply'](dn).astype(np.float32)

    feats = {}
    for tag, arr in [('eq', sl_eq), ('wt', sl_wt), ('dn', dn_w)]:
        a = ev.causal_median_5_kp(np.asarray(arr, np.float32), warmup='online').copy()
        a[:, :, 2] *= -1.0
        feats[tag] = cf.compute_features(a)

    # port proximity from the SLEAP COM, median over keypoints so a fly-away
    # cannot drag it
    port = np.asarray(raw['port_xy']).ravel()[:2]
    com = np.median(sl_eq, axis=1)[:, :2]
    dist = (np.hypot(com[:, 0] - port[0], com[:, 1] - port[1])
            if np.isfinite(port).all() else np.full(n, np.nan))
    masks = {'all': np.ones(n, bool),
             'near port (<150mm)': dist < NEAR,
             'mid (150-300mm)': (dist >= NEAR) & (dist <= FAR),
             'far (>300mm)': dist > FAR}
    return feats, n, float(fit['residual']), float(1 - known.mean()), masks


def main():
    rows, pooled = [], {}
    for i, (rat, session) in enumerate(SESSIONS, 1):
        out = build(rat, session)
        if out is None:
            print(f'[{i}/{len(SESSIONS)}] {rat}/{session}: unusable', flush=True)
            continue
        feats, n, resid, interp, masks = out
        for feat in FEATURES:
            v = {t: np.asarray(feats[t][feat], float) for t in ('eq', 'wt', 'dn')}
            for stratum, sel in masks.items():
                if sel.sum() < 500:
                    continue
                for a, b in PAIRS:
                    A, B = v[a][sel], v[b][sel]
                    d = A - B
                    rec = dict(rat=rat, session=session, feature=feat,
                               unit=UNITS[feat], pair=f'{a}_vs_{b}',
                               stratum=stratum, n_frames=int(sel.sum()),
                               frac_of_session=float(sel.mean()),
                               procrustes_residual_mm=resid, interp_frac=interp,
                               rmse=float(np.sqrt(np.mean(d ** 2))),
                               med_abs=float(np.median(np.abs(d))),
                               p90_abs=float(np.percentile(np.abs(d), 90)),
                               bias=float(np.mean(d)),
                               pearson_r=float(np.corrcoef(A, B)[0, 1]))
                    if feat == 'cum_heading_change':
                        m = np.abs(A) >= TURN_DEG
                        rec['frac_real_turn'] = float(m.mean())
                        rec['sign_agree_turn_pct'] = (
                            float(100 * (np.sign(A) == np.sign(B))[m].mean())
                            if m.sum() > 100 else np.nan)
                    rows.append(rec)
                    pooled.setdefault((feat, f'{a}_vs_{b}', stratum), []).append(
                        (A.copy(), B.copy()))
        print(f'[{i}/{len(SESSIONS)}] {rat}/{session} done '
              f'(n={n}, interp {100*interp:.1f}%)', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/feature_three_way.csv', index=False)

    # pooled over every frame of every session, not an average of per-session values
    prows = []
    for (feat, pair, stratum), chunks in pooled.items():
        A = np.concatenate([c[0] for c in chunks])
        B = np.concatenate([c[1] for c in chunks])
        dd = A - B
        rec = dict(feature=feat, unit=UNITS[feat], pair=pair, stratum=stratum,
                   n_frames=len(A),
                   rmse=float(np.sqrt(np.mean(dd ** 2))),
                   med_abs=float(np.median(np.abs(dd))),
                   p90_abs=float(np.percentile(np.abs(dd), 90)),
                   bias=float(np.mean(dd)),
                   pearson_r=float(np.corrcoef(A, B)[0, 1]))
        if feat == 'cum_heading_change':
            m = np.abs(A) >= TURN_DEG
            rec['frac_real_turn'] = float(m.mean())
            rec['sign_agree_turn_pct'] = float(100 * (np.sign(A) == np.sign(B))[m].mean())
        prows.append(rec)
    p = pd.DataFrame(prows)
    p.to_csv(f'{OUT}/feature_three_way_pooled.csv', index=False)

    pd.set_option('display.width', 250)
    order = ['eq_vs_dn', 'wt_vs_dn', 'eq_vs_wt']
    nsess = d.groupby(['rat', 'session']).ngroups
    for feat in FEATURES:
        cols = ['pair', 'rmse', 'med_abs', 'p90_abs', 'bias', 'pearson_r']
        if feat == 'cum_heading_change':
            cols += ['sign_agree_turn_pct']
        print(f'\n{"="*92}\nAGGREGATED — {feat} ({UNITS[feat]}), pooled over '
              f'every frame of all {nsess} sessions\n{"="*92}')
        for stratum in STRATA:
            s_ = p[(p.feature == feat) & (p.stratum == stratum)]
            if not len(s_):
                continue
            s_ = s_.set_index('pair').reindex(order).reset_index()
            frac = d[(d.feature == feat) & (d.stratum == stratum) &
                     (d.pair == 'eq_vs_dn')].frac_of_session.median()
            print(f'\n  [{stratum}]  {int(s_.n_frames.iloc[0])} frames, '
                  f'median {100*frac:.0f}% of a session')
            print(s_[cols].round(4).to_string(index=False))

    print(f'\n{"="*92}\nPER SESSION\n{"="*92}')
    for feat in FEATURES:
        for stratum in ['all', 'near port (<150mm)', 'far (>300mm)']:
            sub = d[(d.feature == feat) & (d.stratum == stratum)]
            if not len(sub):
                continue
            print(f'\n--- {feat} ({UNITS[feat]}) | {stratum} ---')
            for metric in ['rmse', 'pearson_r']:
                t = sub.pivot_table(index=['rat', 'session'], columns='pair',
                                    values=metric).reindex(columns=order)
                print(f'\n  {metric}:')
                print(t.round(4).to_string())
    print(f'\nwrote {OUT}/feature_three_way.csv and _pooled.csv')


if __name__ == '__main__':
    main()
