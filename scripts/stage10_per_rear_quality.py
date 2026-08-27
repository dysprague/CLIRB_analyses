"""Stage 10 — per-rear tracking quality (corrected v2 cache).

Per-sector statistics have only 11 points per rat, so a real quality effect can
hide in the noise. This attaches the corrected per-frame quality metrics to each
individual rear and asks:

  (a) does quality predict whether the rule fired?
  (b) does quality predict the SLEAP-DANNCE discrepancy?   <- the mechanistic link
  (c) does conditioning on quality shrink the sector effect?
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from scipy import stats

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

OUT = 'results/spatial_confound'
METRICS = ['reproj_px', 'mean_conf', 'n_cams_conf50', 'fov_min', 'bone_cv',
           'jitter', 'nan_frac']
RNG = np.random.default_rng(0)


def load_q(rat, session):
    p = f'data/cache_quality/{rat}/{session}.npz'
    if not os.path.isfile(p):
        return None
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    with np.errstate(all='ignore'):
        d['fov_min'] = np.nanmin(d['fov_margin'], axis=1)
    return d


def main():
    rears = pd.read_csv(f'{OUT}/rears.csv')
    rows = []
    for (rat, session), g in rears.groupby(['rat', 'session']):
        q = load_q(rat, session)
        if q is None:
            continue
        T = len(q['theta'])
        for _, r in g.iterrows():
            a, b = int(r.start_frame), min(int(r.end_frame) + 1, T)
            ae = int(r.align_end)
            w0, w1 = max(ae - 29, 0), min(ae + 1, T)
            rec = dict(rat=rat, session=session, rear_idx=int(r.rear_idx))
            for m in METRICS:
                v = np.asarray(q[m], dtype=float)
                with np.errstate(all='ignore'):
                    rec[m] = np.nanmean(v[a:b]) if b > a else np.nan
                    rec[m + '_win'] = np.nanmean(v[w0:w1]) if w1 > w0 else np.nan
            rows.append(rec)
    Q = pd.DataFrame(rows)
    d = rears.merge(Q, on=['rat', 'session', 'rear_idx'], how='left')
    d.to_csv(f'{OUT}/rears_with_quality.csv', index=False)
    print(f'{len(d)} rears, quality attached to {d.reproj_px.notna().sum()}')

    res = {}
    # analysis set: gate-on, outside the gate zone
    A = d[(d.min_port_distance > 0) & (~d.in_gate_zone)].copy()
    print(f'analysis set: {len(A)}\n')

    # ---------- (a) does quality predict matched, per rear?
    print('=' * 76); print('(a) PER-REAR: quality vs matched'); print('=' * 76)
    for rat, g in A.groupby('rat'):
        g = g.dropna(subset=['reproj_px'])
        print(f'\n{rat} (n={len(g)})')
        for m in METRICS:
            v = g[m].values; y = g.matched.values.astype(float)
            ok = np.isfinite(v)
            if ok.sum() < 200 or np.nanstd(v[ok]) == 0:
                continue
            rho, p = stats.spearmanr(v[ok], y[ok])
            # matched vs unmatched medians
            m1 = np.nanmedian(v[ok & g.matched.values])
            m0 = np.nanmedian(v[ok & ~g.matched.values])
            print(f'   {m:14} rho={rho:+.3f} p={p:.2e}   median matched={m1:8.3f} '
                  f'unmatched={m0:8.3f}')
            res[f'{rat}_{m}_vs_matched_rho'] = float(rho)
            res[f'{rat}_{m}_vs_matched_p'] = float(p)

    # ---------- (b) quality vs the SLEAP-DANNCE discrepancy
    print('\n' + '=' * 76)
    print('(b) PER-REAR: quality vs the SLEAP-DANNCE discrepancy (mechanistic link)')
    print('=' * 76)
    dan = pd.read_csv(f'{OUT}/rears_dannce.csv')
    M = A.merge(dan[['rat', 'session', 'rear_idx', 'd_dannce', 'resid_abs']],
                on=['rat', 'session', 'rear_idx'], how='inner')
    print(f'merged with DANNCE: {len(M)}')
    for rat, g in M.groupby('rat'):
        g = g.dropna(subset=['reproj_px', 'resid_abs'])
        print(f'\n{rat} (n={len(g)})')
        for m in METRICS:
            v = g[m].values; y = g.resid_abs.values
            ok = np.isfinite(v) & np.isfinite(y)
            if ok.sum() < 200 or np.nanstd(v[ok]) == 0:
                continue
            rho, p = stats.spearmanr(v[ok], y[ok])
            flag = '  <<<' if p < 0.001 and abs(rho) > 0.1 else ''
            print(f'   {m:14} vs |SLEAP-DANNCE resid|: rho={rho:+.3f} p={p:.2e}{flag}')
            res[f'{rat}_{m}_vs_resid_rho'] = float(rho)

    # ---------- (c) does quality explain the sector effect?
    print('\n' + '=' * 76)
    print('(c) does conditioning on per-rear quality shrink the sector effect?')
    print('=' * 76)
    from sklearn.linear_model import LogisticRegression
    for rat, g in A.groupby('rat'):
        g = g.dropna(subset=METRICS).copy()
        vc = g.sector_30.value_counts()
        g = g[g.sector_30.isin(vc[vc >= 60].index)]
        if g.sector_30.nunique() < 4 or len(g) < 400:
            continue
        X0 = pd.get_dummies(g.sector_30.astype(int), prefix='s',
                            drop_first=True).astype(float)
        names = list(X0.columns)

        def contrast(X):
            mdl = LogisticRegression(penalty=None, max_iter=3000)
            mdl.fit(X.values, g.matched.astype(int).values)
            c = np.concatenate([[0.0], mdl.coef_.ravel()[:len(names)]])
            return float(c.max() - c.min())

        base = contrast(X0)
        Xq = X0.copy()
        for m in METRICS:
            v = g[m].values.astype(float)
            if np.nanstd(v) > 0:
                Xq[m] = (v - np.nanmean(v)) / np.nanstd(v)
        withq = contrast(Xq)
        print(f'{rat}: sector log-odds contrast  {base:.3f} -> {withq:.3f} '
              f'after conditioning on quality  ({(withq-base)/base*100:+.0f}%)')
        res[f'{rat}_contrast_base'] = base; res[f'{rat}_contrast_withq'] = withq

    with open(f'{OUT}/stage10_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f'\nwrote {OUT}/rears_with_quality.csv and stage10_results.json')


if __name__ == '__main__':
    main()
