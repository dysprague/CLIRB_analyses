"""Stage 4 — conditioning + the counterfactual map.

4.1 Does sector still predict reward once true behavior (d_dannce) is held fixed?
    - hierarchical logistic (session-clustered bootstrap)
    - matched pairs on d_dannce across sectors
    - mediation split
4.2 Counterfactual: take a FIXED library of behavior, corrupt it with the
    residual noise MEASURED IN EACH SECTOR (block bootstrap of 30-frame
    snippets, preserving temporal autocorrelation), run the exact online rule.
    -> P(match | identical behavior, sector)
"""
import os, sys, json
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb import events as ev, templates as tp
from clirb.spatial.geometry import sector_center

OUT = 'results/spatial_confound'
SECTOR_W = 30
RNG = np.random.default_rng(0)
NPERM = 10_000


# ------------------------------------------------------------------ helpers
def session_boot(df, fn, n=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    us = df['session'].unique()
    if len(us) < 2:
        return np.nan, np.nan
    g = {s: df[df.session == s] for s in us}
    out = []
    for _ in range(n):
        pick = rng.choice(us, size=len(us), replace=True)
        out.append(fn(pd.concat([g[s] for s in pick], ignore_index=True)))
    o = np.array(out, dtype=float)
    o = o[np.isfinite(o)]
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) else (np.nan, np.nan)


def logistic_fit(df):
    """matched ~ sector dummies + d_dannce (+ apex proxies). Returns the
    max-min sector log-odds contrast after conditioning."""
    from sklearn.linear_model import LogisticRegression
    d = df.dropna(subset=['d_dannce']).copy()
    if d.matched.nunique() < 2 or len(d) < 200:
        return np.nan, np.nan, None
    d['ld'] = np.log10(np.maximum(d.d_dannce, 1e-6))
    # keep only sectors with enough data so dummies are estimable
    vc = d.sector.value_counts()
    d = d[d.sector.isin(vc[vc >= 60].index)]
    if d.sector.nunique() < 3:
        return np.nan, np.nan, None
    X = pd.get_dummies(d['sector'].astype(int), prefix='s', drop_first=True).astype(float)
    names = list(X.columns)
    X['ld'] = (d['ld'] - d['ld'].mean()).values
    try:
        m = LogisticRegression(penalty=None, max_iter=2000, solver='lbfgs')
        m.fit(X.values, d['matched'].astype(int).values)
    except Exception:
        return np.nan, np.nan, None
    coef = m.coef_.ravel()
    sec_coefs = np.concatenate([[0.0], coef[:len(names)]])  # include dropped ref
    return float(sec_coefs.max() - sec_coefs.min()), float(coef[-1]), m


def matched_pairs(df, tol_frac=0.15, min_pairs=30, rng=None):
    """Pair rears from DIFFERENT sectors with d_dannce within tol_frac (in log
    space). Returns (n_pairs, mean signed difference in match rate between the
    higher-P and lower-P sector member, per-sector table)."""
    rng = rng or np.random.default_rng(0)
    d = df.dropna(subset=['d_dannce']).copy()
    d = d[d.d_dannce > 0]
    if len(d) < 100:
        return None
    d['ld'] = np.log10(d.d_dannce)
    # bin behavior quality finely; within a bin, compare sectors
    d['qb'] = pd.qcut(d.ld, 20, labels=False, duplicates='drop')
    rows = []
    for qb, g in d.groupby('qb'):
        if g.sector.nunique() < 2:
            continue
        for s, gg in g.groupby('sector'):
            if len(gg) < 5:
                continue
            rows.append(dict(qb=qb, sector=s, n=len(gg), p=gg.matched.mean()))
    t = pd.DataFrame(rows)
    if t.empty:
        return None
    # within each quality bin, spread across sectors
    spread = t.groupby('qb').apply(
        lambda g: pd.Series(dict(spread=g.p.max() - g.p.min(),
                                 n_sec=g.sector.nunique(),
                                 n=g.n.sum())), include_groups=False)
    return t, spread


# ------------------------------------------------ counterfactual simulation
def block_bootstrap(snippets, sector_ids, sector, n, rng):
    ii = np.where(sector_ids == sector)[0]
    if len(ii) == 0:
        return None
    pick = rng.choice(ii, size=n, replace=True)
    return snippets[pick]


def counterfactual_map(library, snippets, sector_ids, template, bounds,
                       accepted_errors=ev.ACCEPTED_ERRORS, n_draw=4000,
                       rng=None, sectors=None):
    """P(match | fixed behavior, sector).

    library : (L, 30, 2) TRUE behavior windows (DANNCE)
    snippets: (S, 30, 2) measured SLEAP-DANNCE residual windows
    Applies the online rule's out-of-bounds count to library + residual.
    """
    rng = rng or np.random.default_rng(0)
    sectors = sectors if sectors is not None else np.unique(sector_ids)
    rows = []
    for s in sectors:
        res = block_bootstrap(snippets, sector_ids, s, n_draw, rng)
        if res is None:
            continue
        beh = library[rng.integers(0, len(library), size=n_draw)]
        z = beh + res
        oob = ((z >= template + bounds) | (z <= template - bounds)
               ).reshape(len(z), -1).sum(axis=1)
        p = float((oob <= accepted_errors).mean())
        rows.append(dict(sector=int(s), theta_c=float(sector_center(int(s), SECTOR_W)),
                         n_snip=int((sector_ids == s).sum()), p_match=p))
    return pd.DataFrame(rows)


def permutation_null(library, snippets, sector_ids, template, bounds,
                     n_perm=1000, n_draw=1500, rng=None):
    """Spatial variance of the counterfactual map under shuffled sector labels."""
    rng = rng or np.random.default_rng(1)
    stats = []
    for _ in range(n_perm):
        sh = rng.permutation(sector_ids)
        m = counterfactual_map(library, snippets, sh, template, bounds,
                               n_draw=n_draw, rng=rng)
        if len(m) > 1:
            stats.append(m.p_match.var())
    return np.array(stats)


def main():
    rears = pd.read_csv(f'{OUT}/rears.csv')
    dan = pd.read_csv(f'{OUT}/rears_dannce.csv')
    print(f'rears={len(rears)}  dannce-scored={len(dan)}')

    # gate-on sessions, outside the port zone
    d = dan[(dan.min_port_distance > 0) & (dan.port_dist >= dan.min_port_distance)].copy()
    print(f'analysis set (gate-on, outside gate zone): {len(d)}')

    results = {}

    # ---------------- 4.1 conditioning ----------------
    print('\n' + '=' * 78); print('4.1 CONDITIONING'); print('=' * 78)
    for rat, g in d.groupby('rat'):
        raw_spread = (g.groupby('sector').matched.mean().max() -
                      g.groupby('sector').matched.mean().min())
        contrast, ld_coef, m = logistic_fit(g)
        print(f'\n{rat}: n={len(g)}  raw sector spread in P={raw_spread:.3f}')
        print(f'   logistic: max-min sector log-odds AFTER conditioning on '
              f'd_dannce = {contrast:.3f}  (d_dannce coef={ld_coef:.3f})')
        # unconditioned contrast for comparison
        c0, _, _ = logistic_fit(g.assign(d_dannce=1.0))
        print(f'   logistic: same contrast WITHOUT conditioning       = {c0:.3f}')
        mp = matched_pairs(g)
        if mp is not None:
            t, spread = mp
            w = spread[spread.n_sec >= 3]
            print(f'   matched-pairs: {len(w)} quality bins with >=3 sectors, '
                  f'median within-bin P spread = {w.spread.median():.3f} '
                  f'(mean n/bin={w.n.mean():.0f})')
            results[f'{rat}_matched_pair_spread'] = float(w.spread.median())
        results[f'{rat}_raw_spread'] = float(raw_spread)
        results[f'{rat}_cond_contrast'] = contrast
        results[f'{rat}_uncond_contrast'] = c0

    # ---------------- 4.2 counterfactual ----------------
    print('\n' + '=' * 78); print('4.2 COUNTERFACTUAL MAP'); print('=' * 78)
    cf_all = []
    for rat in ['R4', 'R5', 'R6']:
        f = f'{OUT}/residual_snippets_{rat}.npz'
        if not os.path.isfile(f):
            continue
        z = np.load(f)
        snips, secs = z['snippets'], z['sector']
        lib_all, lib_sec = z['library'], z['library_sector']

        # template/bounds: the rat's most-used production template
        tfile = dan[dan.rat == rat].template_file.mode().iloc[0]
        trow = dan[(dan.rat == rat) & (dan.template_file == tfile)].iloc[0]
        t = tp.load_template(rat, trow.session)
        tmpl = t['template'][:, :2]; bnds = t['bounds_to_use'][:, :2]

        keep = np.array([s for s in np.unique(secs)
                         if (secs == s).sum() >= 40 and (lib_sec == s).sum() >= 40])
        if len(keep) < 3:
            continue

        # ---- SELF-CONSISTENCY: behavior and noise from the SAME sector must
        # reproduce the observed P(match) in that sector.
        rng = np.random.default_rng(0)
        self_rows = []
        obs = dan[(dan.rat == rat) & (dan.min_port_distance > 0) &
                  (dan.port_dist >= dan.min_port_distance)]
        for s in keep:
            li = np.where(lib_sec == s)[0]; si = np.where(secs == s)[0]
            beh = lib_all[rng.choice(li, 3000)]
            res = snips[rng.choice(si, 3000)]
            zz = beh + res
            oob = ((zz >= tmpl + bnds) | (zz <= tmpl - bnds)).reshape(len(zz), -1).sum(axis=1)
            o = obs[obs.sector == s]
            self_rows.append(dict(sector=int(s), sim=float((oob <= ev.ACCEPTED_ERRORS).mean()),
                                  obs=float(o.matched.mean()) if len(o) >= 30 else np.nan,
                                  n_obs=len(o)))
        sc_df = pd.DataFrame(self_rows).dropna()
        if len(sc_df) > 2:
            rho = np.corrcoef(sc_df.sim, sc_df.obs)[0, 1]
        else:
            rho = np.nan

        # ---- COUNTERFACTUAL: ONE fixed, sector-balanced behavior library,
        # sector-specific noise.
        n_bal = min(int((lib_sec == s).sum()) for s in keep)
        bal = np.concatenate([rng.choice(np.where(lib_sec == s)[0], n_bal, replace=False)
                              for s in keep])
        lib = lib_all[bal]

        m = counterfactual_map(lib, snips, secs, tmpl, bnds,
                               rng=np.random.default_rng(0), sectors=keep)
        m['rat'] = rat
        obs_var = m.p_match.var()
        null = permutation_null(lib, snips, secs, tmpl, bnds, n_perm=400,
                                rng=np.random.default_rng(1))
        p = float((null >= obs_var).mean())
        print(f'\n{rat} (template {tfile}, {len(snips)} snippets, '
              f'library {len(lib)} balanced windows)')
        print(f'   self-consistency sim-vs-observed P(match) across sectors: '
              f'r={rho:+.3f} (n={len(sc_df)} sectors)')
        print(m[['theta_c', 'n_snip', 'p_match']].to_string(index=False,
              float_format=lambda v: f'{v:.3f}'))
        print(f'   spread {m.p_match.min():.3f} -> {m.p_match.max():.3f}  '
              f'(best theta={m.loc[m.p_match.idxmax(),"theta_c"]:.0f}, '
              f'worst={m.loc[m.p_match.idxmin(),"theta_c"]:.0f})')
        print(f'   spatial variance {obs_var:.5f} vs shuffled null '
              f'mean={null.mean():.5f} p95={np.percentile(null,95):.5f}  p={p:.4f}')
        results[f'{rat}_selfconsistency_r'] = float(rho) if np.isfinite(rho) else None
        results[f'{rat}_cf_min'] = float(m.p_match.min())
        results[f'{rat}_cf_max'] = float(m.p_match.max())
        results[f'{rat}_cf_best_theta'] = float(m.loc[m.p_match.idxmax(), 'theta_c'])
        results[f'{rat}_cf_p'] = p
        cf_all.append(m)
        sc_df.to_csv(f'{OUT}/selfconsistency_{rat}.csv', index=False)
    if cf_all:
        pd.concat(cf_all).to_csv(f'{OUT}/counterfactual_map.csv', index=False)
    with open(f'{OUT}/stage4_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nwrote {OUT}/counterfactual_map.csv and stage4_results.json')


if __name__ == '__main__':
    main()
