"""Stage 6 — synthesis: residual field, the direct H2 test, and debiased stats."""
import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy import stats

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from clirb import templates as tp
from clirb.spatial.geometry import sector_index, sector_center

OUT = 'results/spatial_confound'
SECTOR_W = 30
RNG = np.random.default_rng(0)
res = {}


# ---------------------------------------------------------------- 1. residual field
print('=' * 78); print('1. RESIDUAL FIELD  mu(theta) = SLEAP - DANNCE in PC space'); print('=' * 78)
dan = pd.read_csv(f'{OUT}/rears_dannce.csv')
for rat in ['R4', 'R5', 'R6']:
    f = f'{OUT}/residual_snippets_{rat}.npz'
    if not os.path.isfile(f):
        continue
    z = np.load(f)
    sn, sec = z['snippets'], z['sector']
    tfile = dan[dan.rat == rat].template_file.mode().iloc[0]
    trow = dan[(dan.rat == rat) & (dan.template_file == tfile)].iloc[0]
    t = tp.load_template(rat, trow.session)
    bnds = t['bounds_to_use'][:, :2]
    rows = []
    for s in np.unique(sec):
        m = sec == s
        if m.sum() < 40:
            continue
        r = sn[m]
        # residual in units of the rule's own tolerance -> directly interpretable
        rows.append(dict(theta_c=sector_center(int(s), SECTOR_W), n=int(m.sum()),
                         mu_pc0=float(r[..., 0].mean()), mu_pc1=float(r[..., 1].mean()),
                         sd_pc0=float(r[..., 0].std()), sd_pc1=float(r[..., 1].std()),
                         rel_sd=float((np.abs(r) / bnds[None]).mean())))
    d = pd.DataFrame(rows)
    print(f'\n{rat} (bounds mean PC0={bnds[:,0].mean():.0f} PC1={bnds[:,1].mean():.0f})')
    print(d.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(f'   mu_pc0 range {d.mu_pc0.min():.1f} -> {d.mu_pc0.max():.1f} '
          f'({(d.mu_pc0.max()-d.mu_pc0.min())/bnds[:,0].mean()*100:.0f}% of the PC0 tolerance)')
    res[f'{rat}_mu_pc0_range_pct_of_bounds'] = float(
        (d.mu_pc0.max() - d.mu_pc0.min()) / bnds[:, 0].mean() * 100)
    d.to_csv(f'{OUT}/residual_field_{rat}.csv', index=False)


# ------------------------------------------------------- 2. direct H2 test
print('\n' + '=' * 78)
print('2. H2 TEST — is P(match) highest in the template BUILD sector?')
print('=' * 78)
rears = pd.read_csv(f'{OUT}/rears.csv')
prov = pd.read_csv(f'{OUT}/template_provenance.csv')
prov['sector'] = sector_index(prov.theta.values, SECTOR_W)
ev = rears[(rears.min_port_distance > 0) & (~rears.in_gate_zone)]

rows = []
for (rat, tfile), g in prov.groupby(['rat', 'template']):
    build_sec = int(g.sector.value_counts().idxmax())
    build_share = float(g.sector.value_counts(normalize=True).max())
    use = ev[(ev.rat == rat) & (ev.template_file == tfile)]
    if len(use) < 200:
        continue
    pm = use.groupby('sector_30').agg(n=('matched', 'size'), p=('matched', 'mean'))
    pm = pm[pm.n >= 40]
    if len(pm) < 4:
        continue
    best = int(pm.p.idxmax())
    rank = int((pm.p.rank(ascending=False)).get(build_sec, np.nan)) \
        if build_sec in pm.index else np.nan
    rows.append(dict(rat=rat, template=tfile, n_build=len(g), build_sector=build_sec,
                     build_theta=sector_center(build_sec, SECTOR_W),
                     build_share=build_share, best_sector=best,
                     best_theta=sector_center(best, SECTOR_W),
                     n_rears=int(use.shape[0]), n_sectors=len(pm),
                     build_rank=rank,
                     p_build=float(pm.p.get(build_sec, np.nan)),
                     p_best=float(pm.p.max()), p_mean=float(pm.p.mean())))
h2 = pd.DataFrame(rows)
print(h2.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
h2.to_csv(f'{OUT}/h2_build_vs_best.csv', index=False)

ok = h2.dropna(subset=['build_rank'])
if len(ok):
    # is the build sector's rank better than chance (uniform over n_sectors)?
    exp = (ok.n_sectors + 1) / 2
    t, p = stats.wilcoxon(ok.build_rank - exp) if len(ok) > 5 else (np.nan, np.nan)
    print(f'\n  build-sector P(match) rank: mean={ok.build_rank.mean():.2f} '
          f'vs chance {exp.mean():.2f}   Wilcoxon p={p:.4f}  (n={len(ok)} templates)')
    print(f'  build sector IS the best sector in {(ok.build_rank==1).sum()}/{len(ok)} templates')
    print(f'  mean P(match) in build sector = {ok.p_build.mean():.3f} '
          f'vs sector-mean {ok.p_mean.mean():.3f}  (delta {ok.p_build.mean()-ok.p_mean.mean():+.3f})')
    res['h2_build_rank_mean'] = float(ok.build_rank.mean())
    res['h2_chance_rank'] = float(exp.mean())
    res['h2_p'] = float(p) if np.isfinite(p) else None
    res['h2_build_is_best'] = int((ok.build_rank == 1).sum())
    res['h2_n_templates'] = int(len(ok))
    res['h2_delta_p'] = float(ok.p_build.mean() - ok.p_mean.mean())


# --------------------------------------------- 3. referee validation
print('\n' + '=' * 78); print('3. REFEREE VALIDATION — does DANNCE quality itself vary by sector?')
print('=' * 78)
# proxy: does d_dannce vary by sector in a way that mirrors P(match)?
for rat, g in dan[(dan.min_port_distance > 0) &
                  (dan.port_dist >= dan.min_port_distance)].groupby('rat'):
    t = g.groupby('sector').agg(n=('matched', 'size'), p=('matched', 'mean'),
                                d=('d_dannce', 'median'))
    t = t[t.n >= 40]
    r, p = stats.spearmanr(t.d, t.p)
    print(f'{rat}: rho(median d_dannce, P(match)) across sectors = {r:+.3f} p={p:.3f} '
          f'(n={len(t)} sectors); d_dannce range {t.d.min():.0f}-{t.d.max():.0f}')
    res[f'{rat}_sector_ddannce_vs_p'] = float(r)


# --------------------------------------------- 4. debiased matched-pairs
print('\n' + '=' * 78)
print('4. CONDITIONED SECTOR EFFECT — variance of P across sectors within')
print('   d_dannce quality bins, vs a within-bin label-shuffle null')
print('=' * 78)
for rat, g in dan[(dan.min_port_distance > 0) &
                  (dan.port_dist >= dan.min_port_distance)].groupby('rat'):
    d = g.dropna(subset=['d_dannce']).copy()
    d = d[d.d_dannce > 0]
    d['qb'] = pd.qcut(np.log10(d.d_dannce), 10, labels=False, duplicates='drop')

    def stat(frame):
        vals = []
        for qb, gg in frame.groupby('qb'):
            t = gg.groupby('sector').agg(n=('matched', 'size'), p=('matched', 'mean'))
            t = t[t.n >= 20]
            if len(t) >= 3:
                vals.append(np.average((t.p - np.average(t.p, weights=t.n)) ** 2,
                                       weights=t.n))
        return float(np.mean(vals)) if vals else np.nan

    obs = stat(d)
    null = []
    for _ in range(500):
        dd = d.copy()
        dd['sector'] = dd.groupby('qb')['sector'].transform(
            lambda s: RNG.permutation(s.values))
        null.append(stat(dd))
    null = np.array([v for v in null if np.isfinite(v)])
    p = float((null >= obs).mean())
    print(f'{rat}: within-quality-bin sector variance in P = {obs:.4f}  '
          f'(null mean {null.mean():.4f}, p95 {np.percentile(null,95):.4f})  p={p:.4f}')
    print(f'    -> equivalent sector SD in P(match) at fixed behavior = '
          f'{np.sqrt(max(obs-null.mean(),0)):.3f}')
    res[f'{rat}_cond_sector_var'] = obs
    res[f'{rat}_cond_sector_p'] = p
    res[f'{rat}_cond_sector_sd'] = float(np.sqrt(max(obs - null.mean(), 0)))

with open(f'{OUT}/stage6_results.json', 'w') as fh:
    json.dump(res, fh, indent=2)
print(f'\nwrote {OUT}/stage6_results.json')
