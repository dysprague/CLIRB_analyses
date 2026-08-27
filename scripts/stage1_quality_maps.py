"""Stage 1 — truth-free tracking quality vs arena angle, all sessions.

Occupancy-weighted (every frame), and restricted to rear frames, so a
confound that only appears at rear height is visible separately.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from scipy import stats

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from clirb.spatial.geometry import sector_index, sector_center

OUT = 'results/spatial_confound'
SECTOR_W = 30
NSEC = int(360 / SECTOR_W)
METRICS = ['reproj_px', 'mean_conf', 'n_cams_conf50', 'bone_cv', 'jitter',
           'fov_min', 'nan_frac']


def load_q(rat, session):
    p = f'data/cache_quality/{rat}/{session}.npz'
    if not os.path.isfile(p):
        return None
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    d['fov_min'] = np.nanmin(d['fov_margin'], axis=1)
    d['conf_cam0'] = d['conf_per_cam'][:, 0]
    d['conf_cam1'] = d['conf_per_cam'][:, 1]
    d['conf_cam2'] = d['conf_per_cam'][:, 2]
    return d


def main():
    rears = pd.read_csv(f'{OUT}/rears.csv')
    rows_all, rows_rear = [], []
    sess = sorted({(r, os.path.basename(f)[:-4])
                   for r in ['R4', 'R5', 'R6']
                   for f in glob.glob(f'data/cache_quality/{r}/*.npz')})
    print(f'{len(sess)} sessions with a quality cache')
    for rat, s in sess:
        q = load_q(rat, s)
        if q is None:
            continue
        sec = sector_index(q['theta'], SECTOR_W)
        # --- occupancy-weighted (all frames)
        for k in range(NSEC):
            m = sec == k
            if m.sum() < 100:
                continue
            rec = dict(rat=rat, session=s, sector=k, n=int(m.sum()))
            for met in METRICS + ['conf_cam0', 'conf_cam1', 'conf_cam2']:
                v = q[met][m].astype(float)
                rec[met] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
            rows_all.append(rec)
        # --- rear frames only
        rr = rears[(rears.rat == rat) & (rears.session == s)]
        if len(rr):
            mask = np.zeros(len(q['theta']), bool)
            for a, b in zip(rr.start_frame.values, rr.end_frame.values):
                mask[a:min(b + 1, len(mask))] = True
            secr = sec[mask]
            for k in range(NSEC):
                m = secr == k
                if m.sum() < 50:
                    continue
                rec = dict(rat=rat, session=s, sector=k, n=int(m.sum()))
                for met in METRICS:
                    v = q[met][mask][m].astype(float)
                    rec[met] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
                rows_rear.append(rec)

    A = pd.DataFrame(rows_all); R = pd.DataFrame(rows_rear)
    A.to_csv(f'{OUT}/quality_by_sector_allframes.csv', index=False)
    R.to_csv(f'{OUT}/quality_by_sector_rearframes.csv', index=False)

    # ---- P(match|rear) per rat x sector, from the event table
    ev = rears[(rears.min_port_distance > 0) & (~rears.in_gate_zone)]
    pm = ev.groupby(['rat', 'sector_30']).agg(
        n=('matched', 'size'), p=('matched', 'mean')).reset_index()
    pm = pm[pm.n >= 60].rename(columns={'sector_30': 'sector'})

    print('\n' + '=' * 78)
    print('QUALITY vs SECTOR (rear frames, session-median then rat-median)')
    print('=' * 78)
    res = {}
    for rat in ['R4', 'R5', 'R6']:
        g = R[R.rat == rat].groupby('sector')[METRICS].median()
        g['theta_c'] = [sector_center(i, SECTOR_W) for i in g.index]
        gp = pm[pm.rat == rat].set_index('sector')
        g['P_match'] = gp['p']; g['n_rears'] = gp['n']
        print(f'\n{rat}')
        print(g[['theta_c', 'n_rears', 'P_match', 'reproj_px', 'mean_conf',
                 'n_cams_conf50', 'bone_cv', 'jitter', 'fov_min']].to_string(
              float_format=lambda v: f'{v:.3f}'))
        # correlate each metric with P(match) across sectors
        gg = g.dropna(subset=['P_match'])
        print('  corr with P(match) across sectors:')
        for met in METRICS:
            v = gg[met].values
            if np.isfinite(v).sum() < 5:
                continue
            r, p = stats.spearmanr(v, gg['P_match'].values, nan_policy='omit')
            print(f'    {met:15s} rho={r:+.3f}  p={p:.3f}')
            res[f'{rat}_{met}_rho'] = float(r); res[f'{rat}_{met}_p'] = float(p)
    with open(f'{OUT}/stage1_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f'\nwrote {OUT}/quality_by_sector_*.csv and stage1_results.json')


if __name__ == '__main__':
    main()
