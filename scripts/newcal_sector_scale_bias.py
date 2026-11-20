"""Is the DANNCE/SLEAP size bias spatially structured, and is it centred on the
water port?

Hypothesis under test (the user's): the mechanism added at the water port since
the September sessions obstructs some DANNCE views, so the reconstruction
should be worse near the port and roughly normal elsewhere.

Measured as the DANNCE-to-SLEAP ratio of skeleton bone lengths, per arena
sector.  The ratio is posture-free by construction -- both modalities see the
same frames, so whatever the animal is doing cancels -- which is what makes it
safe to compare across sectors that the animal uses very differently.

The arena is refitted per era from pooled DANNCE COM (the fixed ARENA in
clirb.spatial.geometry was fitted in the old room), and the port's own sector
is located from each session's `water_port_location.npy` so "near the port" is
read off the data rather than assumed.

Three eras: September (before the port mechanism), Oct 22-23 (mechanism in,
task off) and Oct 26 (mechanism in, task on).

Writes results/newcal/sector_scale_bias.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import dual as du
from clirb.spatial.geometry import fit_arena, sector_index
from config import NODE_IDX as NI

OUT = 'results/newcal'
SECTOR_W = 60
PAIRS = {'SpineF-SpineM': ('SpineF', 'SpineM'), 'HipL-KneeL': ('HipL', 'KneeL'),
         'ShoulderL-ElbowL': ('ShoulderL', 'ElbowL'),
         'Snout-SpineF': ('Snout', 'SpineF'), 'SpineF-TailBase': ('SpineF', 'TailBase')}
MIN_FRAMES = 300


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    sept = inv[inv.era == 'sept_baseline'].sort_values('session').tail(9)
    use = pd.concat([sept, inv[inv.era.isin(['oct_b', 'oct_26'])]])

    # one arena per era, from pooled trimmed DANNCE COM
    coms = {}
    for r in use.itertuples():
        raw = sc.get_raw_session(r.rat, r.session, want_dual=True)
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        n = min(len(raw['triang_3d']), len(d3))
        coms.setdefault(r.era, []).append(
            np.asarray(d3[:n], np.float32).mean(axis=1)[::10, :2])
    arenas = {}
    for e, lst in coms.items():
        P = np.concatenate(lst)
        P = P[np.isfinite(P).all(axis=1)]
        c0 = np.median(P, axis=0)
        rad = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
        a, cost = fit_arena(P[rad <= np.percentile(rad, 99.5)])
        arenas[e] = a
        print(f'{e}: arena ({a.cx:.0f},{a.cy:.0f}) r={a.radius:.0f} cost={cost:.1f}mm', flush=True)

    rows = []
    for r in use.itertuples():
        raw = sc.get_raw_session(r.rat, r.session, want_dual=True)
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        n = min(len(raw['triang_3d']), len(d3))
        sl = np.asarray(raw['triang_3d'][:n], np.float32)
        dn = np.asarray(d3[:n], np.float32)
        arena = arenas[r.era]
        com = sl.mean(axis=1)[:, :2]
        sec = sector_index(arena.theta(com), SECTOR_W)
        port = np.asarray(raw['port_xy']).ravel()[:2]
        port_sec = (int(sector_index(np.atleast_1d(arena.theta(port[None, :]))[0], SECTOR_W))
                    if np.isfinite(port).all() else -1)
        for k in range(int(360 // SECTOR_W)):
            m = sec == k
            if m.sum() < MIN_FRAMES:
                continue
            rec = dict(rat=r.rat, session=r.session, era=r.era, sector=k,
                       centre_deg=-180 + SECTOR_W * (k + 0.5),
                       is_port_sector=(k == port_sec),
                       frac=float(m.mean()), n_frames=int(m.sum()))
            for nm, (a, b) in PAIRS.items():
                ia, ib = NI[a], NI[b]
                rec[nm] = float(np.median(np.linalg.norm(dn[m, ia] - dn[m, ib], axis=-1)) /
                                np.median(np.linalg.norm(sl[m, ia] - sl[m, ib], axis=-1)))
            rows.append(rec)
        print(f'  {r.rat}/{r.session} ({r.era}) port sector={port_sec}', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/sector_scale_bias.csv', index=False)
    pd.set_option('display.width', 240)
    for e in ['sept_baseline', 'oct_b', 'oct_26']:
        s = d[d.era == e]
        if not len(s):
            continue
        g = s.groupby('sector').agg(
            centre=('centre_deg', 'first'), port=('is_port_sector', 'max'),
            occ=('frac', 'median'), **{k: (k, 'median') for k in PAIRS}).reset_index()
        print(f'\n=== {e} — DANNCE/SLEAP bone ratio by {SECTOR_W}° sector '
              f'({s.session.nunique()} sessions) ===')
        print(g.round(3).to_string(index=False))
        pm = s[s.is_port_sector]
        om = s[~s.is_port_sector]
        if len(pm) and len(om):
            print('  port sector vs the rest:')
            for k in PAIRS:
                print(f'    {k:18s} port={pm[k].median():.3f}  elsewhere={om[k].median():.3f}  '
                      f'Δ={pm[k].median()-om[k].median():+.3f}')
    print(f'\nwrote {OUT}/sector_scale_bias.csv')


if __name__ == '__main__':
    main()
