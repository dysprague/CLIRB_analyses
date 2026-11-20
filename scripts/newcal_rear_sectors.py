"""Sector breakdown of the ear_mean_z rear detector, on a per-ERA arena frame.

Reuses the cached traces from scripts/newcal_rear_detector.py (ear_mean_z for
each modality, DANNCE COM, per-session arena fit), so nothing expensive is
recomputed.

WHY A PER-ERA ARENA RATHER THAN PER SESSION.  `fit_arena` needs the animal to
visit most of the perimeter.  R9's 2026-10-22 session has only 7 rears and very
low coverage, and its own fit lands 80 mm from R7's and R8's on the same day
with a fit cost of 56 mm against their 2.8-3.8 mm.  The arena is physically one
object per era, so it is fitted once per era from the pooled, radius-trimmed
DANNCE COM of every session in that era and that single frame is used for every
session in it.

Two operating points are reported:
  calibrated  the best matched-threshold setting for the most recent era
  stress      a deliberately harder setting, to probe for latent spatial
              structure: when the detector is pushed until it does make
              mistakes, do those mistakes cluster anywhere in the arena?

Per-sector recall carries a Wilson 95% interval, because a sector with 20 rears
and no misses is not evidence of better performance than one with 40.

Writes results/newcal/rear_sector_by_era.csv
"""
import os, sys, glob
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial.geometry import fit_arena, sector_index, ArenaFrame
from scripts.newcal_rear_detector import detect_rears_fast, match_overlap, peaks_of
from scripts.newcal_eras import ERAS, ERA_SHORT

OUT = 'results/newcal'
TRACEDIR = f'{OUT}/rear_traces'
SECTOR_WIDTHS = [60, 30]
#: The setting the rig is ACTUALLY RUNNING since 2026-10-26 (campy_config
#: reward_params: head_min 140, reset_threshold 100, hold_time 0.5 s at 20 fps).
#: Using the deployed values rather than a freshly swept optimum makes the
#: sector analysis answer the operational question directly. The swept optimum
#: for that era, (120, 100, 5), is inside the same plateau and also scores 1.000.
CALIBRATED = dict(rise=140, reset=100, min_frames=10)
STRESS = dict(rise=90, reset=50, min_frames=5)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_traces():
    out = {}
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    era = {(r.rat, r.session): r.era for r in inv.itertuples()}
    for f in sorted(glob.glob(f'{TRACEDIR}/*.npz')):
        base = os.path.basename(f)[:-4]
        rat, session = base.split('_', 1)
        if (rat, session) not in era:
            continue
        z = np.load(f)
        out[(rat, session)] = dict(zs=z['zs'], zd=z['zd'], com=z['com'],
                                   era=era[(rat, session)])
    return out


def era_arenas(traces):
    """One arena per era, fitted on pooled radius-trimmed DANNCE COM."""
    arenas = {}
    for e in ERAS:
        pts = [v['com'][::10] for v in traces.values() if v['era'] == e]
        if not pts:
            continue
        P = np.concatenate(pts)
        P = P[np.isfinite(P).all(axis=1)]
        c0 = np.median(P, axis=0)
        r = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
        P = P[r <= np.percentile(r, 99.5)]        # drop tracking blow-ups
        a, cost = fit_arena(P)
        arenas[e] = (a, float(cost), len(P))
        print(f'{e}: arena centre=({a.cx:.1f},{a.cy:.1f}) r={a.radius:.1f} '
              f'fit_cost={cost:.2f}mm  ({len(P)} pooled COM samples)', flush=True)
    return arenas


def main():
    traces = load_traces()
    arenas = era_arenas(traces)

    rows = []
    for op_name, op in [('calibrated', CALIBRATED), ('stress', STRESS)]:
        for (rat, session), t in traces.items():
            e = t['era']
            if e not in arenas:
                continue
            arena = arenas[e][0]
            zs, zd, com = t['zs'], t['zd'], t['com']
            rs = detect_rears_fast(zs, op['rise'], op['reset'], op['min_frames'])
            rd = detect_rears_fast(zd, op['rise'], op['reset'], op['min_frames'])
            pairs, _ = match_overlap(rs, rd)
            sm = np.zeros(len(rs), bool); dm = np.zeros(len(rd), bool)
            for i, j in pairs:
                sm[i] = dm[j] = True
            theta = arena.theta(com)
            rad = arena.radius_of(com)
            err = np.abs(zs - zd)
            for W in SECTOR_WIDTHS:
                sec = sector_index(theta, W)
                sd = sector_index(theta[rd[:, 1]], W) if len(rd) else np.empty(0, int)
                ss = sector_index(theta[rs[:, 1]], W) if len(rs) else np.empty(0, int)
                for k in range(int(round(360 / W))):
                    md, ms, occ = sd == k, ss == k, sec == k
                    rows.append(dict(
                        op=op_name, rat=rat, session=session, era=e,
                        sector_w=W, sector=k,
                        centre_deg=-180 + W * (k + 0.5),
                        n_dannce=int(md.sum()), n_sleap=int(ms.sum()),
                        tp_recall=int(dm[md].sum()) if len(rd) else 0,
                        tp_precision=int(sm[ms].sum()) if len(rs) else 0,
                        occupancy=int(occ.sum()),
                        med_abs_err=float(np.median(err[occ])) if occ.any() else np.nan,
                        p90_abs_err=float(np.percentile(err[occ], 90)) if occ.any() else np.nan,
                        med_radius=float(np.median(rad[occ])) if occ.any() else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/rear_sector_by_era.csv', index=False)

    pd.set_option('display.width', 240)
    for op_name in ['calibrated', 'stress']:
        for W in SECTOR_WIDTHS:
            for e in ERAS:
                g = d[(d.op == op_name) & (d.sector_w == W) & (d.era == e)]
                if not len(g):
                    continue
                a = g.groupby('sector').agg(
                    centre=('centre_deg', 'first'), n_d=('n_dannce', 'sum'),
                    n_s=('n_sleap', 'sum'), tpr=('tp_recall', 'sum'),
                    tpp=('tp_precision', 'sum'), occ=('occupancy', 'sum'),
                    err=('med_abs_err', 'median'), p90=('p90_abs_err', 'median'),
                    rad=('med_radius', 'median')).reset_index()
                a['occ%'] = 100 * a.occ / a.occ.sum()
                a['rears/1k'] = 1000 * a.n_d / a.occ.replace(0, np.nan)
                a['recall'] = a.tpr / a.n_d.replace(0, np.nan)
                a['prec'] = a.tpp / a.n_s.replace(0, np.nan)
                lo, hi = zip(*[wilson(k, n) for k, n in zip(a.tpr, a.n_d)])
                a['r_lo'], a['r_hi'] = lo, hi
                print(f'\n=== {op_name} | {W} deg | {e} '
                      f'(rise={CALIBRATED["rise"] if op_name=="calibrated" else STRESS["rise"]}) '
                      f'| {int(a.n_d.sum())} DANNCE rears ===')
                print(a[['centre', 'occ%', 'n_d', 'rears/1k', 'n_s', 'recall',
                         'r_lo', 'r_hi', 'prec', 'err', 'p90', 'rad']]
                      .round(3).to_string(index=False))
                miss = int((a.n_d - a.tpr).sum()); fp = int((a.n_s - a.tpp).sum())
                print(f'    total misses={miss}  false positives={fp}   '
                      f'per-frame |error| across sectors: '
                      f'{a.err.min():.2f}-{a.err.max():.2f} mm '
                      f'({a.err.max()/a.err.min():.2f}x)')
    print(f'\nwrote {OUT}/rear_sector_by_era.csv')


if __name__ == '__main__':
    main()
