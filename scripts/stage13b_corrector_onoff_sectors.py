"""Stage 13b — does turning the corrector off change the sector reward gradient?

Extends stage13's bone-CV natural experiment to the actual reward-rule outcome.
`rears.csv` already contains a handful of corrector-OFF sessions (2026-07-29,
07-30, 08-03 for R4/R5) that were never revisited; several more corrector-OFF
sessions exist in the local cache (through 2026-08-26) but were never run
through build_event_table.py. This runs the SAME per-session pipeline
(clirb.events, the single source of truth) on those additional sessions for
all three rats, so R6 gets OFF-session coverage for the first time, then
compares P(match | rear) by sector between the corrector-ON and corrector-OFF
eras.

This is an OBSERVATIONAL natural experiment, not a controlled A/B toggle --
time, animal experience, and protocol may co-vary with the corrector switch.
Read accordingly.
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb import templates as tp
from clirb.spatial.geometry import ARENA, sector_index, apex_height

OUT = 'results/spatial_confound/corrector_onoff_rears.csv'
OFF_SESSIONS = ['2026_08_05_1', '2026_08_06_1', '2026_08_11_1', '2026_08_25_1', '2026_08_26_1']


def process_session(rat, session, sidx):
    c = sc.get_raw_session(rat, session, want_dual=False)
    if c is None:
        print(f'  {rat}/{session}: no raw'); return []
    tmpl = tp.load_template(rat, session)
    if tmpl is None:
        print(f'  {rat}/{session}: no template'); return []

    triang = c['triang_3d']
    pcs_full, _ = ev.project_session_to_pcs(triang, tmpl, flip_z=True)
    pcs_to_use = np.asarray(tmpl['pcs_to_use']).ravel().astype(int)
    pc0 = pcs_full[:, pcs_to_use[0]]

    matches, _ = ev.replay_rule_offline(pcs_full, tmpl)
    rears = ev.detect_rears(pc0)
    if not rears:
        print(f'  {rat}/{session}: 0 rears'); return []
    rs = np.array([r[0] for r in rears], int)
    re_ = np.array([r[1] for r in rears], int)

    hit, mend = ev.rear_matched(rs, re_, matches)
    gate = ev.port_gate_frame(re_, mend, len(triang))

    com = triang.mean(axis=1)
    gxy = com[gate, :2]
    port = np.asarray(c['port_xy']).ravel()[:2]
    pdist = np.linalg.norm(gxy - port[None, :], axis=1)
    theta = ARENA.theta(gxy)
    mpd = float(c['min_port_distance'])

    rows = []
    for k in range(len(rears)):
        rows.append(dict(
            rat=rat, session=session, session_idx=sidx, date=session[:10],
            theta=float(theta[k]), sector_30=int(sector_index(theta[k], 30)),
            port_dist=float(pdist[k]), min_port_distance=mpd,
            in_gate_zone=bool(pdist[k] < mpd) if mpd > 0 else False,
            matched=bool(hit[k]),
        ))
    print(f'  {rat}/{session}: {len(rears)} rears, {hit.sum()} matched ({hit.mean():.3f})')
    return rows


def main():
    rows = []
    for rat in ['R4', 'R5', 'R6']:
        offs = [s for s in OFF_SESSIONS if os.path.isfile(f'data/cache/{rat}/{s}.npz')]
        for i, s in enumerate(offs):
            rows.extend(process_session(rat, s, 1000 + i))
    new_off = pd.DataFrame(rows)
    new_off['corrector'] = 'OFF'

    old = pd.read_csv('results/spatial_confound/rears.csv')
    old['corrector'] = np.where(old.date < '2026_07_29', 'ON', 'OFF')
    old_off = old[old.corrector == 'OFF'][['rat', 'session', 'session_idx', 'date', 'theta',
                                           'sector_30', 'port_dist', 'min_port_distance',
                                           'in_gate_zone', 'matched', 'corrector']]
    old_on = old[old.corrector == 'ON'][['rat', 'session', 'session_idx', 'date', 'theta',
                                         'sector_30', 'port_dist', 'min_port_distance',
                                         'in_gate_zone', 'matched', 'corrector']]

    combined = pd.concat([old_on, old_off, new_off], ignore_index=True)
    combined.to_csv(OUT, index=False)

    print('\n' + '=' * 74)
    print('P(match | rear), gate zone excluded, by corrector era')
    print('=' * 74)
    sub = combined[(combined.min_port_distance > 0) & (~combined.in_gate_zone)]
    summ = sub.groupby(['rat', 'corrector']).agg(
        n_sessions=('session', 'nunique'), n=('matched', 'size'), p=('matched', 'mean')).round(3)
    print(summ.to_string())

    print('\n' + '=' * 74)
    print('P(match | rear) by sector_30, ON vs OFF (min n=20)')
    print('=' * 74)
    for rat, g in sub.groupby('rat'):
        piv = g.groupby(['sector_30', 'corrector']).matched.agg(['size', 'mean']).unstack()
        print(f'\n{rat}')
        print(piv.round(3).to_string())

    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
