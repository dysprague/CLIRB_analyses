"""Build the per-rear event table -> results/spatial_confound/rears.parquet

One row per detected rear, across every locally cached session.
"""
import os, sys, glob
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO)
os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb import templates as tp
from clirb.spatial.geometry import ARENA, sector_index, apex_height

OUT = 'results/spatial_confound/rears.csv'
CACHE_ROOT = 'data/cache'


def cached_sessions():
    out = []
    for rat in ['R4', 'R5', 'R6']:
        for f in sorted(glob.glob(f'{CACHE_ROOT}/{rat}/*.npz')):
            out.append((rat, os.path.basename(f)[:-4]))
    return out


def build():
    rows = []
    sessions = cached_sessions()
    # chronological index per rat
    sidx = {}
    for rat in ['R4', 'R5', 'R6']:
        ss = sorted(s for r, s in sessions if r == rat)
        sidx.update({(rat, s): i for i, s in enumerate(ss)})

    for n, (rat, session) in enumerate(sessions):
        c = sc.get_raw_session(rat, session, want_dual=False)
        if c is None:
            print(f'  {rat}/{session}: no raw'); continue
        tmpl = tp.load_template(rat, session)
        if tmpl is None:
            print(f'  {rat}/{session}: no template'); continue

        triang = c['triang_3d']
        pcs_full, _ = ev.project_session_to_pcs(triang, tmpl, flip_z=True)
        pcs_to_use = np.asarray(tmpl['pcs_to_use']).ravel().astype(int)
        pc0 = pcs_full[:, pcs_to_use[0]]
        pcs_used = pcs_full[:, pcs_to_use[:ev.N_COMPONENTS]]
        pc_template = tmpl['template'][:, :ev.N_COMPONENTS]

        matches, _ = ev.replay_rule_offline(pcs_full, tmpl)
        rears = ev.detect_rears(pc0)
        if not rears:
            print(f'  {rat}/{session}: 0 rears'); continue
        rs = np.array([r[0] for r in rears], int)
        re_ = np.array([r[1] for r in rears], int)
        n_above = np.array([r[2] for r in rears], int)

        hit, mend = ev.rear_matched(rs, re_, matches)
        gate = ev.port_gate_frame(re_, mend, len(triang))

        com = triang.mean(axis=1)
        gxy = com[gate, :2]
        port = np.asarray(c['port_xy']).ravel()[:2]
        pdist = np.linalg.norm(gxy - port[None, :], axis=1)
        theta = ARENA.theta(gxy)
        rad = ARENA.radius_of(gxy)

        # licks / attempts
        lf = ev.lick_frames_from_idx(c['lick_idx_sleap_cams'], len(triang))
        lb = ev.lick_bouts(lf)
        a_ri, a_lf = ev.attempts(re_, lb)
        is_att = np.zeros(len(rears), bool); att_lick = np.full(len(rears), -1)
        is_att[a_ri] = True; att_lick[a_ri] = a_lf

        # online triggers
        trig = np.zeros(len(rears), bool)
        if c['has_task_log'] and len(c['behavior_log_frames']):
            fm = c['frame_mapping_cam0']
            tp_pf = c['behavior_log_frames']
            tp_pf = tp_pf[tp_pf < len(fm)]
            tcam0 = fm[tp_pf]
            for k in range(len(rears)):
                trig[k] = np.any((tcam0 >= rs[k]) & (tcam0 <= re_[k] + ev.ALIGN_BUF_FR))

        mpd = float(c['min_port_distance'])
        for k in range(len(rears)):
            ae, amse = ev.best_align_end(rs[k], re_[k], pcs_used, pc_template)
            rows.append(dict(
                rat=rat, session=session, session_idx=sidx[(rat, session)],
                date=session[:10], train_state=float(c['train_state']),
                min_port_distance=mpd,
                template_file=tmpl['_file'], template_sha=tmpl['_sha'],
                rear_idx=k, start_frame=int(rs[k]), end_frame=int(re_[k]),
                gate_frame=int(gate[k]), align_end=int(ae), n_above=int(n_above[k]),
                duration_s=float((re_[k] - rs[k]) / ev.FPS),
                x=float(gxy[k, 0]), y=float(gxy[k, 1]), z=float(com[gate[k], 2]),
                theta=float(theta[k]), radius=float(rad[k]),
                wall_dist=float(ARENA.radius - rad[k]),
                sector_20=int(sector_index(theta[k], 20)),
                sector_30=int(sector_index(theta[k], 30)),
                sector_45=int(sector_index(theta[k], 45)),
                apex_z=apex_height(triang, rs[k], re_[k]),
                port_dist=float(pdist[k]),
                in_gate_zone=bool(pdist[k] < mpd) if mpd > 0 else False,
                matched=bool(hit[k]), match_end=int(mend[k]),
                online_triggered=bool(trig[k]),
                align_mse=float(amse),
                is_attempt=bool(is_att[k]), lick_bout_frame=int(att_lick[k]),
                latency_s=float((att_lick[k] - re_[k]) / ev.FPS) if is_att[k] else np.nan,
                n_frames=len(triang),
            ))
        if n % 20 == 0:
            print(f'  [{n+1}/{len(sessions)}] {rat}/{session}: '
                  f'{len(rears)} rears, {hit.sum()} matched')

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}: {len(df)} rears, {df.groupby(["rat","session"]).ngroups} sessions')
    print(df.groupby('rat').agg(n=('matched', 'size'), p=('matched', 'mean')).round(3))
    return df


if __name__ == '__main__':
    build()
