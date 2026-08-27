"""Stage 7 — re-define rears from DANNCE, then re-run the §3 conditioning.

The threat (M3): rears are currently defined by a threshold on SLEAP PC0, so the
event set is selected by the very bias under test — a collider that conditioning
cannot repair.

Fix: put DANNCE PC0 on the SLEAP scale using the per-session SECTOR-BALANCED
modality offset (a single constant per session, so it carries no spatial bias),
then run the identical hysteresis detector on it. Event selection is then
DANNCE-driven; only the scale calibration touches SLEAP, and it is spatially flat.

Outputs: results/spatial_confound/rears_dannce_detected.csv
         results/spatial_confound/dannce_lib_<rat>.npz
"""
import os, sys, glob
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du
from clirb.spatial.geometry import ARENA, sector_index

OUT = 'results/spatial_confound'
SECTOR_W, NSEC, N_PER_SECTOR, PAD = 30, 12, 80, 30
RNG = np.random.default_rng(0)


def balanced_offset(s2, d2, sec_f):
    offs = []
    for k in range(NSEC):
        m = sec_f == k
        if m.sum() >= 200:
            offs.append(s2[m].mean(axis=0) - d2[m].mean(axis=0))
    return np.mean(offs, axis=0) if offs else s2.mean(axis=0) - d2.mean(axis=0)


def align_to(pcs, start, end, target, pad=PAD, w=du.W):
    e_min = max(start - pad, w - 1); e_max = min(end + pad, len(pcs) - 1)
    if e_max < e_min:
        return -1, np.nan
    ends = np.arange(e_min, e_max + 1)
    idx = (ends - (w - 1))[:, None] + np.arange(w)[None, :]
    mse = ((pcs[idx, :2] - target[None]) ** 2).mean(axis=(1, 2))
    k = int(np.argmin(mse))
    return int(ends[k]), float(mse[k])


def collect(rat):
    recs = []
    for i, f in enumerate(sorted(glob.glob(f'data/cache/{rat}/*.npz'))):
        session = os.path.basename(f)[:-4]
        raw = sc.get_raw_session(rat, session, want_dual=True)
        if raw is None or not raw.get('has_dual', False):
            continue
        t = tp.load_template(rat, session)
        if t is None:
            continue
        s_pcs, d_pcs = du.project_both(raw, t)
        if d_pcs is None:
            continue
        pcu = np.asarray(t['pcs_to_use']).ravel().astype(int)
        n = min(len(s_pcs), len(d_pcs))
        s2 = s_pcs[:n, pcu[:2]].astype(np.float32)
        d2 = d_pcs[:n, pcu[:2]].astype(np.float32)
        com = raw['triang_3d'][:n].mean(axis=1)
        theta_f = ARENA.theta(com[:, :2]); rad_f = ARENA.radius_of(com[:, :2])
        sec_f = sector_index(theta_f, SECTOR_W)
        off = balanced_offset(s2, d2, sec_f)

        # --- DANNCE-defined rears, on the SLEAP scale
        d0 = d2[:, 0] + off[0]
        rears_d = [r for r in ev.detect_rears(d0) if r[1] < n]
        # --- SLEAP-defined rears, for the side-by-side rate comparison
        rears_s = [r for r in ev.detect_rears(s_pcs[:, pcu[0]]) if r[1] < n]
        matches, _ = ev.replay_rule_offline(s_pcs, t)
        if not rears_d:
            continue
        rs = np.array([r[0] for r in rears_d], int)
        re_ = np.array([r[1] for r in rears_d], int)
        hit, mend = ev.rear_matched(rs, re_, matches)
        gate = ev.port_gate_frame(re_, mend, n)
        port = np.asarray(raw['port_xy']).ravel()[:2]
        recs.append(dict(session=session, s2=s2, d2=d2 + off[None, :], rs=rs, re=re_,
                         hit=hit, sec=sec_f[gate], theta=theta_f[gate],
                         pdist=np.linalg.norm(com[gate, :2] - port[None, :], axis=1),
                         mpd=float(raw['min_port_distance']),
                         n_rears_sleap=len(rears_s), n_rears_dannce=len(rears_d),
                         sec_f=sec_f, rad_f=rad_f,
                         sleap_sec=sector_index(
                             theta_f[np.clip([r[1] for r in rears_s], 0, n - 1)], SECTOR_W),
                         template_file=t['_file']))
        if i % 15 == 0:
            print(f'  {rat} [{i+1}] {session}: DANNCE {len(rears_d)} / SLEAP {len(rears_s)} rears',
                  flush=True)
    return recs


def main():
    rows, rate_rows = [], []
    for rat in ['R4', 'R5', 'R6']:
        print(f'=== {rat}', flush=True)
        recs = collect(rat)
        if not recs:
            continue

        # ---- detector-rate comparison by sector (wall frames only)
        occ = np.zeros(NSEC); nd = np.zeros(NSEC); ns = np.zeros(NSEC)
        for r in recs:
            w = r['rad_f'] > 250
            occ += np.bincount(r['sec_f'][w], minlength=NSEC)
            nd += np.bincount(r['sec'], minlength=NSEC)
            ns += np.bincount(r['sleap_sec'], minlength=NSEC)
        for k in range(NSEC):
            rate_rows.append(dict(rat=rat, sector=k, theta=k * 30 - 165,
                                  wall_min=occ[k] / 20 / 60,
                                  dannce_rears=int(nd[k]), sleap_rears=int(ns[k])))

        # ---- DANNCE reference from DANNCE-DETECTED rears, sector-stratified.
        # Seed = mean window ending at rear_end (a DANNCE-derived landmark),
        # then re-align every rear to that seed.
        seed_wins = []
        for r in recs:
            for k in range(len(r['rs'])):
                e = int(r['re'][k])
                if e >= du.W - 1 and e < len(r['d2']):
                    seed_wins.append(r['d2'][e - du.W + 1:e + 1])
        if len(seed_wins) < 50:
            continue
        seed = np.stack(seed_wins).mean(axis=0)

        aligned, secs = [], []
        for r in recs:
            for k in range(len(r['rs'])):
                e, _ = align_to(r['d2'], r['rs'][k], r['re'][k], seed)
                if e >= du.W - 1:
                    aligned.append(r['d2'][e - du.W + 1:e + 1]); secs.append(r['sec'][k])
        A = np.stack(aligned); S = np.asarray(secs)
        pick = np.concatenate([RNG.choice(np.where(S == s)[0],
                                          min((S == s).sum(), N_PER_SECTOR), replace=False)
                               for s in np.unique(S)])
        ref = A[pick].mean(axis=0)
        for _ in range(3):
            mse = ((A[pick] - ref[None]) ** 2).mean(axis=(1, 2))
            ref = A[pick][mse <= np.percentile(mse, 80)].mean(axis=0)
        print(f'  {rat}: reference from {len(pick)} DANNCE-detected stratified windows',
              flush=True)

        lib, lsec = [], []
        for r in recs:
            for k in range(len(r['rs'])):
                de, dmse = align_to(r['d2'], r['rs'][k], r['re'][k], ref)
                if de < du.W - 1:
                    continue
                lib.append(r['d2'][de - du.W + 1:de + 1]); lsec.append(r['sec'][k])
                rows.append(dict(rat=rat, session=r['session'], sector=int(r['sec'][k]),
                                 theta=float(r['theta'][k]), matched=bool(r['hit'][k]),
                                 port_dist=float(r['pdist'][k]),
                                 min_port_distance=r['mpd'], d_dannce=dmse,
                                 template_file=r['template_file']))
        np.savez_compressed(f'{OUT}/dannce_lib_{rat}.npz',
                            library=np.stack(lib).astype(np.float32),
                            library_sector=np.asarray(lsec, dtype=np.int16), reference=ref)
        print(f'  {rat}: {len(lib)} DANNCE-detected rears scored', flush=True)

    pd.DataFrame(rows).to_csv(f'{OUT}/rears_dannce_detected.csv', index=False)
    pd.DataFrame(rate_rows).to_csv(f'{OUT}/detector_rate_comparison.csv', index=False)
    print(f'\nwrote rears_dannce_detected.csv ({len(rows)}) + detector_rate_comparison.csv')


if __name__ == '__main__':
    main()
