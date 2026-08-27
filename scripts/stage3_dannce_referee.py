"""Stage 3 — DANNCE referee (v2).

Builds a per-rat DANNCE reference for the SAME behavior the online rule targets,
without letting SLEAP match OUTCOME select anything:

  1. per-session modality offset  mu = mean(sleap_pcs) - mean(dannce_pcs)
     computed over a SECTOR-BALANCED frame sample (so it carries no spatial bias)
  2. seed = production_template - mu   -> the rule's target, expressed in DANNCE space
     (this fixes the phase: R4/R6 templates are the rear DESCENT, R5's is the ASCENT)
  3. align every rear's DANNCE window to the seed over [rs-30, re+30]
  4. average a SECTOR-STRATIFIED subsample of those aligned windows -> reference
  5. d_dannce = min-MSE of the DANNCE window to the reference   (behavior quality)
  6. residual snippets eps = sleap_window - dannce_window at the DANNCE-aligned end

Nothing above uses `matched`.
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
SECTOR_W = 30
NSEC = 12
N_PER_SECTOR = 80
SEARCH_PAD = 30
RNG = np.random.default_rng(0)


def balanced_offset(s2, d2, sec_per_frame):
    """mean(sleap) - mean(dannce), averaging sectors equally."""
    offs = []
    for k in range(NSEC):
        m = sec_per_frame == k
        if m.sum() < 200:
            continue
        offs.append(s2[m].mean(axis=0) - d2[m].mean(axis=0))
    if not offs:
        return s2.mean(axis=0) - d2.mean(axis=0)
    return np.mean(offs, axis=0)


def align_to(pcs, start, end, target, pad=SEARCH_PAD, w=du.W):
    e_min = max(start - pad, w - 1)
    e_max = min(end + pad, len(pcs) - 1)
    if e_max < e_min:
        return -1, np.nan
    ends = np.arange(e_min, e_max + 1)
    idx = (ends - (w - 1))[:, None] + np.arange(w)[None, :]
    win = pcs[idx, :2]
    mse = ((win - target[None]) ** 2).mean(axis=(1, 2))
    k = int(np.argmin(mse))
    return int(ends[k]), float(mse[k])


def collect(rat):
    recs = []
    files = sorted(glob.glob(f'data/cache/{rat}/*.npz'))
    for i, f in enumerate(files):
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

        matches, _ = ev.replay_rule_offline(s_pcs, t)
        rears = [r for r in ev.detect_rears(s_pcs[:, pcu[0]]) if r[1] < n]
        if not rears:
            continue
        rs = np.array([r[0] for r in rears], int)
        re_ = np.array([r[1] for r in rears], int)
        hit, mend = ev.rear_matched(rs, re_, matches)
        gate = ev.port_gate_frame(re_, mend, n)
        com = raw['triang_3d'][:n].mean(axis=1)
        theta_f = ARENA.theta(com[:, :2])
        sec_f = sector_index(theta_f, SECTOR_W)
        port = np.asarray(raw['port_xy']).ravel()[:2]
        recs.append(dict(session=session, s2=s2, d2=d2, rs=rs, re=re_, hit=hit,
                         theta=theta_f[gate], sec=sec_f[gate],
                         offset=balanced_offset(s2, d2, sec_f),
                         pdist=np.linalg.norm(com[gate, :2] - port[None, :], axis=1),
                         mpd=float(raw['min_port_distance']),
                         train_state=float(raw['train_state']),
                         template=t, template_file=t['_file']))
        if i % 15 == 0:
            print(f'  {rat} [{i+1}/{len(files)}] {session}: {len(rears)} rears', flush=True)
    return recs


def main():
    all_rows = []
    for rat in ['R4', 'R5', 'R6']:
        print(f'=== {rat}: collecting', flush=True)
        recs = collect(rat)
        if not recs:
            continue
        print(f'  {rat}: {len(recs)} dual sessions', flush=True)

        # ---- pass 1: align each rear's DANNCE window to (template - offset)
        aligned, secs, keys = [], [], []
        for ri, r in enumerate(recs):
            seed = r['template']['template'][:, :2] - r['offset'][None, :]
            for k in range(len(r['rs'])):
                e, m = align_to(r['d2'], r['rs'][k], r['re'][k], seed)
                if e < 0:
                    continue
                aligned.append(r['d2'][e - du.W + 1:e + 1])
                secs.append(r['sec'][k]); keys.append((ri, k, e))
        if len(aligned) < 200:
            continue
        A = np.stack(aligned); S = np.asarray(secs)

        # ---- pass 2: sector-stratified average -> the DANNCE reference
        pick = []
        for s in np.unique(S):
            ii = np.where(S == s)[0]
            pick.extend(RNG.choice(ii, size=min(len(ii), N_PER_SECTOR), replace=False))
        pick = np.array(pick)
        ref = A[pick].mean(axis=0)
        for _ in range(2):                       # trim-and-reaverage
            mse = ((A[pick] - ref[None]) ** 2).mean(axis=(1, 2))
            ref = A[pick][mse <= np.percentile(mse, 80)].mean(axis=0)
        print(f'  {rat}: reference from {len(pick)} stratified windows over '
              f'{len(np.unique(S))} sectors', flush=True)

        # ---- pass 3: score every rear against the reference + residuals
        snips, snip_sec, lib, lib_sec = [], [], [], []
        for r in recs:
            for k in range(len(r['rs'])):
                de, dmse = align_to(r['d2'], r['rs'][k], r['re'][k], ref)
                resid = np.nan
                if de >= du.W - 1 and de < len(r['s2']):
                    a, b = de - du.W + 1, de + 1
                    rr = r['s2'][a:b] - r['d2'][a:b]
                    resid = float(np.abs(rr).mean())
                    snips.append(rr); snip_sec.append(r['sec'][k])
                    lib.append(r['d2'][a:b]); lib_sec.append(r['sec'][k])
                all_rows.append(dict(
                    rat=rat, session=r['session'], rear_idx=k,
                    theta=float(r['theta'][k]), sector=int(r['sec'][k]),
                    matched=bool(r['hit'][k]), port_dist=float(r['pdist'][k]),
                    min_port_distance=r['mpd'], train_state=r['train_state'],
                    template_file=r['template_file'],
                    d_dannce=dmse, resid_abs=resid, d_end=de))
        np.savez_compressed(
            f'{OUT}/residual_snippets_{rat}.npz',
            snippets=np.stack(snips).astype(np.float32),
            sector=np.asarray(snip_sec, dtype=np.int16),
            library=np.stack(lib).astype(np.float32),
            library_sector=np.asarray(lib_sec, dtype=np.int16),
            reference=ref)
        print(f'  {rat}: {len(snips)} snippets + library saved', flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(f'{OUT}/rears_dannce.csv', index=False)
    print(f'\nwrote {OUT}/rears_dannce.csv: {len(df)} rears')
    from scipy import stats as st
    for rat, g in df.groupby('rat'):
        g = g.dropna(subset=['d_dannce'])
        r, p = st.spearmanr(g.d_dannce, g.matched)
        print(f'  {rat}: n={len(g)} median d_dannce={g.d_dannce.median():.0f} '
              f'rho(d_dannce, matched)={r:+.3f} p={p:.1e}')


if __name__ == '__main__':
    main()
