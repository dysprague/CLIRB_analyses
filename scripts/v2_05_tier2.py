"""Corrector v2, step 5 — Tier-2: does correction reduce the SPATIAL reward bias?

For each pose source (uncorrected / each corrector) we push poses through the
online rule and re-measure the spatial-confound metrics.

Template options (both requested):
  (a) EXISTING per-session template  -> the gain on the DEPLOYED rule
  (b) REBUILT sector-balanced template -> the theoretical ceiling

Metrics per source x template option:
  * P(match) by sector, and its spread                   (the headline)
  * mu(theta): mean PC0 residual vs DANNCE, range as % of the rule's tolerance
  * behaviour-conditioned sector effect: within DANNCE-quality bins, the
    n-weighted variance of P(match) across sectors, vs a shuffle null

Frames come from 48-frame segments, so a full 30-frame template window plus
alignment buffer fits inside each one.

NOTE ON FRAMES. triang_keys_3D lives in the CALIBRATION frame (it reprojects at
~17 px once the processed/acquisition bases are bridged), and the template PC
basis was fitted there. Corrector outputs are in DANNCE space, so they are mapped
back through the inverse per-session Procrustes before projection.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb import events as ev, templates as tp
from clirb.spatial.geometry import sector_center
from skeleton import normalize_skeleton_batch

OUT = 'results/corrector_v2'
W, NC = ev.TEMPLATE_LEN, ev.N_COMPONENTS
SECTOR_W, NSEC = 30, 12
RNG = np.random.default_rng(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def inv_transform(P, tf):
    """DANNCE space -> calibration frame (inverse of fit['apply'])."""
    R = np.asarray(tf['R']); s = tf['s']; t = np.asarray(tf['t'])
    flat = P.reshape(-1, 3)
    out = (flat - t) @ np.linalg.inv(s * R).T
    if tf['z']:
        out = out.copy(); out[:, 2] = -out[:, 2]
    return out.reshape(P.shape)


def to_pcs(pose_cal, tmpl):
    """Calibration-frame poses -> template PC space (the online pipeline)."""
    pcs, _ = ev.project_session_to_pcs(pose_cal, tmpl, flip_z=True)
    pcu = np.asarray(tmpl['pcs_to_use']).ravel().astype(int)
    return pcs[:, pcu[:NC]]


def seg_windows(n):
    """End-frame indices inside a segment of length n that admit a full window."""
    return np.arange(W - 1, n)


def match_and_score(pcs_seg, template, bounds, accepted=ev.ACCEPTED_ERRORS):
    """Best (min-MSE) window in the segment; whether the rule would fire."""
    ends = seg_windows(len(pcs_seg))
    if len(ends) == 0:
        return False, np.nan
    idx = (ends - (W - 1))[:, None] + np.arange(W)[None, :]
    win = pcs_seg[idx]
    mse = ((win - template[None]) ** 2).mean(axis=(1, 2))
    oob = ((win >= template + bounds) | (win <= template - bounds)
           ).reshape(len(win), -1).sum(axis=1)
    return bool((oob <= accepted).any()), float(mse.min())


def build_balanced_template(pcs_by_seg, sectors, n_per_sector=40):
    """Option (b): sector-balanced template + max_dev bounds."""
    wins, secs = [], []
    for p, s in zip(pcs_by_seg, sectors):
        if len(p) >= W:
            wins.append(p[:W]); secs.append(s)
    if len(wins) < 18:
        return None, None
    Xw = np.stack(wins); S = np.asarray(secs)
    pick = []
    for u in np.unique(S):
        ii = np.where(S == u)[0]
        pick.extend(RNG.choice(ii, min(len(ii), n_per_sector), replace=False))
    Xs = Xw[np.array(pick)]
    tmpl = Xs.mean(axis=0)
    for _ in range(3):
        mse = ((Xs - tmpl[None]) ** 2).mean(axis=(1, 2))
        tmpl = Xs[mse <= np.percentile(mse, 80)].mean(axis=0)
    lo, hi = Xs.min(axis=0), Xs.max(axis=0)
    return tmpl, np.maximum(tmpl - lo, hi - tmpl)


def cond_sector_variance(df, nboot=400):
    """n-weighted variance of P(match) across sectors WITHIN DANNCE-quality bins,
    against a within-bin sector-shuffle null."""
    d = df.dropna(subset=['d_dannce']).copy()
    if len(d) < 120:
        return np.nan, np.nan
    d['qb'] = pd.qcut(d.d_dannce.rank(method='first'), 5, labels=False, duplicates='drop')

    def stat(f):
        vals = []
        for _, g in f.groupby('qb'):
            t = g.groupby('sector').match.agg(['size', 'mean'])
            t = t[t['size'] >= 8]
            if len(t) >= 3:
                vals.append(np.average((t['mean'] - np.average(t['mean'], weights=t['size']))**2,
                                       weights=t['size']))
        return float(np.mean(vals)) if vals else np.nan
    obs = stat(d)
    null = []
    for _ in range(nboot):
        dd = d.copy()
        dd['sector'] = dd.groupby('qb')['sector'].transform(lambda s: RNG.permutation(s.values))
        v = stat(dd)
        if np.isfinite(v):
            null.append(v)
    null = np.array(null)
    return obs, float((null >= obs).mean()) if len(null) else np.nan


def main():
    z = np.load(f'{OUT}/dataset.npz')
    X, Y, A, XC = z['X'], z['Y'], z['ASP'], z['XCAL']
    M = pd.read_csv(f'{OUT}/dataset_meta.csv')
    TF = json.load(open(f'{OUT}/frame_transforms.json'))
    split = json.load(open(f'{OUT}/split.json'))
    test_sess = {f'{r}/{s}' for r, s in split['test']}
    print(f'dataset {X.shape}; test sessions: {sorted(test_sess)}')

    key = (M.rat + '/' + M.session).values

    sys.path.insert(0, 'scripts')
    from v2_04_train import BUILD, CTX
    # 'deployed' = triang_keys_3D, the bone-regularised pose the rig ACTUALLY uses.
    # Without it, the corrector would be compared only against a raw triangulation
    # that is worse than production, flattering the result.
    sources = {'deployed': 'triang', 'uncorrected': None}
    for name, cls in BUILD.items():
        p = f'{OUT}/checkpoints/{name}.pt'
        if os.path.isfile(p):
            m = cls().to(DEV); m.load_state_dict(torch.load(p, map_location=DEV)); m.eval()
            sources[name] = m

    # deployed poses at the same frames, via the processed->acquisition bridge
    import session_cache as sc
    DRD = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
    deployed = np.full_like(XC, np.nan)
    for sk in np.unique(key):
        rat, session = sk.split('/')
        c = sc.get_raw_session(rat, session, want_dual=False)
        if c is None:
            continue
        fmv = pd.read_csv(f'{DRD}/{rat}/{session}/sleap/frame_mapping.csv',
                          skipinitialspace=True)['cam0_frame'].to_numpy(int)
        m = key == sk
        kk = M.k.values[m]
        acq = np.clip(fmv[np.clip(kk, 0, len(fmv) - 1)], 0, len(c['triang_3d']) - 1)
        deployed[m] = c['triang_3d'][acq]

    rows = []
    for src, model in sources.items():
        # ---- produce poses in the CALIBRATION frame for every frame
        if isinstance(model, str) and model == 'triang':
            pose_cal = deployed.copy()
        elif model is None:
            pose_cal = XC.copy()
        else:
            pred = np.empty_like(X)
            off = np.arange(-CTX + 1, 1)
            gid = M.gid.values
            starts = np.r_[0, np.where(gid[1:] != gid[:-1])[0] + 1, len(gid)]
            with torch.no_grad():
                for a, b in zip(starts[:-1], starts[1:]):
                    n = b - a
                    if n < CTX:
                        pred[a:b] = X[a:b]; continue
                    t = np.arange(a + CTX - 1, b)
                    win = X[t[:, None] + off[None, :]]
                    c = win[:, -1].mean(axis=1, keepdims=True)[:, None]
                    wt = torch.from_numpy(win - c).to(DEV)
                    at = torch.from_numpy(A[t]).to(DEV)
                    p = model(wt, at).cpu().numpy() + c[:, 0]
                    pred[a:a + CTX - 1] = X[a:a + CTX - 1]
                    pred[t] = p
            pose_cal = np.empty_like(pred)
            for sk in np.unique(key):
                m = key == sk
                pose_cal[m] = inv_transform(pred[m], TF[sk])
        # DANNCE reference in the calibration frame (same mapping)
        dn_cal = np.empty_like(Y)
        for sk in np.unique(key):
            m = key == sk
            dn_cal[m] = inv_transform(Y[m], TF[sk])

        # ---- per session: project to PCs, then per segment run the rule
        recs = []
        for sk in sorted(set(key)):
            rat, session = sk.split('/')
            tmpl = tp.load_template(rat, session)
            if tmpl is None:
                continue
            m = key == sk
            pcs_s = to_pcs(pose_cal[m], tmpl)
            pcs_d = to_pcs(dn_cal[m], tmpl)
            sub = M[m].reset_index(drop=True)
            T_ex = tmpl['template'][:, :NC]; B_ex = tmpl['bounds_to_use'][:, :NC]
            segs, sec_of, pcs_list, pcsd_list = [], [], [], []
            for g, idx in sub.groupby('gid').groups.items():
                idx = np.asarray(idx)
                if len(idx) < W:
                    continue
                segs.append(g); sec_of.append(int(sub.sector.iloc[idx[0]]))
                pcs_list.append(pcs_s[idx]); pcsd_list.append(pcs_d[idx])
            if len(segs) < 12:
                continue
            # option (b): rebuild a sector-balanced template from a DISJOINT
            # half of the segments, then evaluate on the other half -- otherwise
            # the template is fitted to the very windows it is scored on and
            # everything matches trivially.
            half = np.arange(len(pcs_list)) % 2 == 0
            T_bal, B_bal = build_balanced_template(
                [p for p, h in zip(pcs_list, half) if h],
                [s_ for s_, h in zip(sec_of, half) if h], n_per_sector=12)
            eval_mask = ~half
            # Behaviour covariate must be IDENTICAL across sources and template
            # options, otherwise the conditioning is not comparable. Compute it
            # once from DANNCE against the EXISTING template.
            d_ref = [match_and_score(pdn, T_ex, B_ex)[1] for pdn in pcsd_list]
            for opt, (Tt, Bb) in {'existing': (T_ex, B_ex),
                                  'rebuilt': (T_bal, B_bal)}.items():
                if Tt is None:
                    continue
                for i2, (p, pdn, s_, g, dref) in enumerate(
                        zip(pcs_list, pcsd_list, sec_of, segs, d_ref)):
                    if opt == 'rebuilt' and not eval_mask[i2]:
                        continue
                    hit, mse = match_and_score(p, Tt, Bb)
                    recs.append(dict(source=src, option=opt, rat=rat, session=session,
                                     gid=g, sector=s_, match=hit, mse=mse,
                                     d_dannce=dref,
                                     resid_pc0=float(np.mean(p[:, 0] - pdn[:, 0]))))
        R = pd.DataFrame(recs)
        if R.empty:
            print(f'{src}: no records'); continue
        for opt, g in R.groupby('option'):
            t = g.groupby('sector').match.agg(['size', 'mean'])
            t = t[t['size'] >= 10]
            spread = float(t['mean'].max() - t['mean'].min()) if len(t) > 1 else np.nan
            mu = g.groupby('sector').resid_pc0.mean()
            obs, p = cond_sector_variance(g)
            rows.append(dict(source=src, option=opt, n=len(g),
                             p_match=float(g.match.mean()), sector_spread=spread,
                             mu_range=float(mu.max() - mu.min()),
                             cond_var=obs, cond_p=p))
            print(f'{src:12} [{opt:8}] n={len(g):5d}  P(match)={g.match.mean():.3f}  '
                  f'sector spread={spread:.3f}  mu(theta) range={mu.max()-mu.min():7.2f}  '
                  f'cond var={obs:.5f} (p={p:.3f})', flush=True)
        R.to_csv(f'{OUT}/tier2_records_{src}.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{OUT}/tier2_summary.csv', index=False)
    print(f'\nwrote {OUT}/tier2_summary.csv')


if __name__ == '__main__':
    main()
