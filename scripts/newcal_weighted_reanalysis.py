"""Re-run the core analyses on CONFIDENCE-WEIGHTED triangulation, paired
within session against the rig's own equal-weight triangulation.

The rig's `triang_keys_3D` is a plain 3-camera DLT that weights every view
equally.  Scaling each camera's two DLT rows by its SLEAP confidence costs
nothing and uses only data already on disk, and improved rigidity and jitter by
~20% in the pilot.  This measures what it does to everything downstream.

Paired by construction: both variants are computed from the same
`sleap_keys_2D`, over the same full acquisition grid, against the same DANNCE,
with their own Procrustes fits.

The weighted variant is re-triangulated on the PROCESSED frames and the gaps
are then filled by the same linear interpolation the rig applies, so both
series are contiguous and full length.  That matters: `ear_mean_z`,
`cum_heading_change`, the PC projection and the rear detector are all temporal,
and scoring them on a non-contiguous subsequence silently applies the 5-frame
causal median and the 30-frame rolling sum across frame gaps.  A first pass
that scored processed frames only produced September PC1 correlations of
0.2-0.5 against the 0.86 the contiguous series gives -- an artifact of exactly
that.

September sessions are included so the comparison spans all four eras on the
same footing.

Writes results/newcal/weighted_reanalysis.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, features as cf
from clirb.spatial import dual as du
from clirb.spatial import quality as ql
from config import NODE_IDX
from scripts.newcal_sleap_dannce_eval import (robust_proc, balanced_sample,
                                              COMMON_TEMPLATE, GROSS_FRAME_RMSE_MM)
from scripts.newcal_rear_detector import detect_rears_fast, match_overlap, prf

OUT = 'results/newcal'
CONF_T = 0.5
N_CAMS, N_KP = 3, 23
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
DEPLOYED = dict(rise=140.0, reset=100.0, min_frames=10)
TURN_DEG = 45.0
N_PER_ERA = {'sept_baseline': 9, 'oct_a': 4, 'oct_b': 6, 'oct_26': 3}


def dlt_chunked(und, P_list, w, chunk=200_000):
    T = und.shape[0]
    x = und[..., 0].transpose(0, 2, 1).reshape(-1, N_CAMS)
    y = und[..., 1].transpose(0, 2, 1).reshape(-1, N_CAMS)
    ww = w.transpose(0, 2, 1).reshape(-1, N_CAMS)
    P = np.stack(P_list)
    N = x.shape[0]
    out = np.full((N, 3), np.nan)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        A = np.empty((e - s, 2 * N_CAMS, 4))
        for i in range(N_CAMS):
            A[:, 2 * i] = (x[s:e, i, None] * P[i, 2][None, :] - P[i, 0][None, :]) * ww[s:e, i, None]
            A[:, 2 * i + 1] = (y[s:e, i, None] * P[i, 2][None, :] - P[i, 1][None, :]) * ww[s:e, i, None]
        ok = (ww[s:e] > 0).sum(axis=1) >= 2
        if ok.any():
            _, _, Vt = np.linalg.svd(A[ok])
            h = Vt[:, -1, :]
            d = np.where(np.abs(h[:, 3]) < 1e-12, 1.0, h[:, 3])
            blk = np.full((e - s, 3), np.nan)
            blk[ok] = h[:, :3] / d[:, None]
            out[s:e] = blk
    return out.reshape(T, N_KP, 3)


def fill_gaps(kp, known):
    """Linear interpolation over frames, as the rig does for frames it could
    not process in time.  Keeps the series contiguous and full length."""
    out = kp.copy()
    idx = np.where(known)[0]
    if len(idx) < 2:
        return out
    allf = np.arange(len(kp))
    for j in range(kp.shape[1]):
        for a in range(3):
            out[:, j, a] = np.interp(allf, idx, kp[idx, j, a])
    return out


def metrics(sl, dn, common, tag):
    """All downstream numbers for one SLEAP variant, full contiguous series."""
    good = np.isfinite(sl).all(axis=(1, 2)) & np.isfinite(dn).all(axis=(1, 2))
    if good.sum() < 1000:
        return None
    s, d = sl, dn
    gi = np.where(good)[0]
    samp0, _ = balanced_sample(np.median(s[gi], axis=1)[:, :2],
                               rng=np.random.default_rng(0))
    if len(samp0) < 500:
        return None
    samp = gi[samp0]
    fit = robust_proc(d[samp], s[samp])
    dw = fit['apply'](d)
    dist = np.linalg.norm(s - dw, axis=-1)
    dist[~good] = np.nan
    frm = np.sqrt(np.nanmean(dist ** 2, axis=1))
    r = {f'{tag}_n': int(good.sum()),
         f'{tag}_fit_residual': float(fit['residual']),
         f'{tag}_kp_med': float(np.nanmedian(dist)),
         f'{tag}_frame_rmse_p50': float(np.nanpercentile(frm, 50)),
         f'{tag}_frame_rmse_p90': float(np.nanpercentile(frm, 90)),
         f'{tag}_gross_frac': float(np.nanmean(frm > GROSS_FRAME_RMSE_MM)),
         f'{tag}_flyaway_frac': float(np.nanmean((dist > 100).any(axis=1))),
         f'{tag}_bone_cv': float(np.nanmedian(ql.bone_length_cv(s))),
         f'{tag}_jitter': float(np.nanmedian(
             np.linalg.norm(np.diff(s, n=2, axis=0), axis=-1).mean(axis=1)))}
    # Egocentric and PC space take the RAW DANNCE, not the Procrustes-aligned
    # copy: project_session_to_pcs applies its own z-flip convention (True for
    # SLEAP, False for DANNCE), and the Procrustes fit already contains a
    # z-flip, so feeding it `dw` applies the flip twice.  That inverted PC0
    # (correlation -0.96 instead of +0.99) and inflated the egocentric RMSE
    # six-fold in the first pass.  Egocentric normalisation removes translation
    # and yaw anyway, so the alignment is not needed here.
    _, srot = ev.project_session_to_pcs(s, common, flip_z=True)
    _, drot = ev.project_session_to_pcs(d, common, flip_z=False)
    r[f'{tag}_ego_rmse'] = float(np.sqrt(((srot - drot) ** 2).sum(-1).mean()))
    sp, _ = ev.project_session_to_pcs(s, common, flip_z=True)
    dp, _ = ev.project_session_to_pcs(d, common, flip_z=False)
    pcs = [int(v) for v in np.asarray(common['pcs_to_use']).ravel()[:2]]
    for j, p in enumerate(pcs):
        a, b = sp[:, p], dp[:, p]
        r[f'{tag}_pc{j}_rmse'] = float(np.sqrt(((a - b) ** 2).mean()))
        r[f'{tag}_pc{j}_r'] = float(np.corrcoef(a, b)[0, 1])
    # features
    S = ev.causal_median_5_kp(s, warmup='online').copy(); S[:, :, 2] *= -1
    D = ev.causal_median_5_kp(dw, warmup='online').copy(); D[:, :, 2] *= -1
    fs, fd = cf.compute_features(S), cf.compute_features(D)
    e = fs['ear_mean_z'] - fd['ear_mean_z']
    r[f'{tag}_ear_rmse'] = float(np.sqrt((e ** 2).mean()))
    r[f'{tag}_ear_med'] = float(np.median(np.abs(e)))
    rs = detect_rears_fast(fs['ear_mean_z'], **DEPLOYED)
    rd = detect_rears_fast(fd['ear_mean_z'], **DEPLOYED)
    _, tp = match_overlap(rs, rd)
    p_, rc_, f_ = prf(tp, len(rs), len(rd))
    r.update({f'{tag}_rear_n': len(rd), f'{tag}_rear_p': p_,
              f'{tag}_rear_r': rc_, f'{tag}_rear_f1': f_})
    a, b = fs['cum_heading_change'], fd['cum_heading_change']
    r[f'{tag}_turn_rmse'] = float(np.sqrt(((a - b) ** 2).mean()))
    m = np.abs(a) >= TURN_DEG
    r[f'{tag}_turn_sign'] = (float(100 * (np.sign(a) == np.sign(b))[m].mean())
                             if m.sum() > 100 else np.nan)
    kp = np.sqrt((dist ** 2).mean(axis=0))
    for k, name in enumerate(KP_NAMES):
        r[f'{tag}_kpmed_{name}'] = float(np.nanmedian(dist[:, k]))
    return r


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    use = pd.concat([inv[inv.era == e].sort_values('session').tail(n)
                     for e, n in N_PER_ERA.items()])
    common = dict(np.load(COMMON_TEMPLATE, allow_pickle=True))
    rows = []
    for i, r in enumerate(use.itertuples(), 1):
        raw = sc.get_raw_session(r.rat, r.session, want_dual=True)
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        sl_rig = np.asarray(raw['triang_3d'], np.float32)
        n = min(len(sl_rig), len(d3))
        k2 = np.asarray(raw['sleap_keys_2D'])
        fm = np.asarray(raw['frame_mapping_cam0'], int)
        nproc = min(len(fm), len(k2))
        P, K, D, T_, _t = ql.load_calib(ql.calib_dir(r.rat, r.session))[0:5]
        und = ql.undistort_all(k2[:nproc, :, :, :2].astype(np.float64), K, D)
        conf = k2[:nproc, :, :, 2].astype(np.float64)
        wt = dlt_chunked(und, P, np.clip(conf, 0, None))
        # scatter onto the acquisition grid; score only processed frames
        sl_wt = np.full((n, N_KP, 3), np.nan, np.float32)
        fmp = fm[:nproc]
        ok = ((fmp >= 0) & (fmp < n)
              & np.isfinite(wt).all(axis=(1, 2))[:len(fmp)])
        sl_wt[fmp[ok]] = wt[ok]
        known = np.zeros(n, bool); known[fmp[ok]] = True
        sl_wt = fill_gaps(sl_wt, known)      # contiguous, as the rig's own is
        dn = np.asarray(d3[:n], np.float32)
        sl_eq = sl_rig[:n].copy()

        rec = dict(rat=r.rat, session=r.session, era=r.era,
                   n_known=int(known.sum()), n_acq=int(n),
                   frac_retriangulated=float(known.mean()))
        for tag, arr in [('eq', sl_eq), ('wt', sl_wt)]:
            m = metrics(arr, dn, common, tag)
            if m is None:
                rec = None
                break
            rec.update(m)
        if rec is None:
            print(f'[{i}/{len(use)}] {r.rat}/{r.session}: unusable', flush=True)
            continue
        rows.append(rec)
        print(f'[{i}/{len(use)}] {r.rat}/{r.session} ({r.era}) '
              f'kp_med {rec["eq_kp_med"]:.2f} -> {rec["wt_kp_med"]:.2f} mm | '
              f'bone_cv {rec["eq_bone_cv"]:.3f} -> {rec["wt_bone_cv"]:.3f} | '
              f'PC1 r {rec["eq_pc1_r"]:.3f} -> {rec["wt_pc1_r"]:.3f}', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/weighted_reanalysis.csv', index=False)
    pd.set_option('display.width', 250)
    base = ['kp_med', 'frame_rmse_p50', 'gross_frac', 'flyaway_frac', 'bone_cv',
            'jitter', 'ego_rmse', 'pc0_r', 'pc1_r', 'pc1_rmse',
            'ear_rmse', 'rear_f1', 'turn_rmse', 'turn_sign', 'fit_residual']
    g = d.groupby('era')[[f'{t}_{c}' for c in base for t in ('eq', 'wt')]].median()
    print('\n=== equal-weight (the rig) vs confidence-weighted, era medians ===')
    for c in base:
        line = f'  {c:16s}'
        for e in ['sept_baseline', 'oct_a', 'oct_b', 'oct_26']:
            if e not in g.index:
                continue
            a, b = g.loc[e, f'eq_{c}'], g.loc[e, f'wt_{c}']
            line += f' | {e[:7]:7s} {a:8.3f}->{b:8.3f}'
        print(line)
    print(f'\nwrote {OUT}/weighted_reanalysis.csv')


if __name__ == '__main__':
    main()
