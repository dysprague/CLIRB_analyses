"""Stage 17b — P(rewarded) = P(SLEAP says matched | DANNCE unbiasedly detected
this as a rear), by sector, by rat, for four candidate feature spaces:

  full23   - the REAL production template/bounds (all 23 keypoints, current PCs)
  axial    - fresh 1-component PCA on 9 axial/core keypoints only
  snout_z  - egocentric height of Snout above SpineM
  bodyrot  - angle between (SpineF-SpineM) and (SpineM-TailBase)

Everything is computed from the ORIGINAL production SLEAP 2D
(`sleap/sleap_keys_2D.npy`), re-triangulated per session with the same
validated pipeline as stage14a/15/17a (correct Procrustes direction,
processed-frame-counter bridge via `dannce_idx_for_sleap_cams_keys`) -- but
this time keeping each session's FULL, in-order trajectory (not pooled into a
sector sample), so real rear events can be detected and matched.

A handful of example sessions per rat (already validated in §8j/8k, so the
alignment quality is known-good).

DANNCE-based rear detection mirrors stage7_dannce_rears.py exactly: DANNCE PC0
put on the SLEAP scale via a sector-balanced per-session offset, then the
identical hysteresis detector -- so event SELECTION never depends on SLEAP.

For axial/snout_z/bodyrot, a new reference template + bounds is fit from a
sector-balanced sample of SLEAP-side windows aligned to the DANNCE rear ends
(mirroring how real templates are built from SLEAP matches) -- pooled across
the example sessions, per rat. `accepted_errors` is scaled to keep the same
per-feature-slot tolerance as the real rule's 3-out-of-60 (2 PCs x 30 frames):
round(1.5 * n_features_used).

Writes results/spatial_confound/p_rewarded_by_feature.csv
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial import quality as q
from clirb.spatial.geometry import ARENA, sector_index
from clirb import events as ev, templates as tp
from config import NODE_IDX
from corrector.world_alignment import fit_procrustes

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = f'{DR}/calibrations/2026_05_04'
OUT = 'results/spatial_confound'
SECTOR_W, NSEC, W = 30, 12, ev.TEMPLATE_LEN
BUF = ev.ALIGN_BUF_FR
AXIAL = ['Snout', 'EarL', 'EarR', 'SpineF', 'SpineM', 'SpineL', 'TailBase', 'HipL', 'HipR']
AXIAL_IDX = [NODE_IDX[n] for n in AXIAL]
RNG = np.random.default_rng(0)

EXAMPLE_SESSIONS = {
    'R4': ['2026_06_05_1', '2026_06_18_2', '2026_06_23_2'],
    'R5': ['2026_06_05_1', '2026_06_18_2', '2026_06_23_1'],
    'R6': ['2026_06_05_1', '2026_06_22_2', '2026_06_25_2'],
}


def robust_proc(A, B, iters=5, keep=0.7):
    idx = np.arange(len(A)); f = None
    for _ in range(iters):
        f = fit_procrustes(A[idx], B[idx], try_z_flip=True)
        r = np.linalg.norm(f['apply'](A) - B, axis=-1).mean(axis=1)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 200:
            break
    return f


def body_rotation_angle(pose):
    SF = pose[:, NODE_IDX['SpineF'], :]; SM = pose[:, NODE_IDX['SpineM'], :]
    TB = pose[:, NODE_IDX['TailBase'], :]
    v1 = SF - SM; v2 = SM - TB
    n1 = np.linalg.norm(v1, axis=-1); n2 = np.linalg.norm(v2, axis=-1)
    cos = (v1 * v2).sum(-1) / np.maximum(n1 * n2, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def build_session(rat, session, P, K, D, R, T):
    ap = f'{DR}/{rat}/{session}/processed/aligned_data.npz'
    dnp = f'{DR}/{rat}/{session}/processed/sleap_dannce_keys.npz'
    orig2d_path = f'{DR}/{rat}/{session}/sleap/sleap_keys_2D.npy'
    with np.load(ap, allow_pickle=True) as z:
        dk = z['dannce_idx_for_sleap_cams_keys']
    with np.load(dnp, allow_pickle=True) as z:
        dn_raw = z['dannce_keys_3D']
    dn = (dn_raw.squeeze(axis=1).transpose(0, 2, 1) if dn_raw.ndim == 4
          else np.transpose(dn_raw, [0, 2, 1]))
    n_avail = min(len(dk), np.load(orig2d_path, mmap_mode='r').shape[0])
    k2 = np.asarray(np.load(orig2d_path, mmap_mode='r')[:n_avail])
    tgt = dn[np.clip(dk[:n_avail], 0, len(dn) - 1)].astype(np.float64)
    und = q.undistort_all(np.nan_to_num(k2[..., :2]).astype(np.float64), K, D)
    X = q.triangulate_batch(und, P)
    good = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(tgt).all(axis=(1, 2))
    rad = np.linalg.norm(X - X.mean(axis=1, keepdims=True), axis=-1).max(axis=1)
    good &= rad < 250
    if good.sum() < 400:
        return None
    idx_map = np.where(good)[0]                 # good-row -> original-row
    X, tgt = X[good], tgt[good]
    com = X[:, :, :2].mean(axis=1)
    th = np.degrees(np.arctan2(com[:, 1], com[:, 0]))
    sec = sector_index(th, SECTOR_W)
    samp = []
    for s in range(NSEC):
        ii = np.where(sec == s)[0]
        if len(ii) > 20:
            samp.extend(RNG.choice(ii, min(len(ii), 400), replace=False))
    samp = np.array(samp) if len(samp) > 300 else np.arange(len(X))
    fit = robust_proc(tgt[samp], X[samp])
    dn_cal = fit['apply'](tgt)
    resid = float(np.median(np.linalg.norm(dn_cal - X, axis=-1)))
    if resid > 60:
        print(f'  {rat}/{session}: resid {resid:.1f} mm too high, skip'); return None
    print(f'  {rat}/{session}: {len(X)} good frames, resid {resid:.1f} mm', flush=True)
    return dict(X=X, dn_cal=dn_cal, sector=sector_index(ARENA.theta(X[:, :, :2].mean(axis=1)), SECTOR_W))


def align_best(traj_1d, start, end, tmpl_1d, bounds_1d, buf=BUF, accepted_errors=2):
    """traj_1d: (T, nf). tmpl_1d/bounds_1d: (W, nf). Best-aligned window in
    [start, end+buf]; returns (matched: bool, end_frame)."""
    e_min = max(start, W - 1); e_max = min(end + buf, len(traj_1d) - 1)
    if e_max < e_min:
        return False, -1
    ends = np.arange(e_min, e_max + 1)
    idx = (ends - (W - 1))[:, None] + np.arange(W)[None, :]
    win = traj_1d[idx]                                            # (n, W, nf)
    mse = ((win - tmpl_1d[None]) ** 2).mean(axis=(1, 2))
    k = int(np.argmin(mse))
    e = int(ends[k])
    outside = (win[k] >= (tmpl_1d + bounds_1d)) | (win[k] <= (tmpl_1d - bounds_1d))
    return bool(outside.sum() <= accepted_errors), e


def build_ref_1d(feat_by_session, ends_by_session, sector_by_session, n_per_sector=60):
    wins, secs = [], []
    for feat, ends, sec in zip(feat_by_session, ends_by_session, sector_by_session):
        ends = np.asarray(ends, dtype=np.int64)
        if len(ends) == 0:
            continue
        ok = (ends >= W - 1) & (ends < len(feat))
        e = ends[ok]; s_ = np.asarray(sec)[e]      # sector AT each rear's end frame
        idx = (e - (W - 1))[:, None] + np.arange(W)[None, :]
        wins.append(feat[idx]); secs.append(s_)
    X = np.concatenate(wins); S = np.concatenate(secs)
    keep = []
    for s in np.unique(S):
        ii = np.where(S == s)[0]
        keep.extend(RNG.choice(ii, min(len(ii), n_per_sector), replace=False))
    keep = np.array(keep)
    Xs = X[keep]
    tmpl = Xs.mean(axis=0)
    for _ in range(1):
        mse = ((Xs - tmpl[None]) ** 2).mean(axis=(1, 2))
        good = Xs[mse <= np.percentile(mse, 80)]
        if len(good) >= 10:
            tmpl = good.mean(axis=0)
    lo, hi = Xs.min(axis=0), Xs.max(axis=0)
    bounds = np.maximum(tmpl - lo, hi - tmpl)
    return tmpl, bounds


def main():
    P, K, D, R, T = q.load_calib(CAL)
    rows = []

    for rat in ['R4', 'R5', 'R6']:
        tm = None
        for s_ in sorted({os.path.basename(f)[:-4].split('__')[1]
                          for f in glob.glob(f'results/corrector_v2/keys2d/{rat}__*.npz')}):
            tm = tp.load_template(rat, s_)
            if tm is not None:
                break
        if tm is None:
            print(f'{rat}: no template'); continue
        pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)

        sessions_data = []
        for session in EXAMPLE_SESSIONS[rat]:
            sd = build_session(rat, session, P, K, D, R, T)
            if sd is None:
                continue
            sd['session'] = session
            sessions_data.append(sd)
        if not sessions_data:
            continue

        # ---- per-session: SLEAP/DANNCE rotated_med + PCs (full template basis)
        for sd in sessions_data:
            s_pcs, s_rot = ev.project_session_to_pcs(sd['X'], tm, flip_z=True)
            d_pcs, d_rot = ev.project_session_to_pcs(sd['dn_cal'], tm, flip_z=False)
            sd['s_pcs'], sd['s_rot'], sd['d_pcs'], sd['d_rot'] = s_pcs, s_rot, d_pcs, d_rot

        # ---- fresh axial PCA, fit on pooled sector-balanced DANNCE sample
        all_d_rot = np.concatenate([sd['d_rot'] for sd in sessions_data])
        all_sector = np.concatenate([sd['sector'] for sd in sessions_data])
        fit_idx = []
        for s in range(NSEC):
            ii = np.where(all_sector == s)[0]
            if len(ii):
                fit_idx.extend(RNG.choice(ii, min(len(ii), 4000), replace=False))
        fit_idx = np.array(fit_idx)
        sub = all_d_rot[fit_idx][:, AXIAL_IDX, :].reshape(len(fit_idx), -1)
        mu_ax = sub.mean(axis=0)
        pca = PCA(n_components=1).fit(sub - mu_ax)
        comp = pca.components_[0]
        all_d_pcs0 = np.concatenate([sd['d_pcs'][:, pcu[0]] for sd in sessions_data])
        if np.corrcoef((sub - mu_ax) @ comp, all_d_pcs0[fit_idx])[0, 1] < 0:
            comp = -comp

        for sd in sessions_data:
            sd['s_axial'] = ((sd['s_rot'][:, AXIAL_IDX, :].reshape(len(sd['s_rot']), -1) - mu_ax) @ comp)[:, None]
            sd['d_axial'] = ((sd['d_rot'][:, AXIAL_IDX, :].reshape(len(sd['d_rot']), -1) - mu_ax) @ comp)[:, None]
            sd['s_snout'] = sd['s_rot'][:, NODE_IDX['Snout'], 2:3]
            sd['d_snout'] = sd['d_rot'][:, NODE_IDX['Snout'], 2:3]
            sd['s_bodyrot'] = body_rotation_angle(sd['s_rot'])[:, None]
            sd['d_bodyrot'] = body_rotation_angle(sd['d_rot'])[:, None]

        # ---- DANNCE-defined rears, put on the RAW-SLEAP PC0 scale.
        # A pure additive offset (stage7's trick) assumes SLEAP and DANNCE PC0
        # already share the same amplitude -- true for corrector-adjusted SLEAP
        # (trained toward DANNCE) but NOT for raw/uncorrected SLEAP, which is
        # noisier and has a substantially WIDER PC0 range. Fit a sector-balanced
        # linear (scale + offset) calibration instead of a mean-only offset.
        d_all, s_all, sec_all = [], [], []
        for sd in sessions_data:
            d_all.append(sd['d_pcs'][:, pcu[0]]); s_all.append(sd['s_pcs'][:, pcu[0]])
            sec_all.append(sd['sector'])
        d_all = np.concatenate(d_all); s_all = np.concatenate(s_all); sec_all = np.concatenate(sec_all)
        cal_idx = []
        for k in range(NSEC):
            ii = np.where(sec_all == k)[0]
            if len(ii) >= 200:
                cal_idx.extend(RNG.choice(ii, min(len(ii), 3000), replace=False))
        cal_idx = np.array(cal_idx) if len(cal_idx) > 500 else np.arange(len(d_all))
        a_cal, b_cal = np.polyfit(d_all[cal_idx], s_all[cal_idx], 1)
        r_cal = np.corrcoef(d_all[cal_idx], s_all[cal_idx])[0, 1]
        print(f'  {rat}: DANNCE->raw-SLEAP PC0 calibration  scale={a_cal:.3f}  offset={b_cal:.1f}  '
              f'r={r_cal:+.3f}  (n_cal={len(cal_idx)})')
        # independent sanity check: does DANNCE PC0 track an independent height
        # proxy (max keypoint height above SpineM) in the SAME direction as the
        # fitted scale sign implies? If not, don't trust the calibration.
        height_all = np.concatenate([sd['dn_cal'][:, :, 2].max(axis=1) - sd['dn_cal'][:, NODE_IDX['SpineM'], 2]
                                     for sd in sessions_data])
        r_height = np.corrcoef(d_all[cal_idx], height_all[cal_idx])[0, 1]
        print(f'  {rat}: sanity -- corr(DANNCE PC0, independent height proxy) = {r_height:+.3f} '
              f'({"consistent" if np.sign(r_height) == np.sign(a_cal) else "*** SIGN MISMATCH, DO NOT TRUST ***"})')

        events = []          # (session_idx, start, end)
        for si, sd in enumerate(sessions_data):
            d0 = a_cal * sd['d_pcs'][:, pcu[0]] + b_cal
            rears_d = [r for r in ev.detect_rears(d0) if r[1] < len(sd['X'])]
            sd['rears'] = rears_d
            for r in rears_d:
                events.append((si, r[0], r[1]))
        print(f'{rat}: {len(events)} DANNCE-detected rear events across {len(sessions_data)} sessions')

        # ---- REAL-bounds reference point (not part of the fair comparison):
        # the actual production template/bounds, fit historically on
        # corrector-adjusted SLEAP, applied to RAW/uncorrected SLEAP here.
        # Tells us how much of the online rule's real tolerance band is
        # "used up" by the corrector rather than by genuine behavior tolerance.
        base_tmpl = tm['template'][:, :len(pcu)]
        base_bounds = ev.template_bounds(tm)

        # ---- FAIR comparison: fit a fresh template for ALL FOUR candidates
        # -- including the full 2-PC space -- from the SAME procedure, same
        # training events, same accepted_errors scaling. Isolates the effect
        # of keypoint/feature selection alone, holding fitting method fixed.
        sd_for_key = {'full23_fresh': 's_pcs', 'axial': 's_axial',
                     'snout_z': 's_snout', 'bodyrot': 's_bodyrot'}
        new_templates = {}
        for name, skey in sd_for_key.items():
            ends_list = [np.array([r[1] for r in sd['rears']]) for sd in sessions_data]
            feat_list = [(sd[skey][:, pcu] if skey == 's_pcs' else sd[skey]) for sd in sessions_data]
            sec_list = [sd['sector'] for sd in sessions_data]
            tmpl, bounds = build_ref_1d(feat_list, ends_list, sec_list)
            new_templates[name] = (tmpl, bounds)
            print(f'  {rat} {name}: template built, n_features={tmpl.shape[1]}')

        # ---- score every event under every candidate
        for si, start, end in events:
            sd = sessions_data[si]
            sec = int(sd['sector'][min(end, len(sd['sector']) - 1)])
            theta = float(ARENA.theta(sd['X'][min(end, len(sd['X']) - 1), :, :2].mean(axis=0)))

            m_base, _ = align_best(sd['s_pcs'][:, pcu], start, end, base_tmpl, base_bounds,
                                   accepted_errors=ev.ACCEPTED_ERRORS)
            row = dict(rat=rat, sector=sec, theta=theta, full23_realbounds=m_base)
            for name, skey in sd_for_key.items():
                tmpl, bounds = new_templates[name]
                nfeat = tmpl.shape[1]
                acc = max(1, round(1.5 * nfeat))
                traj = sd[skey][:, pcu] if skey == 's_pcs' else sd[skey]
                m, _ = align_best(traj, start, end, tmpl, bounds, accepted_errors=acc)
                row[name] = m
            rows.append(row)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/p_rewarded_by_feature.csv', index=False)
    print('\n' + '=' * 78)
    print('P(rewarded | DANNCE-detected rear) BY SECTOR, BY RAT, BY FEATURE')
    print('=' * 78)
    feats = ['full23_realbounds', 'full23_fresh', 'axial', 'snout_z', 'bodyrot']
    for rat, g in d.groupby('rat'):
        print(f'\n{rat}  (n={len(g)} DANNCE-detected rears)')
        piv = g.groupby('sector')[feats].agg(['size', 'mean'])
        print(piv.round(3).to_string())
    print('\nOverall pooled P(rewarded) and sector spread (max-min):')
    print('(full23_realbounds = historical production bounds, NOT a fair comparison --')
    print(' the other four are all freshly fit on this same raw-SLEAP data, fair comparison)')
    for rat, g in d.groupby('rat'):
        for feat in feats:
            by_sec = g.groupby('sector')[feat].mean()
            print(f'  {rat} {feat:18s}: P={g[feat].mean():.3f}  spread={by_sec.max()-by_sec.min():.3f}  n_sectors={len(by_sec)}')
    print(f'\nwrote {OUT}/p_rewarded_by_feature.csv')


if __name__ == '__main__':
    main()
