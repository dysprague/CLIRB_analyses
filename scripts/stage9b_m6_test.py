"""Stage 9b — separate SLEAP network bias (M6) from a calibration account.

Now unblocked: `triang_keys_3D` IS in the camera-calibration frame (it reprojects
at ~17 px once the processed/acquisition frame bases are bridged), so DANNCE can be
carried into the camera frame by a single Procrustes fit to it.

residual = (SLEAP 2D detection) - (projection of the DANNCE pose)

  * CALIBRATION / distortion / residual global misalignment
        -> smooth function of IMAGE POSITION, COMMON to all keypoints
  * SLEAP NETWORK BIAS (M6)
        -> KEYPOINT-SPECIFIC, and varying with the animal's APPARENT ASPECT
           (heading relative to that camera's bearing)

The SLEAP self-residual (detections vs the projection of triang_3d) is reported as
a baseline: triang_3d was fit to those detections, so it absorbs network bias and
should show mainly the calibration component.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du, quality as q
from clirb.spatial.geometry import ARENA, sector_index
from config import NODE_IDX, NODES
from projection import project_3d_to_2d
from corrector.world_alignment import fit_procrustes

OUT = 'results/spatial_confound'
SECTOR_W, NSEC = 30, 12
CONF, UV_BINS, ASP_BINS, STRIDE = 0.5, 6, 8, 2
RNG = np.random.default_rng(0)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def collect(rat, max_sessions=16):
    recs = []
    files = sorted(glob.glob(f'data/cache/{rat}/*.npz'))
    step = max(1, len(files) // max_sessions)
    for f in files[::step]:
        session = os.path.basename(f)[:-4]
        raw = sc.get_raw_session(rat, session, want_dual=True)
        if raw is None or not raw.get('has_dual') or not raw.get('has_keys2d'):
            continue
        t = tp.load_template(rat, session)
        d3 = du.dannce_aligned(raw)
        cal = q.calib_dir(rat, session)
        fm = np.asarray(raw['frame_mapping_cam0'], dtype=int)
        if t is None or d3 is None or cal is None or not len(fm):
            continue
        s3 = raw['triang_3d'].astype(np.float64)
        k2 = raw['sleap_keys_2D']
        n = min(len(s3), len(d3))
        s3, d3 = s3[:n], d3[:n].astype(np.float64)

        com = s3.mean(axis=1)
        th = ARENA.theta(com[:, :2]); rad = ARENA.radius_of(com[:, :2])
        sec = sector_index(th, SECTOR_W)
        pcu = np.asarray(t['pcs_to_use']).ravel().astype(int)
        s_pcs, _ = ev.project_session_to_pcs(s3, t, flip_z=True)
        rears = [r for r in ev.detect_rears(s_pcs[:, pcu[0]]) if r[1] < n]
        if not rears:
            continue

        # DANNCE -> triang_3d frame (== camera frame), sector-balanced fit
        samp = []
        for k in range(NSEC):
            ii = np.where((sec == k) & (rad > 250))[0]
            if len(ii) > 50:
                samp.extend(RNG.choice(ii, min(len(ii), 300), replace=False))
        if len(samp) < 2000:
            continue
        fit = fit_procrustes(d3[np.array(samp)], s3[np.array(samp)])
        dn = fit['apply'](d3)                       # DANNCE in the camera frame

        rear_mask = np.zeros(n, bool)
        for a, b, _ in rears:
            rear_mask[a:b + 1] = True
        rear_mask &= rad > 250

        # processed frames k whose cam0 frame f is a rear-at-wall frame
        K = min(len(fm), len(k2))
        f_of_k = fm[:K]
        sel = np.where((f_of_k < n) & rear_mask[np.clip(f_of_k, 0, n - 1)])[0][::STRIDE]
        if len(sel) < 40:
            continue
        fsel = f_of_k[sel]

        P, Kl, Dl, Rl, Tl = q.load_calib(cal)
        SF = s3[:, NODE_IDX['SpineF'], :2]; SM = s3[:, NODE_IDX['SpineM'], :2]
        head = np.degrees(np.arctan2(SF[:, 1] - SM[:, 1], SF[:, 0] - SM[:, 0]))

        for c in range(3):
            Cc = -Rl[c].T @ Tl[c]
            bear = np.degrees(np.arctan2(com[fsel, 1] - Cc[1], com[fsel, 0] - Cc[0]))
            aspect = wrap(head[fsel] - bear)
            pj_d = project_3d_to_2d(dn[fsel].reshape(-1, 3), Kl[c], Rl[c], Tl[c],
                                    Dl[c]).reshape(len(fsel), 23, 2)
            pj_s = project_3d_to_2d(s3[fsel].reshape(-1, 3), Kl[c], Rl[c], Tl[c],
                                    Dl[c]).reshape(len(fsel), 23, 2)
            det = k2[sel, c, :, :2].astype(np.float64)
            cf = k2[sel, c, :, 2]
            rd = det - pj_d
            rs = det - pj_s
            good = (cf >= CONF) & np.isfinite(rd).all(axis=-1) & \
                   (np.linalg.norm(rd, axis=-1) < 120)
            gi, gj = np.where(good)
            if not len(gi):
                continue
            recs.append(pd.DataFrame({
                'cam': c, 'kp': gj, 'u': det[gi, gj, 0], 'v': det[gi, gj, 1],
                'du': rd[gi, gj, 0], 'dv': rd[gi, gj, 1],
                'su': rs[gi, gj, 0], 'sv': rs[gi, gj, 1],
                'aspect': aspect[gi], 'sector': sec[fsel][gi]}))
    return pd.concat(recs, ignore_index=True) if recs else None


def decompose(d, ucol, vcol, tag):
    d = d.copy()
    d['ub'] = d.groupby('cam')[ucol.replace(ucol, 'u')].transform(
        lambda s: pd.qcut(s, UV_BINS, labels=False, duplicates='drop'))
    d['vb'] = d.groupby('cam')['v'].transform(
        lambda s: pd.qcut(s, UV_BINS, labels=False, duplicates='drop'))
    d['ab'] = pd.cut(d.aspect, np.linspace(-180, 180, ASP_BINS + 1), labels=False)
    com = d.groupby(['cam', 'ub', 'vb'])[[ucol, vcol]].transform('mean')
    d['cu'] = d[ucol] - com[ucol]; d['cv'] = d[vcol] - com[vcol]
    fld = d.groupby(['cam', 'ub', 'vb'])[[ucol, vcol]].mean()
    sd_common = float(np.hypot(fld[ucol].std(), fld[vcol].std()))
    kp = d.groupby(['cam', 'kp'])[['cu', 'cv']].mean()
    sd_kp = float(np.hypot(kp.cu.std(), kp.cv.std()))
    kpm = d.groupby(['cam', 'kp'])[['cu', 'cv']].transform('mean')
    d['ku'] = d.cu - kpm.cu; d['kv'] = d.cv - kpm.cv
    asp = d.groupby(['cam', 'kp', 'ab'])[['ku', 'kv']].agg(['mean', 'size'])
    asp = asp[asp[('ku', 'size')] >= 40]
    sd_asp = float(np.hypot(asp[('ku', 'mean')].std(), asp[('kv', 'mean')].std())) \
        if len(asp) else np.nan
    med = float(np.median(np.hypot(d[ucol], d[vcol])))
    print(f'  {tag}')
    print(f'     median |residual|                : {med:6.2f} px')
    print(f'     [A] common field vs image position: {sd_common:6.2f} px')
    print(f'     [B] keypoint-specific (after A)   : {sd_kp:6.2f} px')
    print(f'     [C] aspect-dependent, per keypoint: {sd_asp:6.2f} px')
    print(f'     B/A = {sd_kp/max(sd_common,1e-9):.2f}   C/A = {sd_asp/max(sd_common,1e-9):.2f}')
    return dict(median=med, A=sd_common, B=sd_kp, C=sd_asp), asp


if __name__ == '__main__':
    res = {}
    for rat in ['R4', 'R5', 'R6']:
        print(f'collecting {rat} ...', flush=True)
        d = collect(rat)
        if d is None or len(d) < 5000:
            print(f'  {rat}: insufficient data'); continue
        print(f'\n{"="*76}\n{rat}: {len(d):,} camera-keypoint observations\n{"="*76}')
        rd, asp = decompose(d, 'du', 'dv', 'residual vs DANNCE projection (independent)')
        rs, _ = decompose(d, 'su', 'sv', 'residual vs SLEAP self-projection (baseline)')
        res[rat] = dict(vs_dannce=rd, vs_sleap=rs)
        if len(asp):
            mag = np.hypot(asp[('ku', 'mean')], asp[('kv', 'mean')])
            top = mag.groupby(level='kp').max().sort_values(ascending=False).head(6)
            print('     most aspect-sensitive keypoints:')
            for k, v in top.items():
                print(f'        {NODES[int(k)]:<10} {v:.2f} px')
    with open(f'{OUT}/stage9b_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f'\nwrote {OUT}/stage9b_results.json')
