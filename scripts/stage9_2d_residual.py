"""Stage 9 — separate a SLEAP-network-bias account (M6) from a calibration account.

Idea: reproject the INDEPENDENT DANNCE 3D pose (mapped into the SLEAP world frame
by the inverse per-session Procrustes) into each SLEAP camera, and compare with the
SLEAP 2D detections. Unlike the ordinary reprojection residual, this residual is
NOT blind to along-ray error, because the 3D reference is independent.

The discriminator:
  * CALIBRATION / lens distortion / residual global misalignment
        -> the 2D residual is a smooth function of IMAGE POSITION and is
           COMMON TO ALL KEYPOINTS (it acts on image coordinates, not body parts)
  * SLEAP NETWORK BIAS (M6)
        -> the residual is KEYPOINT-SPECIFIC and varies with the animal's
           APPARENT ASPECT (heading relative to that camera's bearing)

So: fit the keypoint-pooled field vs image position, then ask how much
keypoint-specific and aspect-dependent structure is left over.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du
from clirb.spatial import quality as q
from clirb.spatial.geometry import ARENA, sector_index
from config import NODE_IDX, NODES
from projection import project_3d_to_2d
from corrector.world_alignment import fit_procrustes

OUT = 'results/spatial_confound'
SECTOR_W, NSEC = 30, 12
CONF = 0.5
UV_BINS = 6          # image-position grid per axis
ASP_BINS = 8         # apparent-aspect bins
STRIDE = 3
RNG = np.random.default_rng(0)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def collect(rat, max_sessions=18):
    recs = []
    files = sorted(glob.glob(f'data/cache/{rat}/*.npz'))
    step = max(1, len(files) // max_sessions)
    for f in files[::step]:
        session = os.path.basename(f)[:-4]
        raw = sc.get_raw_session(rat, session, want_dual=True)
        if raw is None or not raw.get('has_dual', False) or not raw.get('has_keys2d', False):
            continue
        t = tp.load_template(rat, session)
        d3 = du.dannce_aligned(raw)
        cal = q.calib_dir(rat, session)
        if t is None or d3 is None or cal is None:
            continue
        s3 = raw['triang_3d'].astype(np.float64)
        k2 = raw['sleap_keys_2D']
        n = min(len(s3), len(d3), len(k2))
        s3, d3, k2 = s3[:n], d3[:n].astype(np.float64), k2[:n]

        pcu = np.asarray(t['pcs_to_use']).ravel().astype(int)
        s_pcs, _ = ev.project_session_to_pcs(s3, t, flip_z=True)
        rears = [r for r in ev.detect_rears(s_pcs[:, pcu[0]]) if r[1] < n]
        if not rears:
            continue
        com = s3.mean(axis=1)
        th = ARENA.theta(com[:, :2]); rad = ARENA.radius_of(com[:, :2])
        sec = sector_index(th, SECTOR_W)

        # CRITICAL: triang_keys_3D is NOT in the camera-calibration frame (it is
        # a transformed arena frame). Re-triangulate from the 2D detections so the
        # 3D and the cameras agree -- this is what quality.reprojection_residual
        # does, and why it gets ~9 px instead of ~490 px.
        _, cal3d, _ = q.reprojection_residual(k2, cal)
        Pl, Kl, Dl, Rl, Tl = q.load_calib(cal)

        # sector-balanced Procrustes so the alignment carries no spatial bias
        samp = []
        for k in range(NSEC):
            ii = np.where((sec == k) & (rad > 250) &
                          np.isfinite(cal3d).all(axis=(1, 2)))[0]
            if len(ii) > 50:
                samp.extend(RNG.choice(ii, min(len(ii), 300), replace=False))
        if len(samp) < 2000:
            continue
        fit = fit_procrustes(cal3d[np.array(samp)], d3[np.array(samp)])
        dn_in_cal = fit['apply_inverse'](d3)      # DANNCE -> CALIBRATION frame

        fr = np.concatenate([np.arange(a, b + 1) for a, b, _ in rears])
        fr = fr[(fr < n)]
        fr = fr[(rad[fr] > 250) & np.isfinite(cal3d[fr]).all(axis=(1, 2))][::STRIDE]
        if len(fr) < 40:
            continue

        # heading and camera bearing BOTH in the calibration frame
        ccom = cal3d[:, :, :2].mean(axis=1)
        SF = cal3d[:, NODE_IDX['SpineF'], :2]; SM = cal3d[:, NODE_IDX['SpineM'], :2]
        head = np.degrees(np.arctan2(SF[:, 1] - SM[:, 1], SF[:, 0] - SM[:, 0]))

        for c in range(3):
            Cc = -Rl[c].T @ Tl[c]                       # camera centre, cal frame
            bear = np.degrees(np.arctan2(ccom[fr, 1] - Cc[1], ccom[fr, 0] - Cc[0]))
            aspect = wrap(head[fr] - bear)
            proj = project_3d_to_2d(dn_in_cal[fr].reshape(-1, 3),
                                    Kl[c], Rl[c], Tl[c], Dl[c]).reshape(len(fr), 23, 2)
            det = k2[fr, c, :, :2].astype(np.float64)
            cf = k2[fr, c, :, 2]
            resid = det - proj
            good = (cf >= CONF) & np.isfinite(resid).all(axis=-1) & \
                   (np.linalg.norm(resid, axis=-1) < 100)   # drop gross outliers
            gi, gj = np.where(good)
            if len(gi) == 0:
                continue
            recs.append(pd.DataFrame({
                'cam': c, 'kp': gj,
                'u': det[gi, gj, 0], 'v': det[gi, gj, 1],
                'du': resid[gi, gj, 0], 'dv': resid[gi, gj, 1],
                'aspect': aspect[gi], 'sector': sec[fr][gi]}))
    return pd.concat(recs, ignore_index=True) if recs else None


def analyse(rat, d):
    out = {}
    print(f'\n{"="*74}\n{rat}: {len(d):,} camera-keypoint observations\n{"="*74}')
    # image-position bins per camera
    d = d.copy()
    d['ub'] = d.groupby('cam')['u'].transform(
        lambda s: pd.qcut(s, UV_BINS, labels=False, duplicates='drop'))
    d['vb'] = d.groupby('cam')['v'].transform(
        lambda s: pd.qcut(s, UV_BINS, labels=False, duplicates='drop'))
    d['ab'] = pd.cut(d.aspect, np.linspace(-180, 180, ASP_BINS + 1), labels=False)

    # ---- 1. COMMON field: keypoint-pooled mean residual per (cam, u-bin, v-bin)
    common = d.groupby(['cam', 'ub', 'vb'])[['du', 'dv']].transform('mean')
    d['du_c'] = d.du - common.du
    d['dv_c'] = d.dv - common.dv
    fld = d.groupby(['cam', 'ub', 'vb'])[['du', 'dv']].mean()
    sd_common = float(np.hypot(fld.du.std(), fld.dv.std()))

    # ---- 2. KEYPOINT-SPECIFIC structure left after removing the common field
    kp = d.groupby(['cam', 'kp'])[['du_c', 'dv_c']].mean()
    sd_kp = float(np.hypot(kp.du_c.std(), kp.dv_c.std()))

    # ---- 3. ASPECT dependence of the keypoint-specific residual, after also
    #         removing each keypoint's own mean
    kpm = d.groupby(['cam', 'kp'])[['du_c', 'dv_c']].transform('mean')
    d['du_k'] = d.du_c - kpm.du_c
    d['dv_k'] = d.dv_c - kpm.dv_c
    asp = d.groupby(['cam', 'kp', 'ab'])[['du_k', 'dv_k']].agg(['mean', 'size'])
    asp = asp[asp[('du_k', 'size')] >= 50]
    sd_aspect = float(np.hypot(asp[('du_k', 'mean')].std(), asp[('dv_k', 'mean')].std()))

    # aspect dependence of the COMMON (keypoint-pooled) residual, for contrast
    aspc = d.groupby(['cam', 'ab'])[['du', 'dv']].mean()
    sd_aspect_common = float(np.hypot(aspc.du.std(), aspc.dv.std()))

    print(f'  residual magnitude overall            : {np.hypot(d.du, d.dv).median():.2f} px (median)')
    print(f'  [A] COMMON field vs image position    : SD {sd_common:.2f} px')
    print(f'  [B] KEYPOINT-specific (after A)       : SD {sd_kp:.2f} px')
    print(f'  [C] ASPECT-dependent, per keypoint    : SD {sd_aspect:.2f} px')
    print(f'      (aspect dependence pooled over kp : SD {sd_aspect_common:.2f} px)')
    print(f'  ratio keypoint-specific / common      : {sd_kp/max(sd_common,1e-9):.2f}')

    # ---- 4. which predicts the ARENA-SECTOR structure of the residual?
    sec_mean = d.groupby('sector')[['du', 'dv']].mean()
    # prediction from the image-position field alone
    d['pred_u'] = common.du; d['pred_v'] = common.dv
    sec_pred = d.groupby('sector')[['pred_u', 'pred_v']].mean()
    j = sec_mean.join(sec_pred).dropna()
    if len(j) > 3:
        ru = np.corrcoef(j.du, j.pred_u)[0, 1]; rv = np.corrcoef(j.dv, j.pred_v)[0, 1]
        print(f'  sector-structure explained by the image-position field: '
              f'r_u={ru:+.3f}  r_v={rv:+.3f}')
        out['sector_r_u'] = float(ru); out['sector_r_v'] = float(rv)
    # per-keypoint aspect profile: report the most aspect-sensitive keypoints
    if len(asp):
        mag = np.hypot(asp[('du_k', 'mean')], asp[('dv_k', 'mean')])
        top = mag.groupby(level='kp').max().sort_values(ascending=False).head(6)
        print('  most aspect-sensitive keypoints (max |residual| across aspect bins):')
        for kpi, v in top.items():
            print(f'      {NODES[int(kpi)]:<10} {v:.2f} px')
    out.update(dict(median_resid=float(np.hypot(d.du, d.dv).median()),
                    sd_common=sd_common, sd_kp=sd_kp, sd_aspect=sd_aspect,
                    sd_aspect_common=sd_aspect_common))
    return out


if __name__ == '__main__':
    res = {}
    for rat in ['R4', 'R5', 'R6']:
        print(f'collecting {rat} ...', flush=True)
        d = collect(rat)
        if d is None or len(d) < 5000:
            print(f'  {rat}: insufficient data'); continue
        d.to_parquet if False else None
        res[rat] = analyse(rat, d)
    with open(f'{OUT}/stage9_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f'\nwrote {OUT}/stage9_results.json')
