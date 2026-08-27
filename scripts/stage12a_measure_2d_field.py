"""Stage 12a — measure the 2D error field in a BODY-RELATIVE image frame.

To propagate error from image space into 3D at an arbitrary arena position, the
measured 2D error must be a function of viewing geometry ALONE, not of where the
animal happened to be in the image. So each frame's 2D residual is rotated into a
frame aligned with the animal's PROJECTED body axis in that camera:

    phi_c   = image angle of the projected SpineM->SpineF axis in camera c
    r_body  = R(-phi_c) @ (detection - DANNCE_projection)

Binned by (keypoint, camera, viewing aspect) this gives, per cell:
    mean   (2,)    the systematic 2D bias, body-relative   -> predicts mu(theta)
    cov    (2,2)   the 2D noise                            -> predicts Sigma(theta)

Reference is the independent 6-camera DANNCE pose reprojected into each SLEAP
camera, so the field does not inherit SLEAP's own errors.

Writes results/spatial_confound/field2d.npz
"""
import os, sys, glob
import numpy as np

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial import quality as q
from config import NODE_IDX
from projection import project_3d_to_2d
from corrector.world_alignment import fit_procrustes

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = f'{DR}/calibrations/2026_05_04'
IN = 'results/corrector_v2/keys2d'
OUT = 'results/spatial_confound/field2d.npz'
CONF = 0.5
NASP = 12                      # 30-degree aspect bins
MAXRES = 120.0            # px, gross-outlier cut
NKP = 23


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def robust_proc(A, B, iters=5, keep=0.7):
    idx = np.arange(len(A)); f = None
    for _ in range(iters):
        f = fit_procrustes(A[idx], B[idx], try_z_flip=True)
        r = np.linalg.norm(f['apply'](A) - B, axis=-1).mean(axis=1)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 200:
            break
    return f


def main():
    P, K, D, R, T = q.load_calib(CAL)
    cams = np.stack([-R[c].T @ T[c] for c in range(3)])

    S0 = np.zeros((NKP, 3, NASP))            # count
    S1 = np.zeros((NKP, 3, NASP, 2))         # sum of residual
    S2 = np.zeros((NKP, 3, NASP, 2, 2))      # sum of outer products

    for f in sorted(glob.glob(f'{IN}/*.npz')):
        rat, session = os.path.basename(f)[:-4].split('__')
        with np.load(f) as z:
            k = z['k']; k2 = z['keys2d']
        ap = f'{DR}/{rat}/{session}/processed/aligned_data.npz'
        dnp = f'{DR}/{rat}/{session}/processed/sleap_dannce_keys.npz'
        if not (os.path.isfile(ap) and os.path.isfile(dnp)):
            continue
        with np.load(ap, allow_pickle=True) as z:
            dk = z['dannce_idx_for_sleap_cams_keys']
        with np.load(dnp, allow_pickle=True) as z:
            dn_raw = z['dannce_keys_3D']
        dn = (dn_raw.squeeze(axis=1).transpose(0, 2, 1) if dn_raw.ndim == 4
              else np.transpose(dn_raw, [0, 2, 1]))
        ok = k < len(dk)
        k, k2 = k[ok], k2[ok]
        tgt = dn[np.clip(dk[k], 0, len(dn) - 1)].astype(np.float64)

        und = q.undistort_all(np.nan_to_num(k2[..., :2]).astype(np.float64), K, D)
        X = q.triangulate_batch(und, P)
        good = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(tgt).all(axis=(1, 2))
        rad = np.linalg.norm(X - X.mean(axis=1, keepdims=True), axis=-1).max(axis=1)
        good &= rad < 250
        if good.sum() < 400:
            continue
        X, tgt, k2 = X[good], tgt[good], k2[good]

        # DANNCE -> calibration frame
        dn_cal = robust_proc(tgt, X)['apply'](tgt)
        com = dn_cal[:, :, :2].mean(axis=1)
        SF3 = dn_cal[:, NODE_IDX['SpineF']]; SM3 = dn_cal[:, NODE_IDX['SpineM']]
        head = np.degrees(np.arctan2(SF3[:, 1] - SM3[:, 1], SF3[:, 0] - SM3[:, 0]))

        for c in range(3):
            proj = project_3d_to_2d(dn_cal.reshape(-1, 3), K[c], R[c], T[c],
                                    D[c]).reshape(len(dn_cal), NKP, 2)
            pM = project_3d_to_2d(SM3, K[c], R[c], T[c], D[c])
            pF = project_3d_to_2d(SF3, K[c], R[c], T[c], D[c])
            phi = np.arctan2(pF[:, 1] - pM[:, 1], pF[:, 0] - pM[:, 0])
            cp, sp = np.cos(-phi), np.sin(-phi)

            bear = np.degrees(np.arctan2(com[:, 1] - cams[c][1], com[:, 0] - cams[c][0]))
            abin = np.clip(((wrap(head - bear) + 180) / (360 / NASP)).astype(int),
                           0, NASP - 1)

            det = k2[:, c, :, :2].astype(np.float64)
            cf = k2[:, c, :, 2]
            res = det - proj
            # rotate into the body-relative image frame
            rb = np.stack([cp[:, None] * res[..., 0] - sp[:, None] * res[..., 1],
                           sp[:, None] * res[..., 0] + cp[:, None] * res[..., 1]],
                          axis=-1)
            keep = (cf >= CONF) & np.isfinite(rb).all(axis=-1) & \
                   (np.linalg.norm(rb, axis=-1) < MAXRES)
            for j in range(NKP):
                m = keep[:, j]
                if not m.any():
                    continue
                v = rb[m, j]                       # (n,2)
                b = abin[m]
                np.add.at(S0[j, c], b, 1.0)
                np.add.at(S1[j, c], b, v)
                np.add.at(S2[j, c], b, v[:, :, None] * v[:, None, :])
        print(f'  {rat}/{session}: done', flush=True)

    n = np.maximum(S0, 1)[..., None]
    mean = S1 / n
    cov = S2 / np.maximum(S0, 1)[..., None, None] - mean[..., :, None] * mean[..., None, :]
    np.savez_compressed(OUT, count=S0, mean=mean, cov=cov, nasp=NASP)
    print(f'\nwrote {OUT}')
    print(f'  cells with >=100 obs: {(S0 >= 100).sum()} / {S0.size}')
    print(f'  median |mean bias| over populated cells: '
          f'{np.median(np.linalg.norm(mean[S0 >= 100], axis=-1)):.2f} px')
    tr = np.trace(cov, axis1=-2, axis2=-1)
    print(f'  median sqrt(trace cov): {np.median(np.sqrt(tr[S0 >= 100])):.2f} px')


if __name__ == '__main__':
    main()
