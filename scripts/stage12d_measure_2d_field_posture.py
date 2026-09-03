"""Stage 12d — measure the 2D error field binned by (keypoint, camera, aspect,
POSTURE), not aspect alone.

Motivation (SPATIAL_CONFOUND_RESULTS.md §8f): the aspect-only propagation test
(stage12a/b/c) predicts only ~20-25% of the measured mu(theta) amplitude. The
leading suspect named there is that 2D self-occlusion also depends on POSTURE —
a full rear and a crouch present completely different silhouettes at the same
viewing aspect — and posture varies systematically by arena sector (apex_z
spans 2.2-6.9 in R4). This was never conditioned on. Here it is.

posture index = max_z(all 23 DANNCE keypoints) - z(SpineM), evaluated on the
DANNCE pose already placed in the calibration frame (dn_cal, identical
construction to stage12a/stage11). SpineM-relative so it does not depend on
the session's absolute floor level or the Procrustes translation; this is the
same "how far above the animal's own base is its highest point" quantity as
geometry.apex_height, just referenced to SpineM instead of the raw z origin.

Binned into NPOST quantile bins (tertiles by default) fit on the pooled
distribution across all sessions.

Writes results/spatial_confound/field2d_posture.npz
    count (23,3,NASP,NPOST)
    mean  (23,3,NASP,NPOST,2)     body-relative 2D bias
    cov   (23,3,NASP,NPOST,2,2)   body-relative 2D noise
    post_edges (NPOST+1,)         mm, the bin edges used
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
OUT = 'results/spatial_confound/field2d_posture.npz'
CONF = 0.5
NASP = 12                      # 30-degree aspect bins, same as stage12a
NPOST = 3                      # posture tertiles: crouch / mid / rear
MAXRES = 120.0                 # px, gross-outlier cut, same as stage12a
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


def collect():
    P, K, D, R, T = q.load_calib(CAL)
    cams = np.stack([-R[c].T @ T[c] for c in range(3)])
    recs = []
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
        good = (np.isfinite(X).all(axis=(1, 2)) & np.isfinite(tgt).all(axis=(1, 2)))
        rad = np.linalg.norm(X - X.mean(axis=1, keepdims=True), axis=-1).max(axis=1)
        good &= rad < 250
        if good.sum() < 400:
            continue
        X, tgt, k2 = X[good], tgt[good], k2[good]
        dn_cal = robust_proc(tgt, X)['apply'](tgt)

        com = dn_cal[:, :, :2].mean(axis=1)
        SF3 = dn_cal[:, NODE_IDX['SpineF']]; SM3 = dn_cal[:, NODE_IDX['SpineM']]
        head = np.degrees(np.arctan2(SF3[:, 1] - SM3[:, 1], SF3[:, 0] - SM3[:, 0]))
        posture = dn_cal[:, :, 2].max(axis=1) - SM3[:, 2]

        for c in range(3):
            proj = project_3d_to_2d(dn_cal.reshape(-1, 3), K[c], R[c], T[c],
                                    D[c]).reshape(len(dn_cal), NKP, 2)
            pM = project_3d_to_2d(SM3, K[c], R[c], T[c], D[c])
            pF = project_3d_to_2d(SF3, K[c], R[c], T[c], D[c])
            phi = np.arctan2(pF[:, 1] - pM[:, 1], pF[:, 0] - pM[:, 0])
            cp, sp = np.cos(-phi), np.sin(-phi)
            bear = np.degrees(np.arctan2(com[:, 1] - cams[c][1], com[:, 0] - cams[c][0]))
            aspect = wrap(head - bear)

            det = k2[:, c, :, :2].astype(np.float64)
            cf = k2[:, c, :, 2]
            res = det - proj
            rb = np.stack([cp[:, None] * res[..., 0] - sp[:, None] * res[..., 1],
                           sp[:, None] * res[..., 0] + cp[:, None] * res[..., 1]],
                          axis=-1)
            keep = (cf >= CONF) & np.isfinite(rb).all(axis=-1) & \
                   (np.linalg.norm(rb, axis=-1) < MAXRES)
            recs.append(dict(rat=rat, session=session, cam=c, rb=rb, keep=keep,
                             aspect=aspect, posture=posture))
        print(f'  {rat}/{session}: {good.sum()} frames', flush=True)
    return recs


def main():
    recs = collect()
    all_post = np.concatenate([r['posture'] for r in recs])
    edges = np.quantile(all_post, np.linspace(0, 1, NPOST + 1))
    edges[0] -= 1.0; edges[-1] += 1.0
    print(f'\nposture index (max_z - z_SpineM), n={len(all_post)}:')
    print(f'  median={np.median(all_post):.1f} mm  range=[{all_post.min():.1f}, '
          f'{all_post.max():.1f}]  tertile edges={np.round(edges, 1)}')

    S0 = np.zeros((NKP, 3, NASP, NPOST))
    S1 = np.zeros((NKP, 3, NASP, NPOST, 2))
    S2 = np.zeros((NKP, 3, NASP, NPOST, 2, 2))

    for r in recs:
        c = r['cam']
        abin = np.clip(((r['aspect'] + 180) / (360 / NASP)).astype(int), 0, NASP - 1)
        pbin = np.clip(np.digitize(r['posture'], edges) - 1, 0, NPOST - 1)
        lin = abin * NPOST + pbin
        for j in range(NKP):
            m = r['keep'][:, j]
            if not m.any():
                continue
            v = r['rb'][m, j]
            idx = lin[m]
            np.add.at(S0[j, c].reshape(NASP * NPOST), idx, 1.0)
            np.add.at(S1[j, c].reshape(NASP * NPOST, 2), idx, v)
            np.add.at(S2[j, c].reshape(NASP * NPOST, 2, 2), idx,
                     v[:, :, None] * v[:, None, :])

    n = np.maximum(S0, 1)[..., None]
    mean = S1 / n
    cov = S2 / np.maximum(S0, 1)[..., None, None] - mean[..., :, None] * mean[..., None, :]
    np.savez_compressed(OUT, count=S0, mean=mean, cov=cov, nasp=NASP, npost=NPOST,
                        post_edges=edges)
    print(f'\nwrote {OUT}')
    print(f'  cells (kp,cam,aspect,posture) with >=50 obs: {(S0 >= 50).sum()} / {S0.size}')
    print(f'  cells with >=20 obs: {(S0 >= 20).sum()} / {S0.size}')
    print(f'  cells with 0 obs:    {(S0 == 0).sum()} / {S0.size}')
    tr = np.trace(cov, axis1=-2, axis2=-1)
    print(f'  median |mean bias| over populated cells (>=20): '
          f'{np.median(np.linalg.norm(mean[S0 >= 20], axis=-1)):.2f} px')
    print(f'  median sqrt(trace cov) over populated cells (>=20): '
          f'{np.median(np.sqrt(tr[S0 >= 20])):.2f} px')
    # sanity: does bias amplitude actually change with posture, pooled over aspect?
    for p in range(NPOST):
        sub = mean[:, :, :, p]; cnt = S0[:, :, :, p]
        m = cnt >= 20
        print(f'  posture bin {p} ({edges[p]:.0f}-{edges[p+1]:.0f} mm): '
              f'median |bias| = {np.median(np.linalg.norm(sub[m], axis=-1)):.2f} px  '
              f'(n cells={m.sum()})')


if __name__ == '__main__':
    main()
