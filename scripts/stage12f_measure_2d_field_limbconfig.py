"""Stage 12f — measure the 2D error field binned by (keypoint, camera, aspect,
LIMB CONFIGURATION), replacing the scalar height proxy (stage12d) with a
multi-dimensional posture descriptor.

stage12d/e tested "posture = body height above SpineM" and it did not close
the mechanism gap: the field it measured was almost aspect-only (0.4 px
posture-marginal range vs 3.0 px aspect-marginal range). The open alternative
named in SPATIAL_CONFOUND_RESULTS.md §8g was "a richer posture representation
-- limb configuration rather than body height."

Here, posture = a KMeans cluster label fit on the EGOCENTRIC pose shape
(translation + yaw removed via skeleton.normalize_skeleton_batch, exactly the
convention the template builder itself uses for shape clustering) rather than
a single scalar. This captures which limbs are extended/tucked, forelimb
posture, spine curvature, etc. -- not just how high the animal's back is.

Writes results/spatial_confound/field2d_limbconfig.npz, including the KMeans
centroids + normalization so stage12g can assign the SAME cluster labels to
frames at prediction time.
"""
import os, sys, glob
import numpy as np
from sklearn.cluster import KMeans

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial import quality as q
from config import NODE_IDX
from projection import project_3d_to_2d
from corrector.world_alignment import fit_procrustes
from skeleton import normalize_skeleton_batch

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = f'{DR}/calibrations/2026_05_04'
IN = 'results/corrector_v2/keys2d'
OUT = 'results/spatial_confound/field2d_limbconfig.npz'
CONF = 0.5
NASP = 12
NCLUST = 4                     # limb-configuration clusters
MAXRES = 120.0
NKP = 23
RNG_SEED = 0


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
        ego, _, _ = normalize_skeleton_batch(dn_cal)          # (n,23,3) shape only

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
                             aspect=aspect, ego=ego))
        print(f'  {rat}/{session}: {good.sum()} frames', flush=True)
    return recs


def main():
    recs = collect()
    ego_all = np.concatenate([r['ego'] for r in recs])       # (N,23,3), duplicated x3 cams
    # de-duplicate: recs has 3 entries (one per camera) per session-frame-block,
    # all sharing the SAME ego array object per session -> fit KMeans on unique
    # per-session ego blocks only, to avoid triple-weighting the same frames
    seen = set(); uniq = []
    for r in recs:
        key = (r['rat'], r['session'])
        if key in seen:
            continue
        seen.add(key); uniq.append(r['ego'])
    ego_fit = np.concatenate(uniq)
    flat = ego_fit.reshape(len(ego_fit), -1)
    mu, sd = flat.mean(axis=0), flat.std(axis=0) + 1e-9
    z = (flat - mu) / sd
    km = KMeans(n_clusters=NCLUST, n_init=10, random_state=RNG_SEED).fit(z)
    print(f'\nKMeans on egocentric pose shape: n={len(flat)} frames, {NCLUST} clusters')
    print('cluster sizes:', np.bincount(km.labels_))

    S0 = np.zeros((NKP, 3, NASP, NCLUST))
    S1 = np.zeros((NKP, 3, NASP, NCLUST, 2))
    S2 = np.zeros((NKP, 3, NASP, NCLUST, 2, 2))

    for r in recs:
        c = r['cam']
        abin = np.clip(((r['aspect'] + 180) / (360 / NASP)).astype(int), 0, NASP - 1)
        ez = (r['ego'].reshape(len(r['ego']), -1) - mu) / sd
        cbin = km.predict(ez)
        lin = abin * NCLUST + cbin
        for j in range(NKP):
            m = r['keep'][:, j]
            if not m.any():
                continue
            v = r['rb'][m, j]
            idx = lin[m]
            np.add.at(S0[j, c].reshape(NASP * NCLUST), idx, 1.0)
            np.add.at(S1[j, c].reshape(NASP * NCLUST, 2), idx, v)
            np.add.at(S2[j, c].reshape(NASP * NCLUST, 2, 2), idx,
                     v[:, :, None] * v[:, None, :])

    n = np.maximum(S0, 1)[..., None]
    mean = S1 / n
    cov = S2 / np.maximum(S0, 1)[..., None, None] - mean[..., :, None] * mean[..., None, :]
    np.savez_compressed(OUT, count=S0, mean=mean, cov=cov, nasp=NASP, nclust=NCLUST,
                        centroids=km.cluster_centers_, norm_mu=mu, norm_sd=sd)
    print(f'\nwrote {OUT}')
    print(f'  cells (kp,cam,aspect,cluster) with >=50 obs: {(S0 >= 50).sum()} / {S0.size}')
    print(f'  cells with >=20 obs: {(S0 >= 20).sum()} / {S0.size}')
    print(f'  cells with 0 obs:    {(S0 == 0).sum()} / {S0.size}')
    for cl in range(NCLUST):
        sub = mean[:, :, :, cl]; cnt = S0[:, :, :, cl]
        m = cnt >= 20
        print(f'  cluster {cl}: median |bias| = '
              f'{np.median(np.linalg.norm(sub[m], axis=-1)):.2f} px  (n cells={m.sum()})')


if __name__ == '__main__':
    main()
