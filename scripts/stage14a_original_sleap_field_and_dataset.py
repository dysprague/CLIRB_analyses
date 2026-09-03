"""Stage 14a — retry the aspect-mechanism field measurement using the ORIGINAL
production SLEAP 2D output, not the corrector_v2 fine-tuned re-inference.

Everything in stage11/stage12a-g was built on `results/corrector_v2/keys2d/*.npz`
-- 2D keypoints from a FINE-TUNED SLEAP model, re-inferred on a curated,
balanced subset of frames (`v2_01_select_frames.py`/`v2_02_reinfer.py`) for the
(negative) corrector_v2 project. That is NOT the model that actually produced
`triang_keys_3D`/the online reward rule's behavior for R4/R5/R6 -- the real
production 2D lives in `sleap/sleap_keys_2D.npy` per session.

Frame-basis note (the user's reminder, already independently confirmed in
SPATIAL_CONFOUND_RESULTS.md Sec8c): `sleap_keys_2D.npy` is stored dense over the
processed-frame counter (row k = the k-th successfully processed frame; padded
with garbage past that count), NOT the raw acquisition-frame counter. The bridge
used throughout this project for SLEAP<->DANNCE is `aligned_data.npz`'s
`dannce_idx_for_sleap_cams_keys`, which is verified (empirically, below, and
previously in stage11/12a) to be indexed on that SAME processed-frame-counter
basis: `dannce_idx_for_sleap_cams_keys[k]` <-> `sleap_keys_2D[k]`. That is a
DIFFERENT bridge from `frame_mapping.csv`'s `cam0_frame` column, which instead
connects the processed-frame counter to the cam0 ACQUISITION-frame counter (the
basis `triang_keys_3D.npy` uses) -- not needed here since the reference is
DANNCE, not triang_keys_3D.

corrector_v2's own 'k' field is confirmed (by direct comparison) to be a row
index into this SAME `sleap_keys_2D.npy` processed-frame counter -- so the only
things that change here are (1) the keypoint VALUES (original model, not
fine-tuned) and (2) frame COVERAGE (every processed frame, not a curated
~1/8-density training subsample).

Writes:
  results/spatial_confound/field2d_original.npz       (aspect-only field, like field2d.npz)
  results/spatial_confound/dataset_original.npz        (XCAL, DNCAL, SECTOR, RAT -- like corrector_v2/dataset.npz but original 2D, ALL processed frames)
"""
import os, sys, glob
import numpy as np

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial import quality as q
from clirb.spatial.geometry import sector_index
from config import NODE_IDX
from projection import project_3d_to_2d
from corrector.world_alignment import fit_procrustes

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = f'{DR}/calibrations/2026_05_04'
SESS_SRC = 'results/corrector_v2/keys2d'      # just for the (rat, session) list
OUT_FIELD = 'results/spatial_confound/field2d_original.npz'
OUT_DATA = 'results/spatial_confound/dataset_original.npz'
CONF = 0.5
NASP = 12
MAXRES = 120.0
NKP = 23
SECTOR_W = 30
RNG = np.random.default_rng(0)


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

    S0 = np.zeros((NKP, 3, NASP))
    S1 = np.zeros((NKP, 3, NASP, 2))
    S2 = np.zeros((NKP, 3, NASP, 2, 2))

    XCAL_all, DNCAL_all, SEC_all, RAT_all = [], [], [], []
    checked_k_alignment = False

    sessions = sorted({os.path.basename(f)[:-4] for f in glob.glob(f'{SESS_SRC}/*.npz')})
    for base in sessions:
        rat, session = base.split('__')
        ap = f'{DR}/{rat}/{session}/processed/aligned_data.npz'
        dnp = f'{DR}/{rat}/{session}/processed/sleap_dannce_keys.npz'
        orig2d_path = f'{DR}/{rat}/{session}/sleap/sleap_keys_2D.npy'
        if not (os.path.isfile(ap) and os.path.isfile(dnp) and os.path.isfile(orig2d_path)):
            print(f'  {rat}/{session}: missing a required file, skip'); continue
        with np.load(ap, allow_pickle=True) as z:
            dk = z['dannce_idx_for_sleap_cams_keys']
        with np.load(dnp, allow_pickle=True) as z:
            dn_raw = z['dannce_keys_3D']
        dn = (dn_raw.squeeze(axis=1).transpose(0, 2, 1) if dn_raw.ndim == 4
              else np.transpose(dn_raw, [0, 2, 1]))

        n_proc = len(dk)
        k2_full = np.load(orig2d_path, mmap_mode='r')
        n_avail = min(n_proc, k2_full.shape[0])
        k = np.arange(n_avail)
        k2 = np.asarray(k2_full[:n_avail])              # ORIGINAL production 2D

        if not checked_k_alignment:
            # sanity: original 2D at row k should be a physically plausible pose,
            # not garbage/zero-padding, for k well inside n_avail
            probe = k2[min(1000, n_avail - 1)]
            assert np.isfinite(probe).any() and np.abs(probe[..., :2]).max() > 1, \
                'sleap_keys_2D looks like padding at a supposedly-valid row'
            checked_k_alignment = True

        tgt = dn[np.clip(dk[k], 0, len(dn) - 1)].astype(np.float64)
        und = q.undistort_all(np.nan_to_num(k2[..., :2]).astype(np.float64), K, D)
        X = q.triangulate_batch(und, P)
        good = (np.isfinite(X).all(axis=(1, 2)) & np.isfinite(tgt).all(axis=(1, 2)))
        rad = np.linalg.norm(X - X.mean(axis=1, keepdims=True), axis=-1).max(axis=1)
        good &= rad < 250
        if good.sum() < 400:
            print(f'  {rat}/{session}: only {good.sum()} usable frames, skip'); continue
        X, tgt, k2g = X[good], tgt[good], k2[good]

        # sector-balanced Procrustes fit, same convention as v2_03_build_dataset.py
        com = X[:, :, :2].mean(axis=1)
        th = np.degrees(np.arctan2(com[:, 1], com[:, 0]))
        sec = sector_index(th, SECTOR_W)
        samp = []
        for sname in range(int(round(360 / SECTOR_W))):
            ii = np.where(sec == sname)[0]
            if len(ii) > 20:
                samp.extend(RNG.choice(ii, min(len(ii), 400), replace=False))
        samp = np.array(samp) if len(samp) > 300 else np.arange(len(X))
        fit = robust_proc(tgt[samp], X[samp])
        dn_cal = fit['apply'](tgt)
        resid = float(np.median(np.linalg.norm(dn_cal - X, axis=-1)))
        if resid > 60:
            print(f'  {rat}/{session}: Procrustes residual {resid:.1f} mm too high, skip')
            continue

        XCAL_all.append(X.astype(np.float32))
        DNCAL_all.append(dn_cal.astype(np.float32))
        SEC_all.append(sec.astype(np.int16))
        RAT_all.append(np.full(len(X), rat))

        # ---- field2d accumulation (aspect-only, body-relative image frame) ----
        SF3 = dn_cal[:, NODE_IDX['SpineF']]; SM3 = dn_cal[:, NODE_IDX['SpineM']]
        head = np.degrees(np.arctan2(SF3[:, 1] - SM3[:, 1], SF3[:, 0] - SM3[:, 0]))
        for c in range(3):
            proj = project_3d_to_2d(dn_cal.reshape(-1, 3), K[c], R[c], T[c],
                                    D[c]).reshape(len(dn_cal), NKP, 2)
            pM = project_3d_to_2d(SM3, K[c], R[c], T[c], D[c])
            pF = project_3d_to_2d(SF3, K[c], R[c], T[c], D[c])
            phi = np.arctan2(pF[:, 1] - pM[:, 1], pF[:, 0] - pM[:, 0])
            cp, sp = np.cos(-phi), np.sin(-phi)
            bear = np.degrees(np.arctan2(com[:, 1] - cams[c][1],
                                         com[:, 0] - cams[c][0]))
            abin = np.clip(((wrap(head - bear) + 180) / (360 / NASP)).astype(int),
                           0, NASP - 1)

            det = k2g[:, c, :, :2].astype(np.float64)
            cf = k2g[:, c, :, 2]
            res = det - proj
            rb = np.stack([cp[:, None] * res[..., 0] - sp[:, None] * res[..., 1],
                           sp[:, None] * res[..., 0] + cp[:, None] * res[..., 1]],
                          axis=-1)
            keep = (cf >= CONF) & np.isfinite(rb).all(axis=-1) & \
                   (np.linalg.norm(rb, axis=-1) < MAXRES)
            for j in range(NKP):
                m = keep[:, j]
                if not m.any():
                    continue
                v = rb[m, j]; b = abin[m]
                np.add.at(S0[j, c], b, 1.0)
                np.add.at(S1[j, c], b, v)
                np.add.at(S2[j, c], b, v[:, :, None] * v[:, None, :])
        print(f'  {rat}/{session}: {len(X)} frames (n_processed={n_proc}, '
              f'orig2d rows={k2_full.shape[0]}), Procrustes resid {resid:.1f} mm', flush=True)

    n = np.maximum(S0, 1)[..., None]
    mean = S1 / n
    cov = S2 / np.maximum(S0, 1)[..., None, None] - mean[..., :, None] * mean[..., None, :]
    np.savez_compressed(OUT_FIELD, count=S0, mean=mean, cov=cov, nasp=NASP)
    print(f'\nwrote {OUT_FIELD}')
    print(f'  cells with >=100 obs: {(S0 >= 100).sum()} / {S0.size}')
    print(f'  median |mean bias| over populated cells: '
          f'{np.median(np.linalg.norm(mean[S0 >= 100], axis=-1)):.2f} px')

    XCAL = np.concatenate(XCAL_all); DNCAL = np.concatenate(DNCAL_all)
    SEC = np.concatenate(SEC_all); RATA = np.concatenate(RAT_all)
    np.savez_compressed(OUT_DATA, XCAL=XCAL, DNCAL=DNCAL, SECTOR=SEC, RAT=RATA)
    print(f'wrote {OUT_DATA}: {len(XCAL)} frames total')
    for rat in np.unique(RATA):
        print(f'  {rat}: {(RATA == rat).sum()} frames')


if __name__ == '__main__':
    main()
