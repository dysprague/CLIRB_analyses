"""Corrector v2, step 3 — triangulate the new 2D and pair with DANNCE.

  new 2D (fine-tuned model)  --triangulate(05_04)-->  raw 3D in the CALIBRATION frame
  DANNCE target              via dannce_idx_for_sleap_cams_keys[k]
  per-session sector-balanced Procrustes  raw 3D -> DANNCE space

Also stores the ASPECT features the aspect-aware corrector needs: the animal's
heading relative to each camera's bearing, computed in the calibration frame
(where camera geometry lives) BEFORE alignment.

Writes results/corrector_v2/dataset.npz
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial import quality as q
from clirb.spatial.geometry import sector_index
from config import NODE_IDX
from corrector.world_alignment import fit_procrustes

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data/calibrations/2026_05_04'
IN = 'results/corrector_v2/keys2d'
OUT = 'results/corrector_v2'
SECTOR_W, NSEC = 30, 12
RNG = np.random.default_rng(0)


def robust_procrustes(A, B, iters=5, keep=0.7):
    idx = np.arange(len(A))
    f = None
    for _ in range(iters):
        f = fit_procrustes(A[idx], B[idx], try_z_flip=True)
        r = np.linalg.norm(f['apply'](A) - B, axis=-1).mean(axis=1)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 200:
            break
    return f


def main():
    segs = pd.read_csv(f'{OUT}/segments.csv')
    P, K, D, R, T = q.load_calib(CAL)
    cam_centres = np.stack([-R[c].T @ T[c] for c in range(3)])

    X_all, Y_all, ASP_all, XCAL_all = [], [], [], []
    TF = {}
    meta = []
    files = sorted(glob.glob(f'{IN}/*.npz'))
    print(f'{len(files)} session files', flush=True)
    for f in files:
        base = os.path.basename(f)[:-4]
        rat, session = base.split('__')
        with np.load(f) as z:
            k = z['k']; k2 = z['keys2d']
        ap = f'{DR}/{rat}/{session}/processed/aligned_data.npz'
        if not os.path.isfile(ap):
            print(f'  {rat}/{session}: no aligned_data'); continue
        with np.load(ap, allow_pickle=True) as z:
            dk = z['dannce_idx_for_sleap_cams_keys']
        dnp = f'{DR}/{rat}/{session}/processed/sleap_dannce_keys.npz'
        if not os.path.isfile(dnp):
            print(f'  {rat}/{session}: no sleap_dannce_keys'); continue
        with np.load(dnp, allow_pickle=True) as z:
            dn_raw = z['dannce_keys_3D']
        dn = (dn_raw.squeeze(axis=1).transpose(0, 2, 1) if dn_raw.ndim == 4
              else np.transpose(dn_raw, [0, 2, 1]))

        ok_k = k < len(dk)
        k = k[ok_k]; k2 = k2[ok_k]
        tgt = dn[np.clip(dk[k], 0, len(dn) - 1)].astype(np.float64)

        # triangulate the NEW 2D with the 05_04 calibration
        und = q.undistort_all(np.nan_to_num(k2[..., :2]).astype(np.float64), K, D)
        X = q.triangulate_batch(und, P)

        good = (np.isfinite(X).all(axis=(1, 2)) & np.isfinite(tgt).all(axis=(1, 2))
                & (np.nanmedian(k2[..., 2], axis=(1, 2)) > 0.1))
        if good.sum() < 200:
            print(f'  {rat}/{session}: only {good.sum()} usable frames'); continue
        X, tgt, k, k2 = X[good], tgt[good], k[good], k2[good]

        # sector-balanced Procrustes (never a time-based epoch)
        com = X[:, :, :2].mean(axis=1)
        th = np.degrees(np.arctan2(com[:, 1], com[:, 0]))
        sec = sector_index(th, SECTOR_W)
        samp = []
        for sname in range(NSEC):
            ii = np.where(sec == sname)[0]
            if len(ii) > 20:
                samp.extend(RNG.choice(ii, min(len(ii), 400), replace=False))
        samp = np.array(samp) if len(samp) > 300 else np.arange(len(X))
        fit = robust_procrustes(X[samp], tgt[samp])
        Xa = fit['apply'](X)                      # SLEAP raw -> DANNCE space
        resid = float(np.median(np.linalg.norm(Xa - tgt, axis=-1)))
        if resid > 40:
            print(f'  {rat}/{session}: Procrustes residual {resid:.1f} mm > 40, skip')
            continue

        # Context features in the CALIBRATION frame (camera geometry lives there).
        # A POSITION-DEPENDENT bias cannot be corrected by a model that only sees
        # a centroid-centred pose, so the context vector carries both the viewing
        # aspect (heading relative to each camera) AND where the animal is.
        #   0-5  : cos/sin of heading relative to each camera's bearing
        #   6-8  : centroid x,y,z / 500 mm
        #   9    : distance from each camera, mean, / 2000 mm
        #   10-11: cos/sin of arena angle
        SF = X[:, NODE_IDX['SpineF'], :2]; SM = X[:, NODE_IDX['SpineM'], :2]
        head = np.arctan2(SF[:, 1] - SM[:, 1], SF[:, 0] - SM[:, 0])
        asp = np.zeros((len(X), 12), dtype=np.float32)
        for c in range(3):
            bear = np.arctan2(com[:, 1] - cam_centres[c][1], com[:, 0] - cam_centres[c][0])
            rel = head - bear
            asp[:, 2 * c] = np.cos(rel); asp[:, 2 * c + 1] = np.sin(rel)
        cen = X.mean(axis=1)
        asp[:, 6:9] = (cen / 500.0).astype(np.float32)
        asp[:, 9] = np.mean([np.linalg.norm(cen - cc, axis=1) for cc in cam_centres],
                            axis=0) / 2000.0
        th_a = np.arctan2(com[:, 1], com[:, 0])
        asp[:, 10] = np.cos(th_a); asp[:, 11] = np.sin(th_a)

        # segment id per frame, so temporal models never cross a segment boundary
        sg = segs[(segs.rat == rat) & (segs.session == session)]
        seg_of = np.full(len(k), -1, dtype=np.int64)
        for i, (_, r) in enumerate(sg.iterrows()):
            m = (k >= r.k_start) & (k < r.k_end)
            seg_of[m] = i

        X_all.append(Xa.astype(np.float32)); Y_all.append(tgt.astype(np.float32))
        ASP_all.append(asp)
        XCAL_all.append(X.astype(np.float32))       # raw triangulation, CALIB frame
        # store the transform so Tier-2 can map corrected poses (DANNCE space)
        # back into the calibration frame, which is where triang_3d and the
        # template PC basis live
        TF[f'{rat}/{session}'] = dict(R=fit['R'].tolist(), s=float(fit['s']),
                                      t=np.asarray(fit['t']).ravel().tolist(),
                                      z=bool(fit['z_flipped']), resid=resid)
        third = int(sg.third.iloc[0]) if len(sg) else 0
        for i in range(len(k)):
            meta.append((rat, session, third, int(k[i]), int(seg_of[i]), int(sec[i]),
                         float(np.nanmean(k2[i, :, :, 2]))))
        print(f'  {rat}/{session}: {len(k)} frames, Procrustes resid {resid:.1f} mm', flush=True)

    X = np.concatenate(X_all); Y = np.concatenate(Y_all); A = np.concatenate(ASP_all)
    M = pd.DataFrame(meta, columns=['rat', 'session', 'third', 'k', 'seg', 'sector', 'conf'])
    M['gid'] = M.rat + '/' + M.session + '/' + M.seg.astype(str)
    XC = np.concatenate(XCAL_all)
    np.savez_compressed(f'{OUT}/dataset.npz', X=X, Y=Y, ASP=A, XCAL=XC)
    json.dump(TF, open(f'{OUT}/frame_transforms.json', 'w'), indent=1)
    M.to_csv(f'{OUT}/dataset_meta.csv', index=False)
    print(f'\nwrote dataset: X{X.shape} Y{Y.shape} ASP{A.shape}')
    print(f'  sessions {M.session.nunique()}  rats {M.rat.nunique()}  segments {M.gid.nunique()}')
    print(f'  baseline median |SLEAP-DANNCE| = '
          f'{np.median(np.linalg.norm(X - Y, axis=-1)):.2f} mm')
    print('  sector distribution:'); print(M.sector.value_counts(normalize=True).sort_index().round(3).to_string())


if __name__ == '__main__':
    main()
