"""Truth-free per-frame tracking-quality metrics.

None of these need DANNCE, so they cover all 151 sessions and every part of the
arena. The expensive one (reprojection residual) is VECTORIZED here -- the
notebook version (multi_session cell 50) loops T frames x 23 joints with a
Python-level SVD each and caps at 20k frames.
"""
import os
import glob
import numpy as np
import scipy.io as sio
import cv2

import sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from config import EDGES  # noqa: E402
from projection import project_3d_to_2d  # noqa: E402

DATA_ROOT = "/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data"
N_CAMS = 3
N_KP = 23


def calib_dir(rat, session):
    root = os.path.join(DATA_ROOT, rat, session, 'sleap', 'calibration')
    if not os.path.isdir(root):
        return None
    subs = sorted(os.listdir(root))
    return os.path.join(root, subs[0]) if subs else None


def load_calib(cal_dir):
    """(P_list, K_list, dist_list, r_list, t_list). P uses identity for K
    because undistortion already applies it (campy-CLIRB convention)."""
    P, K, D, R, T = [], [], [], [], []
    for cam in range(N_CAMS):
        p = sio.loadmat(os.path.join(cal_dir, f'hires_cam{cam}_params.mat'),
                        simplify_cells=True)
        k = np.transpose(p['K']); r = np.transpose(p['r']); t = -p['t']
        Rd, Td = p['RDistort'], p['TDistort']
        d = [Rd[0], Rd[1], Td[0], Td[1]]
        P.append(np.identity(3) @ np.hstack((r, t.reshape(3, 1))))
        K.append(k); D.append(d); R.append(r); T.append(t)
    return P, K, D, R, T


def undistort_all(keys_2d, K_list, dist_list):
    """(T,3,23,2) pixel -> (T,3,23,2) normalized undistorted, vectorized."""
    T = keys_2d.shape[0]
    out = np.empty((T, N_CAMS, N_KP, 2), dtype=np.float64)
    for cam in range(N_CAMS):
        pts = keys_2d[:, cam, :, :].reshape(-1, 1, 2).astype(np.float64)
        u = cv2.undistortPoints(pts, K_list[cam], np.asarray(dist_list[cam]))
        out[:, cam] = u.reshape(T, N_KP, 2)
    return out


def triangulate_batch(undist, P_list, chunk=200_000):
    """Vectorized DLT triangulation.

    undist : (T, 3, 23, 2) normalized undistorted coords
    returns (T, 23, 3)
    """
    T = undist.shape[0]
    N = T * N_KP
    # A[n] is (2*ncams, 4)
    x = undist[..., 0].transpose(0, 2, 1).reshape(N, N_CAMS)   # (N, cams)
    y = undist[..., 1].transpose(0, 2, 1).reshape(N, N_CAMS)
    P = np.stack(P_list)                                        # (cams, 3, 4)
    out = np.empty((N, 3), dtype=np.float64)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        xs, ys = x[s:e], y[s:e]
        n = e - s
        A = np.empty((n, 2 * N_CAMS, 4), dtype=np.float64)
        for i in range(N_CAMS):
            A[:, 2 * i]     = xs[:, i, None] * P[i, 2][None, :] - P[i, 0][None, :]
            A[:, 2 * i + 1] = ys[:, i, None] * P[i, 2][None, :] - P[i, 1][None, :]
        _, _, Vt = np.linalg.svd(A)
        h = Vt[:, -1, :]
        w = h[:, 3]
        bad = np.abs(w) < 1e-12
        w = np.where(bad, 1.0, w)
        pts = h[:, :3] / w[:, None]
        pts[bad] = np.nan
        out[s:e] = pts
    return out.reshape(T, N_KP, 3)


def reprojection_residual(keys_2d, cal_dir, conf_thresh=0.5):
    """Re-triangulate from the saved 2D detections, reproject, return
    (T,) median-over-(cam,kp) pixel residual and the raw 3D."""
    P, K, D, R, Tt = load_calib(cal_dir)
    kp2d = keys_2d[..., :2].astype(np.float64)
    conf = keys_2d[..., 2]
    undist = undistort_all(kp2d, K, D)
    raw3d = triangulate_batch(undist, P)
    T = keys_2d.shape[0]
    err = np.full((T, N_CAMS, N_KP), np.nan)
    flat = raw3d.reshape(-1, 3)
    ok = np.isfinite(flat).all(axis=1)
    for cam in range(N_CAMS):
        proj = np.full((flat.shape[0], 2), np.nan)
        proj[ok] = project_3d_to_2d(flat[ok], K[cam], R[cam], Tt[cam], D[cam])
        proj = proj.reshape(T, N_KP, 2)
        e = np.linalg.norm(proj - kp2d[:, cam], axis=-1)
        e[conf[:, cam] < conf_thresh] = np.nan
        err[:, cam] = e
    with np.errstate(all='ignore'):
        per_frame = np.nanmedian(err.reshape(T, -1), axis=1)
    return per_frame.astype(np.float32), raw3d, err


def conf_stats(keys_2d, thresh=0.5):
    conf = keys_2d[..., 2]                        # (T, 3, 23)
    mean_conf = np.nanmean(conf, axis=(1, 2))
    per_cam = np.nanmean(conf, axis=2)            # (T, 3)
    # a camera "sees" the animal if the median keypoint conf clears threshold
    n_cams = (np.nanmedian(conf, axis=2) >= thresh).sum(axis=1)
    return (mean_conf.astype(np.float32), per_cam.astype(np.float32),
            n_cams.astype(np.int8))


def bone_length_cv(triang, edges=None):
    """Per-frame mean |L - median(L)| / median(L) over skeleton edges.
    Truth-free rigidity violation: needs no reference at all."""
    e = np.asarray(edges if edges is not None else EDGES, dtype=int)
    a = triang[:, e[:, 0], :]
    b = triang[:, e[:, 1], :]
    L = np.linalg.norm(a - b, axis=-1)            # (T, n_edges)
    med = np.nanmedian(L, axis=0, keepdims=True)
    med = np.where(med == 0, np.nan, med)
    return np.nanmean(np.abs(L - med) / med, axis=1).astype(np.float32)


def jitter(triang_rot):
    """Mean |2nd difference| across keypoints, in egocentric coords (mm/frame^2)."""
    d2 = np.diff(triang_rot, n=2, axis=0)
    j = np.linalg.norm(d2, axis=-1).mean(axis=1)
    return np.concatenate([[np.nan], j, [np.nan]]).astype(np.float32)


def fov_margin(keys_2d, cal_dir, triang):
    """Signed px distance from each camera's image edge for the projected
    highest keypoint. Negative = outside the image. (T, 3)."""
    P, K, D, R, Tt = load_calib(cal_dir)
    T = triang.shape[0]
    top = np.nanargmax(triang[:, :, 2], axis=1)
    pts = triang[np.arange(T), top, :]
    ok = np.isfinite(pts).all(axis=1)
    out = np.full((T, N_CAMS), np.nan, dtype=np.float32)
    for cam in range(N_CAMS):
        w, h = 2 * K[cam][0, 2], 2 * K[cam][1, 2]
        pr = np.full((T, 2), np.nan)
        pr[ok] = project_3d_to_2d(pts[ok], K[cam], R[cam], Tt[cam], D[cam])
        m = np.minimum.reduce([pr[:, 0], pr[:, 1], w - pr[:, 0], h - pr[:, 1]])
        out[:, cam] = m
    return out


# --------------------------------------------------------------------------- #
# Frame-basis alignment (IMPORTANT)
# --------------------------------------------------------------------------- #
# sleap_keys_2D.npy is stored on the PROCESSED-frame basis: only frames that were
# processed in time are saved (n_processed rows, zero-padded out to n_acquired).
# triang_keys_3D.npy is on the CAM0 ACQUISITION-frame basis.
#
# The bridge is frame_mapping.csv's cam0_frame column:
#
#       triang_keys_3D[cam0_frame[k]]   <-->   sleap_keys_2D[k]
#
# Getting this wrong gives ~253 px reprojection and r=+0.23 between the 3D
# position and the animal's image position; getting it right gives 17.4 px and
# r=+0.988.

def reprojection_residual_aligned(keys_2d, triang_3d, frame_mapping_cam0, cal_dir,
                                  conf_thresh=0.5):
    """Reproject the STORED 3D pose into each camera and compare with the SLEAP
    2D detections, respecting the processed/acquisition frame-basis mismatch.

    Returns per-PROCESSED-frame arrays:
      resid  (Kp,)      median pixel residual over (cam, keypoint)
      conf   (Kp, 3)    mean keypoint confidence per camera
      ncam   (Kp,)      cameras whose median keypoint confidence >= conf_thresh
      fov    (Kp, 3)    signed px margin to the image edge for the highest keypoint
    """
    P, K, D, R, T = load_calib(cal_dir)
    fm = np.asarray(frame_mapping_cam0, dtype=int)
    Kp = min(len(fm), len(keys_2d))
    fm = fm[:Kp]
    ok = (fm >= 0) & (fm < len(triang_3d))
    X = np.full((Kp, N_KP, 3), np.nan)
    X[ok] = np.asarray(triang_3d, dtype=np.float64)[fm[ok]]

    det = keys_2d[:Kp, :, :, :2].astype(np.float64)
    conf = keys_2d[:Kp, :, :, 2].astype(np.float64)
    err = np.full((Kp, N_CAMS, N_KP), np.nan)
    fov = np.full((Kp, N_CAMS), np.nan, dtype=np.float32)

    flat = X.reshape(-1, 3)
    fin = np.isfinite(flat).all(axis=1)
    top = np.full(Kp, 0)
    with np.errstate(all='ignore'):
        zz = np.where(np.isfinite(X[:, :, 2]), X[:, :, 2], -np.inf)
        top = np.argmax(zz, axis=1)
    for cam in range(N_CAMS):
        pr = np.full((flat.shape[0], 2), np.nan)
        pr[fin] = project_3d_to_2d(flat[fin], K[cam], R[cam], T[cam], D[cam])
        pr = pr.reshape(Kp, N_KP, 2)
        e = np.linalg.norm(pr - det[:, cam], axis=-1)
        e[conf[:, cam] < conf_thresh] = np.nan
        err[:, cam] = e
        w, h = 2 * K[cam][0, 2], 2 * K[cam][1, 2]
        pt = pr[np.arange(Kp), top]
        fov[:, cam] = np.minimum.reduce([pt[:, 0], pt[:, 1], w - pt[:, 0], h - pt[:, 1]])
    with np.errstate(all='ignore'):
        resid = np.nanmedian(err.reshape(Kp, -1), axis=1).astype(np.float32)
        per_cam = np.nanmean(conf, axis=2).astype(np.float32)
        ncam = (np.nanmedian(conf, axis=2) >= conf_thresh).sum(axis=1).astype(np.int8)
    return resid, per_cam, ncam, fov


def scatter_to_acquisition(vals, frame_mapping_cam0, n_acquired):
    """Place per-processed-frame values onto the cam0 acquisition-frame grid."""
    fm = np.asarray(frame_mapping_cam0, dtype=int)
    k = min(len(fm), len(vals))
    shape = (n_acquired,) + np.shape(vals)[1:]
    out = np.full(shape, np.nan, dtype=np.float32)
    sel = (fm[:k] >= 0) & (fm[:k] < n_acquired)
    out[fm[:k][sel]] = np.asarray(vals)[:k][sel]
    return out
