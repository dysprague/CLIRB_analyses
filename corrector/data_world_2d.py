"""Extended loader for the 2D-input corrector experiment.

In addition to what `load_paired_world` returns (median-filtered SLEAP 3D and
DANNCE 3D on the SLEAP 20 Hz timeline), this loads:

  - sleap_2D_xy   (T, 3, 23, 2)  pixel coordinates per camera
  - sleap_2D_conf (T, 3, 23)     SLEAP detection confidence per camera
  - calib         dict with keys per camera:
        K       (3,3)   intrinsics
        r       (3,3)   rotation
        t       (3,)    translation (already negated as in data_io.load_calibration)
        dist    (4,)    [k1, k2, p1, p2]
        cal_date         string e.g. "2025_07_27"

`load_paired_world_with_2d` returns
    sl_world_smoothed (T, 23, 3) — median-filtered, native SLEAP world frame
    dn_world_smoothed (T, 23, 3) — median-filtered, on SLEAP timeline
    sl_2d_xy          (T, 3, 23, 2)
    sl_2d_conf        (T, 3, 23)
    sl_world_raw      (T, 23, 3) — UNSMOOTHED triangulated 3D (for residual sanity)
    calib             list of 3 dicts (one per SLEAP camera, in cam0..cam2 order)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import median_filter

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from data_io import (load_aligned_data, load_sleap_dannce_keys,  # noqa: E402
                      load_sleap_keys_2d, load_sleap_keys_3d, sleap_path)
from corrector.data_world import SLEAP_MEDFILT, DANNCE_MEDFILT  # noqa: E402


def _session_calibration_folder(rat: str, session: str) -> str:
    """The per-session SLEAP calibration is stored as
    `<session>/sleap/calibration/<YYYY_MM_DD>/`. There is exactly one
    date subfolder per session."""
    parent = os.path.join(sleap_path(rat, session), "calibration")
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"No sleap/calibration folder for {rat}/{session}")
    subs = [s for s in sorted(os.listdir(parent))
            if os.path.isdir(os.path.join(parent, s))]
    if not subs:
        raise FileNotFoundError(f"Empty calibration folder for {rat}/{session}")
    if len(subs) > 1:
        # Unexpected — take the latest by name (lexicographic == chronological
        # for YYYY_MM_DD).
        print(f"warning: multiple calibration subfolders for {rat}/{session}: "
              f"{subs} — using {subs[-1]}", flush=True)
    return os.path.join(parent, subs[-1]), subs[-1]


def load_session_calibration(rat: str, session: str):
    """Returns (list of 3 dicts, cal_date). Each dict has K (3,3), r (3,3),
    t (3,), dist (4,). The list is ordered cam0..cam2."""
    cal_dir, cal_date = _session_calibration_folder(rat, session)
    cam_files = sorted(f for f in os.listdir(cal_dir)
                       if f.startswith("hires_cam") and f.endswith("_params.mat"))
    out = []
    for fname in cam_files:
        params = sio.loadmat(os.path.join(cal_dir, fname), simplify_cells=True)
        K = np.transpose(params["K"]).astype(np.float64)
        r = np.transpose(params["r"]).astype(np.float64)
        t = -np.asarray(params["t"]).astype(np.float64).reshape(3)
        Rdist = np.asarray(params["RDistort"]).astype(np.float64).reshape(-1)
        Tdist = np.asarray(params["TDistort"]).astype(np.float64).reshape(-1)
        dist = np.array([Rdist[0], Rdist[1], Tdist[0], Tdist[1]],
                        dtype=np.float64)
        out.append({"K": K, "r": r, "t": t, "dist": dist,
                    "cal_date": cal_date, "cam_file": fname})
    return out, cal_date


def project_3d_to_2d_batch(points_3d: np.ndarray, cam: dict) -> np.ndarray:
    """Vectorized version of projection.project_3d_to_2d for a (T,23,3) batch.
    Returns (T,23,2) pixel coords. Uses the same model: cam coords -> normalize
    -> Brown-Conrady distortion -> intrinsics."""
    K, r, t, dist = cam["K"], cam["r"], cam["t"], cam["dist"]
    T = points_3d.shape[0]; N = points_3d.shape[1]
    pts = points_3d.reshape(-1, 3).astype(np.float64)
    pts_cam = pts @ r.T + t.reshape(1, 3)
    z = pts_cam[:, 2]
    # In this rig's calibration convention, "in front" of the camera is z<0,
    # not z>0. Avoid division by zero.
    z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x_norm = pts_cam[:, 0] / z_safe
    y_norm = pts_cam[:, 1] / z_safe
    k1, k2, p1, p2 = dist
    r2 = x_norm**2 + y_norm**2
    radial = 1 + k1 * r2 + k2 * r2**2
    x_d = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
    y_d = y_norm * radial + p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x_d + cx
    v = fy * y_d + cy
    out = np.stack([u, v], axis=1).reshape(T, N, 2).astype(np.float32)
    # Points on the wrong side of the camera (positive z in this convention)
    # produce nonsense — flag with NaN.
    wrong_side = (z >= 0).reshape(T, N)
    out[wrong_side] = np.nan
    return out


def reproject_all_cams(points_3d: np.ndarray, calib: list[dict]) -> np.ndarray:
    """points_3d (T,23,3) -> reprojected_2d (T, n_cam, 23, 2)."""
    T, N, _ = points_3d.shape
    out = np.empty((T, len(calib), N, 2), dtype=np.float32)
    for ci, cam in enumerate(calib):
        out[:, ci] = project_3d_to_2d_batch(points_3d, cam)
    return out


def load_paired_world_with_2d(rat: str, session: str):
    """Load SLEAP+DANNCE 3D (smoothed, on SLEAP timeline) plus 2D + calibration.

    Returns
    -------
    sl_world_smoothed (T, 23, 3) float32 — median-11 filtered, native SLEAP world
    dn_world_smoothed (T, 23, 3) float32 — median-25 filtered, on SLEAP timeline
    sl_2d_xy          (T, 3, 23, 2) float32
    sl_2d_conf        (T, 3, 23)    float32
    sl_world_raw      (T, 23, 3) float32 — raw triangulated 3D (no filter)
    calib             list of 3 cam dicts
    cal_date          string
    """
    keys = load_sleap_dannce_keys(rat, session)
    aligned = load_aligned_data(rat, session)
    sl = keys["sleap_keys_3D"].astype(np.float32)
    dn = keys["dannce_keys_3D"].astype(np.float32)
    if dn.ndim == 4:
        dn = dn.squeeze(axis=1).transpose(0, 2, 1)
    else:
        dn = dn.transpose(0, 2, 1)
    sl_smoothed = median_filter(sl, size=(SLEAP_MEDFILT, 1, 1))
    dn_smoothed = median_filter(dn, size=(DANNCE_MEDFILT, 1, 1))
    aidx = aligned["dannce_idx_for_sleap_cams"].astype(int).ravel()
    aidx = np.clip(aidx[: len(sl_smoothed)], 0, len(dn_smoothed) - 1)
    dn_on_sleap = dn_smoothed[aidx]

    k2d = load_sleap_keys_2d(rat, session).astype(np.float32)  # (T, 3, 23, 3)
    if k2d.shape[0] != len(sl_smoothed):
        T = min(k2d.shape[0], len(sl_smoothed))
        k2d = k2d[:T]; sl_smoothed = sl_smoothed[:T]; sl = sl[:T]
        dn_on_sleap = dn_on_sleap[:T]
    xy = k2d[..., :2]
    conf = k2d[..., 2]

    calib, cal_date = load_session_calibration(rat, session)
    if len(calib) != k2d.shape[1]:
        raise RuntimeError(f"{rat}/{session}: {len(calib)} calib cams but 2D "
                            f"has {k2d.shape[1]} cams")

    sl_raw = sl  # keep float32, unsmoothed triangulated 3D
    return (sl_smoothed.astype(np.float32),
            dn_on_sleap.astype(np.float32),
            xy.astype(np.float32),
            conf.astype(np.float32),
            sl_raw.astype(np.float32),
            calib,
            cal_date)
