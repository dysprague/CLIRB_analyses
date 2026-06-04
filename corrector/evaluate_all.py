"""
Unified evaluator: runs world-space keypoint+PC MSE eval AND F1 eval (both
xyz-PC matching and Group-O pairwise-pooled PCA matching) for a single
checkpoint on its held-out test sessions.

Saves a single JSON to corrector/results/<ckpt_stem>_all.json with three keys:
  per_session: list of per-session dicts (rat, session, kp_mse, pc_mse,
              f1_xyz, f1_groupO, with raw/procrustes/corrected variants for F1)
  summary:    dict of per-rat means
  meta:       checkpoint info, time taken

Usage:
    python -m corrector.evaluate_all --ckpt corrector/checkpoints/<...>.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from data_io import (load_aligned_data, load_template)
from exp_utils import (compute_alignment_multi_tol, compute_pairwise_distances,
                        estimate_temporal_offset, run_template_matching)
from skeleton import normalize_skeleton_batch

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.evaluate_world import RAT_TEMPLATE, project_to_template_pcs
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model
from corrector.data_world_2d_from_saved import load_session_2d

RESULTS_DIR = _THIS.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WIN = 30
TOLS = [100, 300, 500]
DEFAULT_BOUNDS = {"R1": 1.5, "R2": 1.0, "R3": 1.0}

N_PCS_SWEEP = [2, 3]
BOUNDS_SWEEP = [1.0, 1.25, 1.5]
MAX_OUT_SWEEP = [1, 2, 3]


def correct_world(model, x, device, batch=8192, ctx: int = 1,
                  vel_acc=False):
    """Apply (single-frame, temporal, or vel/acc) corrector. ctx=1 single frame.
    If vel_acc=True, the model expects (B, 3*23, 3) input — concat
    [pose, velocity, acceleration]."""
    out = np.empty_like(x, dtype=np.float32)
    if ctx <= 1 and not vel_acc:
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xt = torch.from_numpy(x[i:i + batch].astype(np.float32)).to(device)
                out[i:i + batch] = model(xt).cpu().numpy()
        return out

    if vel_acc:
        # Build [pose, vel, acc] features (causal differencing)
        x32 = x.astype(np.float32)
        vel = np.zeros_like(x32); vel[1:] = x32[1:] - x32[:-1]
        acc = np.zeros_like(x32); acc[2:] = vel[2:] - vel[1:-1]
        feats = np.stack([x32, vel, acc], axis=1)  # (T, 3, 23, 3)
        T = len(x)
        with torch.no_grad():
            for i in range(0, T, batch):
                xt = torch.from_numpy(feats[i:i + batch]).to(device)
                out[i:i + batch] = model(xt).cpu().numpy()
        return out

    # Temporal context
    T = len(x); pad = ctx - 1
    x_padded = np.concatenate([np.repeat(x[:1], pad, axis=0), x], axis=0)
    win_starts = np.arange(T)
    with torch.no_grad():
        for i in range(0, T, batch):
            idx = win_starts[i:i + batch]
            windows = np.stack([x_padded[s:s + ctx] for s in idx], axis=0)
            xt = torch.from_numpy(windows.astype(np.float32)).to(device)
            out[i:i + batch] = model(xt).cpu().numpy()
    return out


def correct_triangulation_refiner(model, rat, session, sl_aligned_dn, tx, device,
                                    batch: int = 8192):
    """2D-input corrector path. Returns a (T_sleap, 23, 3) array in DANNCE world,
    where T_sleap == len(sl_aligned_dn). Frames covered by the saved-2D
    processed-frame stream get the corrector output; uncovered frames fall back
    to sl_aligned_dn (Procrustes-only).

    sl_aligned_dn: (T_sleap, 23, 3) DANNCE-world frames produced by tx.apply(sl)
                   on the SLEAP timeline.
    tx           : the Procrustes transform already fit on (x_triang, y_dannce)
                   over the calibration window.

    The model expects (B, 23, 21) features [xyz_triang_dannce, per_cam_xy_norm,
    per_cam_conf, per_cam_reproj_resid, per_cam_visibility]. See
    corrector.train_2d_input.build_features.
    """
    from corrector.train_2d_input import (build_features, calib_to_torch,
                                            VIDEO_W, VIDEO_H, RESID_NORM_PX,
                                            OUTLIER_THRESHOLD_MM)

    # Inference-time guards (open issue from Phase H.6 — the trained model
    # emits wild corrections for test-time triangulation outliers such as
    # R1/2026_02_13_1 which went 513 -> 23,082 kp_mse). Two defenses:
    #   1) Per-sample outlier mask: if the raw triangulated SLEAP has any
    #      |coord| > OUTLIER_THRESHOLD_MM (1000 mm) or is non-finite, that
    #      frame is replaced with the Procrustes-only output, never touching
    #      the model.
    #   2) Residual clip: model output minus its identity (xyz_triang in
    #      DANNCE world) is clamped to +/- RESIDUAL_CLIP_MM before being
    #      written back. Arena is ~500 mm, so a 200 mm residual is the largest
    #      sensible correction.
    RESIDUAL_CLIP_MM = 200.0

    sd = load_session_2d(rat, session, smooth_dannce=True)
    P = len(sd.x_2d)
    if P == 0:
        return sl_aligned_dn.copy()  # nothing to correct

    # Procrustes-align the saved triangulated SLEAP 3D (SLEAP world -> DANNCE).
    x_triang_dannce = tx["apply"](sd.x_triang_3d).astype(np.float32)
    x_triang_sleap = sd.x_triang_3d.astype(np.float32)
    cam_frames = sd.cam_frames                              # (P, 3) int
    # Map each processed_frame -> SLEAP-timeline index via cam0_frame.
    sleap_idx = cam_frames[:, 0].astype(int)
    T_sleap = len(sl_aligned_dn)
    keep = (sleap_idx >= 0) & (sleap_idx < T_sleap)
    if not keep.any():
        return sl_aligned_dn.copy()

    # Per-sample outlier mask (training-side filter, applied at inference).
    flat_tri = x_triang_sleap.reshape(P, -1)
    flat_dn  = x_triang_dannce.reshape(P, -1)
    finite_in = np.isfinite(flat_tri).all(axis=1) & np.isfinite(flat_dn).all(axis=1)
    max_abs_tri = np.where(finite_in, np.abs(flat_tri).max(axis=1), np.inf)
    max_abs_dn  = np.where(finite_in, np.abs(flat_dn).max(axis=1),  np.inf)
    inlier = finite_in & (max_abs_tri < OUTLIER_THRESHOLD_MM) \
                       & (max_abs_dn  < OUTLIER_THRESHOLD_MM)
    n_outlier_frames = int((~inlier & keep).sum())

    # Calibration tensors on device (single session).
    cal_torch = calib_to_torch(sd.calibration, device)
    cal_per_sess = [cal_torch]

    out = sl_aligned_dn.copy()                              # fallback fill
    session_idx_const = torch.zeros((batch,), dtype=torch.long, device=device)
    n_clipped_kp = 0

    # Run in batches, write into `out` at the cam0_frame indices.
    with torch.no_grad():
        for i in range(0, P, batch):
            sl_b = slice(i, min(i + batch, P))
            b = sl_b.stop - sl_b.start
            x_d = torch.from_numpy(x_triang_dannce[sl_b]).to(device,
                                                                non_blocking=True)
            x_s = torch.from_numpy(x_triang_sleap[sl_b]).to(device,
                                                                non_blocking=True)
            x2 = torch.from_numpy(sd.x_2d[sl_b]).to(device, non_blocking=True)
            xc = torch.from_numpy(sd.x_conf[sl_b]).to(device, non_blocking=True)
            sidx = session_idx_const[:b]
            feat = build_features(x_d, x_s, x2, xc, cal_per_sess, sidx)
            pred = model(feat)                              # (b, 23, 3) DANNCE world
            # Clip the residual relative to the model's identity input.
            delta = pred - x_d
            delta = delta.clamp(min=-RESIDUAL_CLIP_MM, max=RESIDUAL_CLIP_MM)
            n_clipped_kp += int((delta.abs() == RESIDUAL_CLIP_MM).any(dim=-1).sum().item())
            pred_clipped = (x_d + delta).cpu().numpy()
            # Scatter into out at the SLEAP-timeline index for each processed frame.
            # Only write inlier rows; outlier rows keep the Procrustes-only fallback.
            local_sleap_idx = sleap_idx[sl_b]
            local_inlier_keep = keep[sl_b] & inlier[sl_b]
            target_rows = local_sleap_idx[local_inlier_keep]
            out[target_rows] = pred_clipped[local_inlier_keep]
    if n_outlier_frames > 0 or n_clipped_kp > 0:
        print(f"    [refiner guard] {rat}/{session}: "
              f"outlier_frames={n_outlier_frames}/{int(keep.sum())}, "
              f"clipped_kp={n_clipped_kp}", flush=True)
    return out


def correct_temporal_triangulation_refiner(model, rat, session, sl_aligned_dn,
                                              tx, device, ctx, batch: int = 4096):
    """Temporal 2D-input corrector. Mirrors correct_triangulation_refiner but
    builds a T_ctx-frame causal window per processed frame.

    For each processed frame p we use frames [p-ctx+1 .. p] (clamped at the
    start to repeat the earliest available frame). The model outputs a
    corrected pose for frame p; the inference outlier guard + residual clip
    applies on the LAST frame (same as the single-frame path).
    """
    from corrector.train_2d_input import (build_features, calib_to_torch,
                                            OUTLIER_THRESHOLD_MM, N_CAM, N_KP)
    from corrector.train_2d_temporal import (build_features_temporal,
                                              _apply_inference_guard)

    sd = load_session_2d(rat, session, smooth_dannce=True)
    P = len(sd.x_2d)
    if P == 0:
        return sl_aligned_dn.copy()

    x_triang_dannce = tx["apply"](sd.x_triang_3d).astype(np.float32)
    x_triang_sleap = sd.x_triang_3d.astype(np.float32)
    cam_frames = sd.cam_frames
    sleap_idx = cam_frames[:, 0].astype(int)
    T_sleap = len(sl_aligned_dn)
    keep = (sleap_idx >= 0) & (sleap_idx < T_sleap)
    if not keep.any():
        return sl_aligned_dn.copy()

    # Build a (P, ctx) gather index that clamps at the start of the session.
    # ctx_idx[p, t] = max(0, p - (ctx - 1 - t)).
    arange_p = np.arange(P)[:, None]
    arange_t = np.arange(ctx)[None, :]
    ctx_idx = np.maximum(arange_p - (ctx - 1 - arange_t), 0)  # (P, ctx)

    # Pre-build the windowed arrays once.
    x_d_win = x_triang_dannce[ctx_idx]                 # (P, ctx, 23, 3)
    x_s_win = x_triang_sleap[ctx_idx]                  # (P, ctx, 23, 3)
    x2_win = sd.x_2d[ctx_idx]                          # (P, ctx, 3, 23, 2)
    xc_win = sd.x_conf[ctx_idx]                        # (P, ctx, 3, 23)

    cal_torch = calib_to_torch(sd.calibration, device)
    cal_per_sess = [cal_torch]

    out = sl_aligned_dn.copy()
    session_idx_const = torch.zeros((batch,), dtype=torch.long, device=device)
    n_outlier_frames = 0
    with torch.no_grad():
        for i in range(0, P, batch):
            sl_b = slice(i, min(i + batch, P))
            b = sl_b.stop - sl_b.start
            x_d = torch.from_numpy(x_d_win[sl_b]).to(device, non_blocking=True)
            x_s = torch.from_numpy(x_s_win[sl_b]).to(device, non_blocking=True)
            x2 = torch.from_numpy(x2_win[sl_b]).to(device, non_blocking=True)
            xc = torch.from_numpy(xc_win[sl_b]).to(device, non_blocking=True)
            sidx = session_idx_const[:b]
            feat = build_features_temporal(x_d, x_s, x2, xc, cal_per_sess, sidx)
            pred = model(feat)                              # (b, 23, 3)
            guarded = _apply_inference_guard(pred,
                                              x_d[:, -1, :, :],
                                              x_s[:, -1, :, :])
            pred_np = guarded.cpu().numpy()
            local_sleap_idx = sleap_idx[sl_b]
            local_keep = keep[sl_b]
            target_rows = local_sleap_idx[local_keep]
            out[target_rows] = pred_np[local_keep]
    return out


def correct_temporal_mlp_2d(model, rat, session, sl_aligned_dn, device,
                              ctx, batch: int = 4096):
    """Inference for TemporalMLPWith2D. Builds per-SLEAP-frame 2D bundles by
    scattering from the processed-frame stream via cam0_frame, then applies the
    model over T_ctx windows.

    sl_aligned_dn: (T_sleap, 23, 3) — Procrustes-aligned SLEAP 3D in DANNCE
                   world space, on the SLEAP timeline (same as the 3D-input
                   correctors).

    Frames that have no processed-frame mapping (no 2D info) fall back to
    sl_aligned_dn (Procrustes-only). Frames with an outlier in the 3D window
    also fall back via the residual guard.
    """
    from corrector.train_temporal_mlp_2d import (VIDEO_W, VIDEO_H, N_CAM, N_KP,
                                                   OUTLIER_THRESHOLD_MM,
                                                   RESIDUAL_CLIP_MM)
    T_sleap = len(sl_aligned_dn)
    sd = load_session_2d(rat, session, smooth_dannce=True)
    if len(sd.x_2d) == 0:
        return sl_aligned_dn.copy()

    # Build per-SLEAP-frame 2D arrays by scattering. Frames without 2D info
    # keep zeros and are flagged via has_2d=False so we fall back to identity.
    x_2d_per_t = np.zeros((T_sleap, N_CAM, N_KP, 2), dtype=np.float32)
    x_conf_per_t = np.zeros((T_sleap, N_CAM, N_KP), dtype=np.float32)
    vis_per_t = np.zeros((T_sleap, N_CAM, N_KP), dtype=np.float32)
    has_2d = np.zeros(T_sleap, dtype=bool)
    cam0 = sd.cam_frames[:, 0].astype(int)
    in_range = (cam0 >= 0) & (cam0 < T_sleap)
    rows = cam0[in_range]
    fr_2d = sd.x_2d[in_range]
    fr_conf = sd.x_conf[in_range]
    fr_finite = np.isfinite(fr_2d).all(axis=-1)
    fr_vis = (fr_conf > 0) & fr_finite
    fr_2d_safe = np.where(fr_vis[..., None], fr_2d, 0.0).astype(np.float32)
    fr_conf_safe = np.where(fr_vis, fr_conf, 0.0).astype(np.float32)
    x_2d_per_t[rows] = fr_2d_safe
    x_conf_per_t[rows] = fr_conf_safe
    vis_per_t[rows] = fr_vis.astype(np.float32)
    has_2d[rows] = True
    # Normalize 2D to [-1, 1].
    scale = np.array([VIDEO_W, VIDEO_H], dtype=np.float32).reshape(1, 1, 1, 2)
    x_2d_per_t = (x_2d_per_t / scale - 0.5) * 2.0

    # Build a (T_sleap, ctx) gather index for the 3D window. Clamp at start.
    arange_t = np.arange(T_sleap)[:, None]
    arange_w = np.arange(ctx)[None, :]
    win_idx = np.maximum(arange_t - (ctx - 1 - arange_w), 0)  # (T_sleap, ctx)
    pose_win = sl_aligned_dn[win_idx]                        # (T_sleap, ctx, 23, 3)

    out = sl_aligned_dn.copy()
    with torch.no_grad():
        for i in range(0, T_sleap, batch):
            sl_b = slice(i, min(i + batch, T_sleap))
            b = sl_b.stop - sl_b.start
            x_pose = torch.from_numpy(pose_win[sl_b]).to(device,
                                                          non_blocking=True)
            x_2d = torch.from_numpy(x_2d_per_t[sl_b]).to(device,
                                                          non_blocking=True)
            x_conf = torch.from_numpy(x_conf_per_t[sl_b]).to(device,
                                                              non_blocking=True)
            x_vis = torch.from_numpy(vis_per_t[sl_b]).to(device,
                                                          non_blocking=True)
            pred = model(x_pose, x_2d, x_conf, x_vis)         # (b, 23, 3)
            x_pose_last = x_pose[:, -1, :, :]
            delta = (pred - x_pose_last).clamp(min=-RESIDUAL_CLIP_MM,
                                                 max=RESIDUAL_CLIP_MM)
            pred_clipped = (x_pose_last + delta).cpu().numpy()

            # Per-sample outlier check + has_2d gate for fallback.
            local_last = pose_win[sl_b, -1]
            flat = local_last.reshape(b, -1)
            finite = np.isfinite(flat).all(axis=1)
            max_abs = np.where(finite, np.abs(flat).max(axis=1), np.inf)
            inlier = finite & (max_abs < OUTLIER_THRESHOLD_MM)
            local_has_2d = has_2d[sl_b]
            keep_local = inlier & local_has_2d
            sel = np.where(keep_local)[0]
            if len(sel):
                out_idx = i + sel
                out[out_idx] = pred_clipped[sel]
    return out


def correct_temporal_mlp_2d_reproj(model, rat, session, sl_aligned_dn, device,
                                     ctx, batch: int = 4096,
                                     smooth_size: int = 11,
                                     smooth_causal: bool = False):
    """Inference for TemporalMLPWith2DReproj. Builds per-SLEAP-frame 2D + conf
    + vis + reprojection-residual bundles by scattering from the processed-
    frame stream, then runs the temporal-windowed model.

    Frames without a processed-frame mapping (no 2D info) keep the Procrustes-
    only sl_aligned_dn value (original triangulated 3D). Outlier-input frames
    also keep Procrustes-only.

    smooth_size: median-filter size (in frames) applied to the final output
    along the time dimension. ~4.4% of SLEAP frames have no 2D mapping and
    fall back to Procrustes-only; the corrector and Procrustes-only outputs
    live in slightly different distributions, producing visible discontinuities
    at every gap frame. Median-11 smooths across the transitions and matches
    the median filter `load_paired_world` applies to the input SLEAP. Set
    smooth_size <= 1 to disable.

    smooth_causal: when True, the median uses only the current frame and the
    (smooth_size - 1) preceding frames — no future leakage. Matches what an
    online pipeline with a circular buffer of length smooth_size would emit.
    """
    from corrector.train_temporal_mlp_2d_reproj import (VIDEO_W, VIDEO_H,
                                                          N_CAM, N_KP,
                                                          RESID_NORM_PX,
                                                          OUTLIER_THRESHOLD_MM,
                                                          RESIDUAL_CLIP_MM)
    from corrector.data_world_2d import reproject_all_cams

    T_sleap = len(sl_aligned_dn)
    sd = load_session_2d(rat, session, smooth_dannce=True)
    if len(sd.x_2d) == 0:
        return sl_aligned_dn.copy()

    # Per-processed-frame reprojection of the saved un-smoothed triangulated 3D.
    reproj_pf = reproject_all_cams(sd.x_triang_3d, sd.calibration)
    detected_pf = sd.x_2d
    conf_pf = sd.x_conf
    vis_pf = ((conf_pf > 0)
              & np.isfinite(detected_pf).all(axis=-1)
              & np.isfinite(reproj_pf).all(axis=-1))
    resid_pf_safe = np.where(vis_pf[..., None],
                              (detected_pf - reproj_pf) / RESID_NORM_PX,
                              0.0).astype(np.float32)
    detected_safe = np.where(vis_pf[..., None], detected_pf, 0.0).astype(np.float32)
    conf_safe = np.where(vis_pf, conf_pf, 0.0).astype(np.float32)

    x_2d_per_t = np.zeros((T_sleap, N_CAM, N_KP, 2), dtype=np.float32)
    x_conf_per_t = np.zeros((T_sleap, N_CAM, N_KP), dtype=np.float32)
    vis_per_t = np.zeros((T_sleap, N_CAM, N_KP), dtype=np.float32)
    reproj_per_t = np.zeros((T_sleap, N_CAM, N_KP, 2), dtype=np.float32)
    has_2d = np.zeros(T_sleap, dtype=bool)
    cam0 = sd.cam_frames[:, 0].astype(int)
    in_range = (cam0 >= 0) & (cam0 < T_sleap)
    rows = cam0[in_range]
    x_2d_per_t[rows] = detected_safe[in_range]
    x_conf_per_t[rows] = conf_safe[in_range]
    vis_per_t[rows] = vis_pf[in_range].astype(np.float32)
    reproj_per_t[rows] = resid_pf_safe[in_range]
    has_2d[rows] = True

    scale = np.array([VIDEO_W, VIDEO_H], dtype=np.float32).reshape(1, 1, 1, 2)
    x_2d_per_t = (x_2d_per_t / scale - 0.5) * 2.0

    arange_t = np.arange(T_sleap)[:, None]
    arange_w = np.arange(ctx)[None, :]
    win_idx = np.maximum(arange_t - (ctx - 1 - arange_w), 0)
    pose_win = sl_aligned_dn[win_idx]                        # (T_sleap, ctx, 23, 3)

    out = sl_aligned_dn.copy()
    with torch.no_grad():
        for i in range(0, T_sleap, batch):
            sl_b = slice(i, min(i + batch, T_sleap))
            b = sl_b.stop - sl_b.start
            x_pose = torch.from_numpy(pose_win[sl_b]).to(device, non_blocking=True)
            x_2d = torch.from_numpy(x_2d_per_t[sl_b]).to(device, non_blocking=True)
            x_conf = torch.from_numpy(x_conf_per_t[sl_b]).to(device, non_blocking=True)
            x_vis = torch.from_numpy(vis_per_t[sl_b]).to(device, non_blocking=True)
            x_reproj = torch.from_numpy(reproj_per_t[sl_b]).to(device, non_blocking=True)
            pred = model(x_pose, x_2d, x_conf, x_vis, x_reproj)
            x_pose_last = x_pose[:, -1, :, :]
            delta = (pred - x_pose_last).clamp(min=-RESIDUAL_CLIP_MM,
                                                 max=RESIDUAL_CLIP_MM)
            pred_clipped = (x_pose_last + delta).cpu().numpy()

            local_last = pose_win[sl_b, -1]
            flat = local_last.reshape(b, -1)
            finite = np.isfinite(flat).all(axis=1)
            max_abs = np.where(finite, np.abs(flat).max(axis=1), np.inf)
            inlier = finite & (max_abs < OUTLIER_THRESHOLD_MM)
            keep_local = inlier & has_2d[sl_b]
            sel = np.where(keep_local)[0]
            if len(sel):
                out[i + sel] = pred_clipped[sel]
    if smooth_size and smooth_size > 1:
        if smooth_causal:
            # Causal median over window [t - (smooth_size - 1), t]. Edge frames
            # at the very start of the session don't have smooth_size-1
            # predecessors yet, so we left-pad with the first frame (matches
            # what a circular buffer initialized at t=0 would do).
            pad = smooth_size - 1
            padded = np.concatenate(
                [np.repeat(out[:1], pad, axis=0), out], axis=0
            )
            T = out.shape[0]
            window_idx = np.arange(T)[:, None] + np.arange(smooth_size)[None, :]
            windowed = padded[window_idx]   # (T, smooth_size, 23, 3)
            out = np.median(windowed, axis=1).astype(np.float32)
        else:
            from scipy.ndimage import median_filter
            out = median_filter(out, size=(smooth_size, 1, 1)).astype(np.float32)
    return out


def compute_metrics_session(rat, session, sleap_corrected, dn, st,
                              tmpl_data, eval_start_frame=0,
                              run_groupO=True):
    """Given final corrected SLEAP (in DANNCE world space), DANNCE, time axis,
    and the rat's xyz-PCA template data, compute:
      kp_mse_align (Procrustes only) and kp_mse_corr
      pc_mse_align/corr (xyz template space)
      F1@300 raw, procrustes, corrected for both xyz and Group-O matching
    """
    sl_aligned = sleap_corrected  # actually procrustes-aligned input
    sl_corr = sleap_corrected     # passed in already corrector-output
    raise NotImplementedError("see compute_full")


def compute_full(rat, session, model, model_ctx, model_vel_acc,
                 device, max_residual=60.0, calibration_minutes=5.0,
                 calibration_n_sample=1000, tmpl_cache=None,
                 run_groupO=True, model_name: str = "mlp",
                 smooth_size: int = 11, smooth_causal: bool = False):
    """Compute all metrics for one session. Returns a dict or {error: ...}.

    model_name is used to dispatch between the 3D-input path (sl, dn from
    load_paired_world) and the 2D-input path (triangulation_refiner: Procrustes
    fit on saved triangulated SLEAP + DANNCE, then per-processed-frame
    correction scattered back to the SLEAP timeline).
    """
    if tmpl_cache is not None and rat in tmpl_cache:
        tmpl_data = tmpl_cache[rat]
    else:
        tmpl_data = dict(load_template(rat, RAT_TEMPLATE[rat]))
    try:
        sl, dn = load_paired_world(rat, session)
    except Exception as e:
        return {"rat": rat, "session": session, "error": f"load: {e}"}
    if len(sl) < 1000:
        return {"rat": rat, "session": session, "error": "too few frames"}

    # ---- Procrustes ----
    # 3D-input path uses (sl, dn) over the calibration window.
    # 2D-input path uses (saved triangulated SLEAP, DANNCE) over the same
    # calibration window — this matches the training-time alignment exactly.
    if model_name in ("triangulation_refiner", "temporal_triangulation_refiner"):
        try:
            sd_for_tx = load_session_2d(rat, session, smooth_dannce=True)
        except Exception as e:
            return {"rat": rat, "session": session, "error": f"load_2d: {e}"}
        if len(sd_for_tx.x_triang_3d) < 1000:
            return {"rat": rat, "session": session, "error": "too few processed frames"}
        idx = calibration_indices(len(sd_for_tx.x_triang_3d),
                                   calibration_minutes, SLEAP_HZ,
                                   calibration_n_sample, seed=0)
        if len(idx) < 100:
            return {"rat": rat, "session": session, "error": "no cal window"}
        tx = fit_procrustes(sd_for_tx.x_triang_3d[idx],
                            sd_for_tx.y_dannce_3d[idx], try_z_flip=True)
    else:
        idx = calibration_indices(len(sl), calibration_minutes, SLEAP_HZ,
                                   calibration_n_sample, seed=0)
        if len(idx) < 100:
            return {"rat": rat, "session": session, "error": "no cal window"}
        tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
    if tx["residual"] > max_residual:
        return {"rat": rat, "session": session,
                "error": f"residual {tx['residual']:.1f} > {max_residual}"}

    # Apply Procrustes + corrector in DANNCE space
    sl_aligned_dn = tx["apply"](sl).astype(np.float32)
    if model_name == "triangulation_refiner":
        sl_corrected_dn = correct_triangulation_refiner(
            model, rat, session, sl_aligned_dn, tx, device)
    elif model_name == "temporal_triangulation_refiner":
        sl_corrected_dn = correct_temporal_triangulation_refiner(
            model, rat, session, sl_aligned_dn, tx, device, ctx=model_ctx)
    elif model_name == "temporal_mlp_2d":
        sl_corrected_dn = correct_temporal_mlp_2d(
            model, rat, session, sl_aligned_dn, device, ctx=model_ctx)
    elif model_name == "temporal_mlp_2d_reproj":
        sl_corrected_dn = correct_temporal_mlp_2d_reproj(
            model, rat, session, sl_aligned_dn, device, ctx=model_ctx,
            smooth_size=smooth_size, smooth_causal=smooth_causal)
    else:
        sl_corrected_dn = correct_world(model, sl_aligned_dn, device,
                                         ctx=model_ctx, vel_acc=model_vel_acc)
    dn_w = dn.astype(np.float32)

    # Eval on the post-calibration window so we don't reward fitting on Procrustes data
    eval_start = int(calibration_minutes * 60 * SLEAP_HZ)
    if eval_start >= len(sl) - 1000:
        eval_start = 0
    a_eval = sl_aligned_dn[eval_start:]
    c_eval = sl_corrected_dn[eval_start:]
    d_eval = dn_w[eval_start:]

    # ---- Keypoint MSE ----
    err_a = (a_eval - d_eval) ** 2
    err_c = (c_eval - d_eval) ** 2
    kp_mse_align = float(err_a.sum(axis=2).mean())
    kp_mse_corr = float(err_c.sum(axis=2).mean())
    per_kp_align = err_a.sum(axis=2).mean(axis=0).tolist()
    per_kp_corr = err_c.sum(axis=2).mean(axis=0).tolist()

    # ---- xyz PC MSE (project to template PC space; bring everything back to SLEAP space first) ----
    sl_aligned_sleap = tx["apply_inverse"](sl_aligned_dn).astype(np.float32)
    sl_corr_sleap = tx["apply_inverse"](sl_corrected_dn).astype(np.float32)

    # The rat's template lives in z-FLIPPED SLEAP egocentric coords
    def for_template(arr):
        out = arr.copy()
        out[:, :, 2] = -out[:, :, 2]
        return out

    sl_a_t = for_template(sl_aligned_sleap[eval_start:])
    sl_c_t = for_template(sl_corr_sleap[eval_start:])
    dn_t = for_template(tx["apply_inverse"](dn_w[eval_start:]))

    pcu = tmpl_data["pcs_to_use"].ravel().astype(int)
    pc_a = project_to_template_pcs(sl_a_t, tmpl_data, pcu)
    pc_c = project_to_template_pcs(sl_c_t, tmpl_data, pcu)
    pc_d = project_to_template_pcs(dn_t, tmpl_data, pcu)

    pc_mse_align = ((pc_a - pc_d) ** 2).mean(axis=0).tolist()
    pc_mse_corr = ((pc_c - pc_d) ** 2).mean(axis=0).tolist()

    # ---- F1 in xyz-PC space ----
    aligned = load_aligned_data(rat, session)
    st_full = np.array(aligned["sleap_times_ms"]).ravel()
    st = st_full[eval_start:eval_start + len(d_eval)]
    n_components = len(pcu)
    feature_stds = tmpl_data["feature_stds"]
    template_pc = tmpl_data["template"][:, pcu]
    bounds_scalar = DEFAULT_BOUNDS[rat]
    bounds = np.tile(feature_stds[pcu] * bounds_scalar, (WIN, 1))

    out = {"rat": rat, "session": session,
           "n_frames": int(len(d_eval)),
           "procrustes_residual": float(tx["residual"]),
           "kp_mse_align": kp_mse_align, "kp_mse_corr": kp_mse_corr,
           "per_kp_align": per_kp_align, "per_kp_corr": per_kp_corr,
           "pc_mse_align": pc_mse_align, "pc_mse_corr": pc_mse_corr}

    gt_m_xyz = run_template_matching(pc_d, template_pc, bounds, max_outside=3,
                                      refractory_frames=WIN)
    if not gt_m_xyz:
        return out  # No GT — record what we have but skip F1

    for label, pcs_arr in [("raw_xyz", project_to_template_pcs(
                              for_template(sl[eval_start:]), tmpl_data, pcu)),
                            ("procrustes_xyz", pc_a),
                            ("corrected_xyz", pc_c)]:
        sl_m = run_template_matching(pcs_arr, template_pc, bounds,
                                      max_outside=3, refractory_frames=WIN)
        if len(sl_m) >= 2 and len(gt_m_xyz) >= 2:
            offset_ms = estimate_temporal_offset(sl_m, gt_m_xyz, st, st)
        else:
            offset_ms = 0.0
        al = compute_alignment_multi_tol(sl_m, gt_m_xyz, TOLS, st, st, offset_ms)
        for tol in TOLS:
            r = al[f"tol_{tol}ms"]
            out[f"{label}_f1_{tol}"] = float(r["f1"])
            out[f"{label}_recall_{tol}"] = float(r["recall"])
            out[f"{label}_precision_{tol}"] = float(r["precision"])
        out[f"{label}_offset_ms"] = float(offset_ms)
        out[f"{label}_n_sleap"] = int(len(sl_m))
    out["n_gt_xyz"] = int(len(gt_m_xyz))

    # ---- F1 in Group-O pairwise pooled-PCA space ----
    # Sweep n_pcs × bounds_scalar × max_outside, take best per variant on F1@300
    def group_o_score(sl_arr, dn_arr, label):
        sl_z = for_template(sl_arr)
        dn_z = for_template(dn_arr)
        sl_rot, _, _ = normalize_skeleton_batch(sl_z.astype(np.float64))
        dn_rot, _, _ = normalize_skeleton_batch(dn_z.astype(np.float64))
        sl_pw = compute_pairwise_distances(sl_rot)
        dn_pw = compute_pairwise_distances(dn_rot)
        pooled = np.vstack([sl_pw, dn_pw])
        pw_mean = pooled.mean(axis=0)
        _, _, Vt = np.linalg.svd(pooled - pw_mean, full_matrices=False)
        comps_full = Vt

        # Anchor template via xyz PC space (find best-DANNCE-window match)
        sl_orig_pc = ((sl_rot.reshape(len(sl_rot), -1) - tmpl_data["feature_means"])
                      @ tmpl_data["pc_weights"].T)[:, pcu]
        dn_orig_pc = ((dn_rot.reshape(len(dn_rot), -1) - tmpl_data["feature_means"])
                      @ tmpl_data["pc_weights"].T)[:, pcu]
        gt_anchor = run_template_matching(dn_orig_pc, template_pc, bounds,
                                           max_outside=3, refractory_frames=WIN)
        if not gt_anchor:
            return None
        best_err, best_f = np.inf, None
        for f in gt_anchor:
            chunk = dn_orig_pc[f - WIN + 1:f + 1]
            if len(chunk) < WIN: continue
            e = np.mean((chunk - template_pc) ** 2)
            if e < best_err:
                best_err, best_f = e, f
        if best_f is None: return None
        win_pw = dn_pw[best_f - WIN + 1:best_f + 1]
        win_pc_full = (win_pw - pw_mean) @ comps_full.T

        best = None
        for n_pcs in N_PCS_SWEEP:
            comps = comps_full[:n_pcs]
            sl_pc = (sl_pw - pw_mean) @ comps.T
            dn_pc = (dn_pw - pw_mean) @ comps.T
            tmpl_o = win_pc_full[:, :n_pcs]
            feat_stds = np.std(np.vstack([sl_pc, dn_pc]), axis=0)
            for scalar in BOUNDS_SWEEP:
                bnds = np.tile(feat_stds * scalar, (WIN, 1))
                gt_m = run_template_matching(dn_pc, tmpl_o, bnds,
                                              max_outside=3, refractory_frames=WIN)
                if not gt_m: continue
                sl_init = run_template_matching(sl_pc, tmpl_o, bnds,
                                                 max_outside=3, refractory_frames=WIN)
                if len(sl_init) >= 2 and len(gt_m) >= 2:
                    offset_ms = estimate_temporal_offset(sl_init, gt_m, st, st)
                else:
                    offset_ms = 0.0
                for mo in MAX_OUT_SWEEP:
                    sl_m = run_template_matching(sl_pc, tmpl_o, bnds,
                                                  max_outside=mo,
                                                  refractory_frames=WIN)
                    al = compute_alignment_multi_tol(sl_m, gt_m, TOLS, st, st,
                                                      offset_ms)
                    f1 = float(al["tol_300ms"]["f1"])
                    if best is None or f1 > best["f1_300"]:
                        best = {
                            "n_pcs": n_pcs, "bounds_scalar": scalar,
                            "max_outside": mo, "offset_ms": float(offset_ms),
                            "f1_300": f1,
                            "recall_300": float(al["tol_300ms"]["recall"]),
                            "precision_300": float(al["tol_300ms"]["precision"]),
                        }
        return best

    if run_groupO:
        sl_ev = sl[eval_start:].astype(np.float32)
        dn_ev = tx["apply_inverse"](dn_w[eval_start:]).astype(np.float32)
        sl_a_ev = sl_aligned_sleap[eval_start:]
        sl_c_ev = sl_corr_sleap[eval_start:]
        for label, sl_v in [("raw_groupO", sl_ev),
                             ("procrustes_groupO", sl_a_ev),
                             ("corrected_groupO", sl_c_ev)]:
            best = group_o_score(sl_v, dn_ev, label)
            if best is None:
                continue
            for k, v in best.items():
                out[f"{label}_{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_residual", type=float, default=60.0)
    ap.add_argument("--no_groupO", action="store_true",
                    help="skip Group O (pairwise) F1 to save time")
    ap.add_argument("--rats", nargs="*", default=None,
                    help="if set, evaluate on these rats (all sessions); "
                         "otherwise use the checkpoint's test split")
    ap.add_argument("--template_suffix", type=str, default=None,
                    help="optional suffix to swap onto the template filename. "
                         "e.g. --template_suffix _rebuild loads "
                         "<rat>_template_1_rebuild.npz instead of "
                         "<rat>_template_1.npz. Use to A/B the H.1 rebuild.")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="optional explicit list of sessions in the form RAT/SESSION. "
                         "If set, overrides --rats and the test split.")
    ap.add_argument("--out_tag", type=str, default=None,
                    help="suffix appended to the results filename "
                         "(<ckpt_stem><out_tag>_all.json).")
    ap.add_argument("--smooth_size", type=int, default=11,
                    help="median-filter size applied inside the "
                         "temporal_mlp_2d_reproj corrector. Set <=1 to disable.")
    ap.add_argument("--smooth_causal", action="store_true",
                    help="use a causal median (only frames preceding the "
                         "current frame) instead of the default symmetric "
                         "scipy median_filter. Models what an online pipeline "
                         "with a circular buffer of length smooth_size would emit.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    name = ck["model_name"]
    if name == "gnn":
        model_kwargs = dict(hidden=ck.get("hidden", 64),
                            n_layers=ck.get("n_layers", 3))
    elif name == "perrat_head":
        model_kwargs = dict(base_ckpt=ck.get("base_ckpt"),
                            hidden=ck.get("hidden", 64),
                            n_hidden_layers=ck.get("n_hidden_layers", 2))
    elif name == "triangulation_refiner":
        model_kwargs = dict(hidden=ck.get("hidden", 128),
                            n_per_kp_layers=ck.get("n_per_kp_layers", 3),
                            global_dim=ck.get("global_dim", 64),
                            dropout=ck.get("dropout", 0.0))
    elif name == "temporal_triangulation_refiner":
        model_kwargs = dict(ctx=ck.get("ctx", 5),
                            hidden=ck.get("hidden", 128),
                            n_per_kp_layers=ck.get("n_per_kp_layers", 3),
                            global_dim=ck.get("global_dim", 64),
                            dropout=ck.get("dropout", 0.0))
    elif name == "temporal_mlp_2d":
        model_kwargs = dict(ctx=ck.get("ctx", 5),
                            hidden=ck.get("hidden", 128),
                            n_hidden_layers=ck.get("n_hidden_layers", 2),
                            dropout=ck.get("dropout", 0.0))
    elif name == "temporal_mlp_2d_reproj":
        model_kwargs = dict(ctx=ck.get("ctx", 5),
                            hidden=ck.get("hidden", 128),
                            n_hidden_layers=ck.get("n_hidden_layers", 2),
                            dropout=ck.get("dropout", 0.0))
    else:
        model_kwargs = dict(hidden=ck.get("hidden", 128),
                            n_hidden_layers=ck.get("n_hidden_layers", 2))
        if name == "temporal_mlp":
            model_kwargs["ctx"] = ck.get("ctx", 5)
    model = build_model(name, **model_kwargs)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    # Inference-side context handling
    if name == "temporal_mlp":
        model_ctx = ck.get("ctx", 5)
        model_vel_acc = False
    elif name == "perrat_head":
        # Inherit from base
        base_kind = getattr(model, "_base_kind", None)
        model_ctx = getattr(model, "_base_ctx", 1) if base_kind == "temporal_mlp" else 1
        model_vel_acc = (base_kind == "velacc_mlp")
    elif name == "velacc_mlp":
        model_ctx = 1
        model_vel_acc = True
    elif name == "triangulation_refiner":
        # ctx/vel_acc unused for the 2D path — the model gets its own feature
        # bundle inside correct_triangulation_refiner.
        model_ctx = 1
        model_vel_acc = False
    elif name == "temporal_triangulation_refiner":
        model_ctx = ck.get("ctx", 5)
        model_vel_acc = False
    elif name == "temporal_mlp_2d":
        model_ctx = ck.get("ctx", 5)
        model_vel_acc = False
    elif name == "temporal_mlp_2d_reproj":
        model_ctx = ck.get("ctx", 5)
        model_vel_acc = False
    else:
        model_ctx = 1
        model_vel_acc = False
    print(f"loaded {args.ckpt}: {ck['model_name']} "
          f"(params={sum(p.numel() for p in model.parameters()):,})", flush=True)

    if args.sessions:
        all_sessions = []
        for spec in args.sessions:
            rat, s = spec.split("/", 1)
            all_sessions.append((rat, s))
    elif args.rats:
        from data_io import get_sessions
        all_sessions = []
        for rat in args.rats:
            for s in sorted(get_sessions(rat=rat)["session"].tolist()):
                all_sessions.append((rat, s))
    else:
        all_sessions = []
        for rat, sessions in ck["splits"]["test"].items():
            for s in sessions:
                all_sessions.append((rat, s))

    # Pre-load template data per rat. The H.1 rebuild lives at
    # <rat>_template_1_rebuild.npz; pass --template_suffix _rebuild to A/B.
    def _resolve_template_name(rat):
        base = RAT_TEMPLATE[rat]
        if not args.template_suffix:
            return base
        stem, ext = base.rsplit(".", 1)
        return f"{stem}{args.template_suffix}.{ext}"
    tmpl_cache = {}
    for rat in set(r for r, _ in all_sessions):
        tname = _resolve_template_name(rat)
        tmpl_cache[rat] = dict(load_template(rat, tname))
    if args.template_suffix:
        print(f"using template suffix {args.template_suffix!r}", flush=True)

    rows = []
    for rat, s in all_sessions:
        t0 = time.time()
        out = compute_full(rat, s, model, model_ctx, model_vel_acc, device,
                            max_residual=args.max_residual,
                            calibration_minutes=5.0,
                            tmpl_cache=tmpl_cache,
                            run_groupO=not args.no_groupO,
                            model_name=name,
                            smooth_size=args.smooth_size,
                            smooth_causal=args.smooth_causal)
        rows.append(out)
        if "error" in out:
            print(f"  {rat}/{s}: ERROR {out['error']}  ({time.time()-t0:.1f}s)",
                  flush=True)
        else:
            print(f"  {rat}/{s}: kp_mse {out['kp_mse_align']:.0f}->{out['kp_mse_corr']:.0f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    # Aggregate per rat
    valid = [r for r in rows if "error" not in r]
    summary = {}
    for rat in sorted(set(r["rat"] for r in valid)):
        sub = [r for r in valid if r["rat"] == rat]
        if not sub: continue
        agg = {"n": len(sub),
               "kp_mse_align": float(np.mean([r["kp_mse_align"] for r in sub])),
               "kp_mse_corr":  float(np.mean([r["kp_mse_corr"]  for r in sub])),
               "pc_mse_align": np.mean([r["pc_mse_align"] for r in sub], axis=0).tolist(),
               "pc_mse_corr":  np.mean([r["pc_mse_corr"]  for r in sub], axis=0).tolist()}
        for label in ("raw_xyz", "procrustes_xyz", "corrected_xyz",
                      "raw_groupO", "procrustes_groupO", "corrected_groupO"):
            f1s = [r.get(f"{label}_f1_300") for r in sub if r.get(f"{label}_f1_300") is not None]
            if f1s:
                agg[f"{label}_f1_300"] = float(np.mean(f1s))
                agg[f"{label}_recall_300"] = float(np.mean(
                    [r.get(f"{label}_recall_300") for r in sub
                     if r.get(f"{label}_recall_300") is not None]))
                agg[f"{label}_precision_300"] = float(np.mean(
                    [r.get(f"{label}_precision_300") for r in sub
                     if r.get(f"{label}_precision_300") is not None]))
        summary[rat] = agg

    print("\n=== Summary ===")
    for rat in sorted(summary):
        s = summary[rat]
        print(f"\n{rat} (n={s['n']}):")
        print(f"  kp_mse: align={s['kp_mse_align']:.1f} -> corr={s['kp_mse_corr']:.1f}")
        print(f"  pc_mse: align={s['pc_mse_align']} -> corr={s['pc_mse_corr']}")
        for label in ("corrected_xyz", "corrected_groupO"):
            if f"{label}_f1_300" in s:
                print(f"  {label}: F1={s[label+'_f1_300']:.3f}  "
                      f"R={s[label+'_recall_300']:.3f}  "
                      f"P={s[label+'_precision_300']:.3f}")

    suffix = args.out_tag or ""
    out_path = RESULTS_DIR / f"{Path(args.ckpt).stem}{suffix}_all.json"
    out_path.write_text(json.dumps({"per_session": rows, "summary": summary,
                                      "ckpt": str(args.ckpt),
                                      "model_name": ck["model_name"]},
                                     indent=2))
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
