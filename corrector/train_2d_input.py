"""Trainer for the 2D-input corrector (TriangulationRefiner).

Pulls per-processed-frame data from `PairedSession2DDataset` (saved-2D path,
no video re-inference) and trains a PointNet-style per-keypoint MLP that takes
[xyz_triang, per_cam_xy, per_cam_conf, per_cam_reproj_resid, per_cam_vis] and
outputs a residual added to the triangulated xyz.

Pipeline per session:
  1. Load (x_2d, x_conf, x_triang_3d, y_dannce_3d, calibration) via
     `load_session_2d` -> SavedSession2D.
  2. Fit Procrustes on (x_triang_3d, y_dannce_3d) over the calibration window
     (first 5 min). Gate by residual <= max_residual.
  3. Pre-compute Procrustes-aligned x_triang_3d (SLEAP-world -> DANNCE-world)
     and store both. Targets are y_dannce_3d.
  4. Sample per-frame; build (B, 23, 21) features on the fly with reprojection
     residuals computed in torch from x_2d, x_conf, calib, and the *aligned*
     triangulated 3D mapped back through `apply_inverse` for the projection
     (since calibration lives in SLEAP world).

Usage:
  python -m corrector.train_2d_input --rats R1 R2 R3 --tag R1R2R3_2d_v1
  python -m corrector.train_2d_input --rats R2 --tag R2_2d_smoke --epochs 2 \\
        --max_sessions_per_rat 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from config import EDGES

from corrector.data_world import session_split_multi, SLEAP_HZ
from corrector.data_world_2d_from_saved import load_session_2d
from corrector.models import build_model
from corrector.world_alignment import calibration_indices, fit_procrustes

CKPT_DIR = _THIS.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

VIDEO_W = 1920.0
VIDEO_H = 1200.0
N_CAM = 3
N_KP = 23

# Per-cam reprojection residual normalization: residual / RESID_NORM_PX.
# 100 px is roughly the std of pre-correction reproj residuals (see Phase H.2
# table); normalizing keeps inputs O(1).
RESID_NORM_PX = 100.0

# Drop training frames where the per-sample max |x_triang_3d| or |y_dannce_3d|
# exceeds this many mm. The arena is ~500 mm; anything past ~1 m is a
# triangulation failure (sparse: ~10-20 frames per ~1.5M-sample train set on
# R1, but single outliers at ~11 m drive train_mse to 10^10).
OUTLIER_THRESHOLD_MM = 1000.0


# ---------------------------------------------------------------------------
# Torch reprojection (mirrors corrector.data_world_2d.project_3d_to_2d_batch)
# ---------------------------------------------------------------------------

def project_3d_to_2d_torch(points_3d: torch.Tensor, cam: dict) -> torch.Tensor:
    """Single-camera projection. points_3d: (..., 3) in SLEAP world.
    Returns (..., 2) pixel coords. NaN for points on the wrong side (z >= 0).

    cam: dict with K (3,3), r (3,3), t (3,), dist (4,) as float tensors on the
    same device as points_3d. Kept around for the eval path
    (correct_triangulation_refiner) which only deals with one session at a time.
    """
    K, r, t, dist = cam["K"], cam["r"], cam["t"], cam["dist"]
    orig_shape = points_3d.shape
    flat = points_3d.reshape(-1, 3)
    pts_cam = flat @ r.t() + t.unsqueeze(0)
    z = pts_cam[:, 2]
    z_safe = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
    x_norm = pts_cam[:, 0] / z_safe
    y_norm = pts_cam[:, 1] / z_safe
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
    r2 = x_norm ** 2 + y_norm ** 2
    radial = 1 + k1 * r2 + k2 * r2 ** 2
    x_d = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
    y_d = y_norm * radial + p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x_d + cx
    v = fy * y_d + cy
    out = torch.stack([u, v], dim=-1).reshape(*orig_shape[:-1], 2)
    # Wrong-side -> NaN
    wrong = (z >= 0).reshape(*orig_shape[:-1])
    out[wrong] = float("nan")
    return out


def project_3d_to_2d_torch_batched(points: torch.Tensor,
                                    K: torch.Tensor, r: torch.Tensor,
                                    t: torch.Tensor, dist: torch.Tensor
                                    ) -> torch.Tensor:
    """Fully-vectorized multi-camera, multi-sample projection.

    points : (B, N_CAM, N_KP, 3)    SLEAP-world keypoints (broadcast across cams)
    K      : (B, N_CAM, 3, 3)       intrinsics
    r      : (B, N_CAM, 3, 3)       rotation
    t      : (B, N_CAM, 3)          translation
    dist   : (B, N_CAM, 4)          distortion [k1, k2, p1, p2]

    Returns (B, N_CAM, N_KP, 2) pixel coords, NaN for z >= 0.

    All math is the same as `project_3d_to_2d_torch`, just shaped to broadcast.
    """
    # Camera-frame points: (B, N_CAM, N_KP, 3)
    # r @ p is matmul over the trailing 3 — use einsum to keep it explicit.
    pts_cam = torch.einsum("bcij,bckj->bcki", r, points) + t.unsqueeze(2)
    z = pts_cam[..., 2]                                       # (B, C, K)
    z_safe = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
    x_norm = pts_cam[..., 0] / z_safe
    y_norm = pts_cam[..., 1] / z_safe

    k1 = dist[..., 0:1]                                       # (B, C, 1)
    k2 = dist[..., 1:2]
    p1 = dist[..., 2:3]
    p2 = dist[..., 3:4]

    r2 = x_norm ** 2 + y_norm ** 2
    radial = 1 + k1 * r2 + k2 * r2 ** 2
    x_d = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
    y_d = y_norm * radial + p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm

    fx = K[..., 0, 0:1]                                       # (B, C, 1)
    fy = K[..., 1, 1:2]
    cx = K[..., 0, 2:3]
    cy = K[..., 1, 2:3]
    u = fx * x_d + cx
    v = fy * y_d + cy
    out = torch.stack([u, v], dim=-1)                         # (B, C, K, 2)

    wrong = (z >= 0).unsqueeze(-1).expand_as(out)
    return torch.where(wrong, torch.full_like(out, float("nan")), out)


def calib_to_torch(calib: list[dict], device, dtype=torch.float32) -> list[dict]:
    """Convert a per-camera list of numpy dicts to per-camera torch dicts.
    Kept for the single-session eval path (`correct_triangulation_refiner`)."""
    out = []
    for cam in calib:
        out.append({
            "K": torch.as_tensor(cam["K"], dtype=dtype, device=device),
            "r": torch.as_tensor(cam["r"], dtype=dtype, device=device),
            "t": torch.as_tensor(cam["t"], dtype=dtype, device=device).reshape(3),
            "dist": torch.as_tensor(cam["dist"], dtype=dtype, device=device).reshape(4),
        })
    return out


def stack_session_calibs(session_calibs: list[list[dict]], device,
                          dtype=torch.float32) -> dict:
    """Stack per-session per-camera calibrations into batched tensors.

    session_calibs[i] is a list of N_CAM camera dicts (numpy arrays) for
    session i. Returns a dict of tensors of shape (S, N_CAM, ...):
        K     (S, N_CAM, 3, 3)
        r     (S, N_CAM, 3, 3)
        t     (S, N_CAM, 3)
        dist  (S, N_CAM, 4)
    where S = len(session_calibs).
    """
    S = len(session_calibs)
    K = torch.empty((S, N_CAM, 3, 3), dtype=dtype, device=device)
    r = torch.empty((S, N_CAM, 3, 3), dtype=dtype, device=device)
    t = torch.empty((S, N_CAM, 3), dtype=dtype, device=device)
    dist = torch.empty((S, N_CAM, 4), dtype=dtype, device=device)
    for si, cams in enumerate(session_calibs):
        if len(cams) != N_CAM:
            raise ValueError(f"session {si}: {len(cams)} cams, expected {N_CAM}")
        for ci, cam in enumerate(cams):
            K[si, ci] = torch.as_tensor(cam["K"], dtype=dtype, device=device)
            r[si, ci] = torch.as_tensor(cam["r"], dtype=dtype, device=device)
            t[si, ci] = torch.as_tensor(cam["t"], dtype=dtype,
                                         device=device).reshape(3)
            dist[si, ci] = torch.as_tensor(cam["dist"], dtype=dtype,
                                             device=device).reshape(4)
    return {"K": K, "r": r, "t": t, "dist": dist}


# ---------------------------------------------------------------------------
# Dataset that holds per-session arrays + transforms
# ---------------------------------------------------------------------------

class Paired2DTrainDataset(Dataset):
    """Per-session loader. Caches Procrustes-aligned triangulated 3D and the
    transform (so we can map back to SLEAP world for reprojection).

    __getitem__ returns a dict of numpy arrays (no per-cam reprojection done
    here — that happens batched in the trainer where calibration tensors live
    on the device).
    """

    def __init__(self, rat_to_sessions: dict[str, list[str]],
                 calibration_minutes: float = 5.0,
                 calibration_n_sample: int = 1000,
                 drop_first_seconds: float = 30.0,
                 max_residual: float = 60.0,
                 max_sessions_per_rat: int | None = None,
                 verbose: bool = True):
        self.sessions = []           # list of dicts per kept session
        self.session_residuals = []  # (rat, session, residual) for all loaded
        drop_n = int(round(drop_first_seconds * SLEAP_HZ))

        for rat, sess_list in rat_to_sessions.items():
            kept_for_rat = 0
            for s in sess_list:
                if (max_sessions_per_rat is not None
                        and kept_for_rat >= max_sessions_per_rat):
                    break
                try:
                    sd = load_session_2d(rat, s, smooth_dannce=True)
                except Exception as e:
                    if verbose:
                        print(f"  skip {rat}/{s}: load failed: {e}", flush=True)
                    continue
                P = len(sd.x_triang_3d)
                if P < 1000:
                    if verbose:
                        print(f"  skip {rat}/{s}: only {P} samples", flush=True)
                    continue
                idx = calibration_indices(P, calibration_minutes, SLEAP_HZ,
                                          calibration_n_sample, seed=0)
                if len(idx) < 100:
                    if verbose:
                        print(f"  skip {rat}/{s}: too few calibration frames",
                              flush=True)
                    continue
                tx = fit_procrustes(sd.x_triang_3d[idx], sd.y_dannce_3d[idx],
                                    try_z_flip=True)
                self.session_residuals.append((rat, s, float(tx["residual"])))
                if tx["residual"] > max_residual:
                    if verbose:
                        print(f"  skip {rat}/{s}: residual={tx['residual']:.1f} mm "
                              f"> {max_residual} (uncalibrated)", flush=True)
                    continue
                # Apply Procrustes to triang 3D (SLEAP world -> DANNCE world)
                x_triang_aligned = tx["apply"](sd.x_triang_3d).astype(np.float32)
                if P <= drop_n:
                    continue

                # Drop the warmup window, then drop outlier frames
                # (triangulation failures). The arena is ~500 mm; anything
                # past OUTLIER_THRESHOLD_MM is a SLEAP triangulation glitch
                # that drives train_mse to absurd values without informing
                # the model.
                x2 = sd.x_2d[drop_n:].astype(np.float32)
                xc = sd.x_conf[drop_n:].astype(np.float32)
                xts = sd.x_triang_3d[drop_n:].astype(np.float32)
                xtd = x_triang_aligned[drop_n:]
                yd = sd.y_dannce_3d[drop_n:].astype(np.float32)

                max_abs_tri = np.abs(xts).reshape(len(xts), -1).max(axis=1)
                max_abs_dn = np.abs(yd).reshape(len(yd), -1).max(axis=1)
                keep = ((max_abs_tri < OUTLIER_THRESHOLD_MM)
                        & (max_abs_dn < OUTLIER_THRESHOLD_MM)
                        & np.isfinite(xts).reshape(len(xts), -1).all(axis=1)
                        & np.isfinite(yd).reshape(len(yd), -1).all(axis=1))
                n_dropped = int((~keep).sum())

                self.sessions.append({
                    "rat": rat, "session": s,
                    "x_2d": x2[keep],
                    "x_conf": xc[keep],
                    "x_triang_sleap": xts[keep],
                    "x_triang_dannce": xtd[keep],
                    "y_dannce": yd[keep],
                    "calibration": sd.calibration,
                    "cal_date": sd.cal_date,
                    "residual": float(tx["residual"]),
                    "n_outliers_dropped": n_dropped,
                })
                kept_for_rat += 1
                if verbose:
                    print(f"  loaded {rat}/{s}: P={int(keep.sum())}  "
                          f"resid={tx['residual']:.1f}  "
                          f"cal={sd.cal_date}  "
                          f"outliers_dropped={n_dropped}", flush=True)

        # Flat (session_idx, sample_idx) index
        self._index = []
        for si, sess in enumerate(self.sessions):
            for f in range(len(sess["x_2d"])):
                self._index.append((si, f))
        if verbose:
            print(f"Paired2DTrainDataset: {len(self.sessions)} sessions, "
                  f"{len(self._index):,} samples", flush=True)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        si, fi = self._index[idx]
        sess = self.sessions[si]
        return {
            "x_2d": sess["x_2d"][fi],                  # (3, 23, 2)
            "x_conf": sess["x_conf"][fi],              # (3, 23)
            "x_triang_sleap": sess["x_triang_sleap"][fi],  # (23, 3)
            "x_triang_dannce": sess["x_triang_dannce"][fi],
            "y_dannce": sess["y_dannce"][fi],          # (23, 3)
            "session_idx": si,
        }


def collate(batch: list[dict]) -> dict:
    """Stack per-frame dicts and keep session_idx as a long tensor so the
    trainer can fetch per-sample calibration."""
    out = {}
    keys_arr = ["x_2d", "x_conf", "x_triang_sleap", "x_triang_dannce", "y_dannce"]
    for k in keys_arr:
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
    out["session_idx"] = torch.as_tensor([b["session_idx"] for b in batch],
                                          dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Feature assembly: build (B, 23, 21) on the device
# ---------------------------------------------------------------------------

def build_features(x_triang_dannce: torch.Tensor,   # (B, 23, 3) — DANNCE world
                   x_triang_sleap: torch.Tensor,    # (B, 23, 3) — SLEAP world (for reproj)
                   x_2d: torch.Tensor,              # (B, 3, 23, 2)
                   x_conf: torch.Tensor,            # (B, 3, 23)
                   sess_calib_stacked,              # dict from stack_session_calibs
                                                    # OR legacy list[list[dict]]
                   session_idx: torch.Tensor,       # (B,) long
                   ) -> torch.Tensor:
    """Returns (B, 23, 21) per-kp features. NaNs in 2D / residuals are zeroed,
    and the visibility channel records what was real.

    sess_calib_stacked may be:
      - the stacked dict from `stack_session_calibs` (preferred; one-time setup
        in the trainer), or
      - a list of per-session lists of per-cam dicts (legacy form used by the
        single-session eval path). The list form is converted to the stacked
        dict on first use of this code path.
    """
    B = x_triang_dannce.shape[0]
    device = x_triang_dannce.device

    # Normalize calibration input to the stacked dict form.
    if isinstance(sess_calib_stacked, list):
        sess_calib_stacked = stack_session_calibs(
            sess_calib_stacked, device, dtype=x_triang_sleap.dtype)

    # Gather per-sample calibration via session_idx, then run one fully-batched
    # reprojection over (B, N_CAM, N_KP, 3). Replaces the previous per-session
    # Python loop, which fired ~600+ small CUDA kernels per batch.
    K_b    = sess_calib_stacked["K"][session_idx]              # (B, C, 3, 3)
    r_b    = sess_calib_stacked["r"][session_idx]              # (B, C, 3, 3)
    t_b    = sess_calib_stacked["t"][session_idx]              # (B, C, 3)
    dist_b = sess_calib_stacked["dist"][session_idx]           # (B, C, 4)
    pts_bcast = x_triang_sleap.unsqueeze(1).expand(B, N_CAM, N_KP, 3)
    reproj = project_3d_to_2d_torch_batched(pts_bcast, K_b, r_b, t_b, dist_b)

    # Normalized 2D pixel coords: (px / (W or H) - 0.5) * 2 -> [-1, 1]-ish
    # Use a single scale to keep things simple (max(W, H) = 1920 captures both).
    scale = torch.tensor([VIDEO_W, VIDEO_H], device=device,
                         dtype=x_2d.dtype).reshape(1, 1, 1, 2)
    xy_norm = (x_2d / scale - 0.5) * 2.0                # (B, 3, 23, 2)
    resid = (x_2d - reproj) / RESID_NORM_PX             # (B, 3, 23, 2)

    # Visibility: detection finite AND confidence > 0
    vis = ((x_conf > 0)
           & torch.isfinite(x_2d).all(dim=-1)
           & torch.isfinite(reproj).all(dim=-1))        # (B, 3, 23)
    vis_f = vis.to(x_2d.dtype)

    # Zero out NaNs in xy_norm and resid where invisible
    xy_norm = torch.where(vis.unsqueeze(-1),
                          xy_norm, torch.zeros_like(xy_norm))
    resid = torch.where(vis.unsqueeze(-1),
                        resid, torch.zeros_like(resid))
    conf_z = torch.where(vis, x_conf, torch.zeros_like(x_conf))

    # Re-arrange per-cam blocks to per-kp: (B, 23, 3*2), (B, 23, 3), etc.
    # xy_norm (B, 3, 23, 2) -> (B, 23, 3, 2) -> (B, 23, 6)
    xy_per_kp = xy_norm.permute(0, 2, 1, 3).reshape(B, N_KP, N_CAM * 2)
    resid_per_kp = resid.permute(0, 2, 1, 3).reshape(B, N_KP, N_CAM * 2)
    conf_per_kp = conf_z.permute(0, 2, 1)        # (B, 23, 3)
    vis_per_kp = vis_f.permute(0, 2, 1)          # (B, 23, 3)

    feat = torch.cat([
        x_triang_dannce,    # 3
        xy_per_kp,          # 6
        conf_per_kp,        # 3
        resid_per_kp,       # 6
        vis_per_kp,         # 3
    ], dim=-1)              # (B, 23, 21)
    return feat


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def bone_length_loss(pred, target, edges):
    e = torch.as_tensor(edges, dtype=torch.long, device=pred.device)
    pred_d = (pred[:, e[:, 0], :] - pred[:, e[:, 1], :]).norm(dim=-1)
    tgt_d = (target[:, e[:, 0], :] - target[:, e[:, 1], :]).norm(dim=-1)
    return ((pred_d - tgt_d) ** 2).mean()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=collate)


def evaluate(model, loader, sess_calib_torch, device):
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_triang_d = batch["x_triang_dannce"].to(device, non_blocking=True)
            x_triang_s = batch["x_triang_sleap"].to(device, non_blocking=True)
            x_2d = batch["x_2d"].to(device, non_blocking=True)
            x_conf = batch["x_conf"].to(device, non_blocking=True)
            y = batch["y_dannce"].to(device, non_blocking=True)
            session_idx = batch["session_idx"].to(device, non_blocking=True)
            feat = build_features(x_triang_d, x_triang_s, x_2d, x_conf,
                                  sess_calib_torch, session_idx)
            pred = model(feat)
            sse += ((pred - y) ** 2).sum().item()
            n += y.numel()
    return sse / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rats", nargs="+", required=True,
                    choices=["R1", "R2", "R3"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_per_kp_layers", type=int, default=3)
    ap.add_argument("--global_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout prob inside per-kp MLP (0 disables)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--bone_weight", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="clip gradient norm to this value; 0 disables")
    ap.add_argument("--max_residual", type=float, default=60.0)
    ap.add_argument("--max_sessions_per_rat", type=int, default=None)
    ap.add_argument("--early_stop_patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_ckpt", type=str, default=None,
                    help="path to a triangulation_refiner ckpt to initialize "
                         "weights from (e.g. for per-rat fine-tuning). "
                         "Architecture (hidden, n_per_kp_layers, global_dim, "
                         "dropout) is taken from the ckpt and the CLI values "
                         "are ignored.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = session_split_multi(args.rats, seed=args.seed)
    print("Split sizes:")
    for which in ("train", "val", "test"):
        sizes = {r: len(splits[which][r]) for r in args.rats}
        print(f"  {which}: {sizes}", flush=True)

    print("\nLoading train sessions...", flush=True)
    train_ds = Paired2DTrainDataset(
        splits["train"], max_residual=args.max_residual,
        max_sessions_per_rat=args.max_sessions_per_rat)
    print("\nLoading val sessions...", flush=True)
    val_ds = Paired2DTrainDataset(
        splits["val"], max_residual=args.max_residual,
        max_sessions_per_rat=args.max_sessions_per_rat)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_ds)} val={len(val_ds)}")

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)

    # Pre-load per-session calibrations onto device as stacked tensors
    # (separate for train vs val because session indices are local to each
    # dataset). The stacked form lets build_features gather per-sample calibs
    # via session_idx in a single op instead of looping over sessions.
    train_cal = stack_session_calibs(
        [s["calibration"] for s in train_ds.sessions], device)
    val_cal = stack_session_calibs(
        [s["calibration"] for s in val_ds.sessions], device)

    if args.init_ckpt:
        init_ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        if init_ck.get("model_name") != "triangulation_refiner":
            raise ValueError(f"--init_ckpt model_name={init_ck.get('model_name')!r}, "
                             "expected 'triangulation_refiner'")
        # Use the ckpt's architecture so weight shapes match.
        args.hidden = init_ck.get("hidden", args.hidden)
        args.n_per_kp_layers = init_ck.get("n_per_kp_layers", args.n_per_kp_layers)
        args.global_dim = init_ck.get("global_dim", args.global_dim)
        args.dropout = init_ck.get("dropout", args.dropout)
        print(f"\ninit from {args.init_ckpt}: hidden={args.hidden} "
              f"n_per_kp_layers={args.n_per_kp_layers} global_dim={args.global_dim} "
              f"dropout={args.dropout}", flush=True)

    model = build_model("triangulation_refiner",
                        hidden=args.hidden,
                        n_per_kp_layers=args.n_per_kp_layers,
                        global_dim=args.global_dim,
                        dropout=args.dropout).to(device)
    if args.init_ckpt:
        model.load_state_dict(init_ck["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: triangulation_refiner  params: {n_params:,}  device: {device}",
          flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    best_val = float("inf"); best_state = None; history = []
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_sse, train_n = 0.0, 0
        for batch in train_loader:
            x_triang_d = batch["x_triang_dannce"].to(device, non_blocking=True)
            x_triang_s = batch["x_triang_sleap"].to(device, non_blocking=True)
            x_2d = batch["x_2d"].to(device, non_blocking=True)
            x_conf = batch["x_conf"].to(device, non_blocking=True)
            y = batch["y_dannce"].to(device, non_blocking=True)
            session_idx = batch["session_idx"].to(device, non_blocking=True)

            feat = build_features(x_triang_d, x_triang_s, x_2d, x_conf,
                                  train_cal, session_idx)
            pred = model(feat)
            loss_mse = ((pred - y) ** 2).mean()
            loss = loss_mse
            if args.bone_weight > 0:
                loss = loss + args.bone_weight * bone_length_loss(pred, y, EDGES)

            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                 max_norm=args.grad_clip)
            opt.step()
            train_sse += ((pred.detach() - y) ** 2).sum().item()
            train_n += y.numel()
        train_mse = train_sse / max(train_n, 1)
        val_mse = evaluate(model, val_loader, val_cal, device)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "val_mse": val_mse, "elapsed_s": elapsed})
        print(f"epoch {epoch:3d}  train_mse={train_mse:.3f}  "
              f"val_mse={val_mse:.3f}  ({elapsed:.1f}s)", flush=True)
        if val_mse < best_val - 1e-4:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = CKPT_DIR / f"{args.tag}.pt"
    save = {
        "model_name": "triangulation_refiner",
        "hidden": args.hidden,
        "n_per_kp_layers": args.n_per_kp_layers,
        "global_dim": args.global_dim,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "rats": args.rats, "tag": args.tag,
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "best_val_mse": best_val, "history": history,
        "splits": splits,
        "max_residual": args.max_residual,
        "bone_weight": args.bone_weight,
        "grad_clip": args.grad_clip,
        "outlier_threshold_mm": OUTLIER_THRESHOLD_MM,
        "session_residuals_train": train_ds.session_residuals,
        "session_residuals_val": val_ds.session_residuals,
        "init_ckpt": args.init_ckpt,
        "lr": args.lr,
        "resid_norm_px": RESID_NORM_PX,
        "video_w": VIDEO_W, "video_h": VIDEO_H,
    }
    torch.save(save, ckpt)
    print(f"\nsaved {ckpt}  best_val_mse={best_val:.3f}", flush=True)


if __name__ == "__main__":
    main()
