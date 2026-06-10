"""Trainer for TemporalMLPWith2DReproj: temporal_mlp + current-frame 2D inputs
+ current-frame per-camera reprojection residuals.

Per session:
  - 3D pose window from load_paired_world (SLEAP timeline, with interpolation
    through dropped video frames). This is the same source temporal_mlp uses.
  - Current-frame 2D + confidence + visibility from load_session_2d, joined to
    the SLEAP timeline via cam0_frame.
  - Current-frame per-cam reprojection residual: project the saved triangulated
    3D (NOT the median-filtered SLEAP from load_paired_world; load_session_2d's
    x_triang_3d is the un-smoothed geometry that actually produced the 2D
    detections) through the session calibration. residual = (detected - reproj)
    / RESID_NORM_PX. NaN/wrong-side gets zeroed and visibility flagged 0.

Procrustes is still fit on (sl, dn) from load_paired_world over the calibration
window — matches the temporal_mlp inference pipeline.

Usage:
    python -m corrector.train_temporal_mlp_2d_reproj --rats R1 R2 R3 \\
        --tag R1R2R3_temporal_mlp_2d_reproj_v1 --ctx 5 --epochs 100 \\
        --early_stop_patience 20 --noise_std_mm 50 --noise_prob 0.3
"""
from __future__ import annotations

import argparse
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

from corrector.data_world import (session_split_multi, SLEAP_HZ,
                                    load_paired_world)
from corrector.data_world_2d_from_saved import load_session_2d
from corrector.data_world_2d import reproject_all_cams
from corrector.models import build_model
from corrector.world_alignment import calibration_indices, fit_procrustes

CKPT_DIR = _THIS.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

VIDEO_W = 1920.0
VIDEO_H = 1200.0
N_CAM = 3
N_KP = 23

# Same normalization as the 2D-input refiners' reprojection residual channel.
RESID_NORM_PX = 100.0

OUTLIER_THRESHOLD_MM = 1000.0
RESIDUAL_CLIP_MM = 200.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TemporalMLP2DReprojDataset(Dataset):
    """Per-session arrays + (session_idx, sleap_t) flat index.

    For each kept session stores:
      sl_aligned : (T_sleap, 23, 3) Procrustes-aligned SLEAP 3D in DANNCE world
      dn         : (T_sleap, 23, 3) DANNCE 3D (med25)
      x_2d       : (T_sleap, 3, 23, 2)  current-frame 2D in [-1, 1]
      x_conf     : (T_sleap, 3, 23)     SLEAP confidence
      x_vis      : (T_sleap, 3, 23)     visibility (1/0)
      x_reproj   : (T_sleap, 3, 23, 2)  (detected - reprojected) / RESID_NORM_PX
      has_2d     : (T_sleap,) bool      whether this SLEAP frame has 2D info
    """

    def __init__(self, rat_to_sessions, ctx,
                 calibration_minutes=5.0, calibration_n_sample=1000,
                 max_residual=60.0, max_sessions_per_rat=None, verbose=True):
        self.ctx = ctx
        self.sessions = []
        self.session_residuals = []
        for rat, sess_list in rat_to_sessions.items():
            kept_for_rat = 0
            for s in sess_list:
                if (max_sessions_per_rat is not None
                        and kept_for_rat >= max_sessions_per_rat):
                    break
                try:
                    sl, dn = load_paired_world(rat, s)
                except Exception as e:
                    if verbose:
                        print(f"  skip {rat}/{s}: load_paired_world failed: {e}",
                              flush=True)
                    continue
                T_sleap = len(sl)
                if T_sleap < max(1000, ctx * 5):
                    if verbose:
                        print(f"  skip {rat}/{s}: T_sleap={T_sleap} too small",
                              flush=True)
                    continue
                try:
                    sd = load_session_2d(rat, s, smooth_dannce=True)
                except Exception as e:
                    if verbose:
                        print(f"  skip {rat}/{s}: load_session_2d failed: {e}",
                              flush=True)
                    continue
                if len(sd.x_2d) < 100:
                    if verbose:
                        print(f"  skip {rat}/{s}: too few processed frames",
                              flush=True)
                    continue
                idx = calibration_indices(T_sleap, calibration_minutes, SLEAP_HZ,
                                          calibration_n_sample, seed=0)
                if len(idx) < 100:
                    if verbose:
                        print(f"  skip {rat}/{s}: no cal window", flush=True)
                    continue
                # Procrustes DANNCE -> SLEAP so the model trains entirely in
                # SLEAP world. Lets online inference skip the runtime Procrustes
                # — `keypoints_3D` straight out of `triangulate(...)` is already
                # in the model's native input frame.
                tx = fit_procrustes(dn[idx], sl[idx], try_z_flip=True)
                self.session_residuals.append((rat, s, float(tx["residual"])))
                if tx["residual"] > max_residual:
                    if verbose:
                        print(f"  skip {rat}/{s}: residual={tx['residual']:.1f} "
                              f"> {max_residual}", flush=True)
                    continue

                sl_aligned = sl.astype(np.float32)               # already in SLEAP world
                dn32 = tx["apply"](dn).astype(np.float32)        # DANNCE -> SLEAP world (target)

                # Per-processed-frame reprojection of the saved triangulated 3D
                # (un-smoothed geometry that actually produced the 2D dets).
                # reproject_all_cams handles z<0 wrong-side -> NaN.
                # Shape (P, 3, 23, 2)
                reproj_pf = reproject_all_cams(sd.x_triang_3d, sd.calibration)
                detected_pf = sd.x_2d                              # (P, 3, 23, 2)
                # Per-cam visibility flag from the saved-2D loader: detection
                # finite AND conf > 0 AND reproj finite.
                conf_pf = sd.x_conf                                # (P, 3, 23)
                vis_pf = ((conf_pf > 0)
                          & np.isfinite(detected_pf).all(axis=-1)
                          & np.isfinite(reproj_pf).all(axis=-1))   # (P, 3, 23) bool
                resid_pf = (detected_pf - reproj_pf) / RESID_NORM_PX
                # Zero out the channels where invisible.
                resid_pf_safe = np.where(vis_pf[..., None],
                                          resid_pf, 0.0).astype(np.float32)
                detected_safe = np.where(vis_pf[..., None],
                                          detected_pf, 0.0).astype(np.float32)
                conf_safe = np.where(vis_pf, conf_pf, 0.0).astype(np.float32)

                # Scatter to SLEAP timeline.
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

                # Normalize detected 2D to [-1, 1].
                scale = np.array([VIDEO_W, VIDEO_H], dtype=np.float32).reshape(1, 1, 1, 2)
                x_2d_per_t = (x_2d_per_t / scale - 0.5) * 2.0

                self.sessions.append({
                    "rat": rat, "session": s,
                    "sl_aligned": sl_aligned,
                    "dn": dn32,
                    "x_2d": x_2d_per_t,
                    "x_conf": x_conf_per_t,
                    "x_vis": vis_per_t,
                    "x_reproj": reproj_per_t,
                    "has_2d": has_2d,
                    "residual": float(tx["residual"]),
                })
                kept_for_rat += 1
                if verbose:
                    n_has = int(has_2d.sum())
                    med_resid_norm = float(np.median(
                        np.linalg.norm(reproj_per_t[has_2d], axis=-1)))
                    print(f"  loaded {rat}/{s}: T_sleap={T_sleap}  "
                          f"has_2d={n_has}  resid={tx['residual']:.1f}  "
                          f"med_reproj_norm={med_resid_norm:.2f}", flush=True)

        self._index = []
        for si, sess in enumerate(self.sessions):
            sl_a = sess["sl_aligned"]; dn = sess["dn"]; has_2d = sess["has_2d"]
            T = len(sl_a)
            flat = sl_a.reshape(T, -1)
            flat_dn = dn.reshape(T, -1)
            finite = np.isfinite(flat).all(axis=1) & np.isfinite(flat_dn).all(axis=1)
            max_abs = np.where(finite, np.abs(flat).max(axis=1), np.inf)
            max_abs_dn = np.where(finite, np.abs(flat_dn).max(axis=1), np.inf)
            inlier_t = finite & (max_abs < OUTLIER_THRESHOLD_MM) \
                              & (max_abs_dn < OUTLIER_THRESHOLD_MM)
            cs = np.concatenate([[0], np.cumsum(inlier_t.astype(np.int32))])
            for t in range(ctx - 1, T):
                if not has_2d[t]:
                    continue
                if cs[t + 1] - cs[t + 1 - ctx] < ctx:
                    continue
                self._index.append((si, t))
        if verbose:
            print(f"TemporalMLP2DReprojDataset: {len(self.sessions)} sessions, "
                  f"{len(self._index):,} windows  (ctx={ctx})", flush=True)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        si, t = self._index[idx]
        sess = self.sessions[si]
        ctx = self.ctx
        return {
            "x_pose": sess["sl_aligned"][t - ctx + 1 : t + 1],
            "x_2d": sess["x_2d"][t],
            "x_conf": sess["x_conf"][t],
            "x_vis": sess["x_vis"][t],
            "x_reproj": sess["x_reproj"][t],
            "y": sess["dn"][t],
        }


def collate(batch):
    out = {}
    for k in ("x_pose", "x_2d", "x_conf", "x_vis", "x_reproj", "y"):
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
    return out


def bone_length_loss(pred, target, edges):
    e = torch.as_tensor(edges, dtype=torch.long, device=pred.device)
    pred_d = (pred[:, e[:, 0], :] - pred[:, e[:, 1], :]).norm(dim=-1)
    tgt_d = (target[:, e[:, 0], :] - target[:, e[:, 1], :]).norm(dim=-1)
    return ((pred_d - tgt_d) ** 2).mean()


def _apply_inference_guard(pred, x_pose_last):
    finite_in = torch.isfinite(x_pose_last).reshape(x_pose_last.shape[0], -1).all(dim=-1)
    max_abs = x_pose_last.reshape(x_pose_last.shape[0], -1).abs().max(dim=-1).values
    inlier = finite_in & (max_abs < OUTLIER_THRESHOLD_MM)
    delta = (pred - x_pose_last).clamp(min=-RESIDUAL_CLIP_MM, max=RESIDUAL_CLIP_MM)
    clipped = x_pose_last + delta
    return torch.where(inlier.view(-1, 1, 1), clipped, x_pose_last)


def evaluate(model, loader, device):
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_pose = batch["x_pose"].to(device, non_blocking=True)
            x_2d = batch["x_2d"].to(device, non_blocking=True)
            x_conf = batch["x_conf"].to(device, non_blocking=True)
            x_vis = batch["x_vis"].to(device, non_blocking=True)
            x_reproj = batch["x_reproj"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            pred = model(x_pose, x_2d, x_conf, x_vis, x_reproj)
            guarded = _apply_inference_guard(pred, x_pose[:, -1, :, :])
            sse += ((guarded - y) ** 2).sum().item()
            n += y.numel()
    return sse / max(n, 1)


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=collate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rats", nargs="+", required=True, choices=["R1", "R2", "R3"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_hidden_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--bone_weight", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_residual", type=float, default=60.0)
    ap.add_argument("--max_sessions_per_rat", type=int, default=None)
    ap.add_argument("--early_stop_patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise_std_mm", type=float, default=50.0,
                    help="Gaussian noise stddev added to input pose. With "
                         "--noise_mode global, applied to every keypoint of "
                         "every frame in the window (the v1/v3 behavior). With "
                         "--noise_mode targeted, only K keypoints in the LAST "
                         "frame are corrupted.")
    ap.add_argument("--noise_prob", type=float, default=0.3)
    ap.add_argument("--noise_mode", choices=["global", "targeted"],
                    default="global",
                    help="how the noise is applied. 'global' = whole-skeleton "
                         "all-frames noise (v1/v3 behavior). 'targeted' = K "
                         "random keypoints in the LAST frame, modeling 1-3 "
                         "misdetections at inference time.")
    ap.add_argument("--targeted_noise_max_kp", type=int, default=3,
                    help="when --noise_mode targeted, K is sampled uniformly "
                         "from [1, targeted_noise_max_kp] per noisy sample.")
    ap.add_argument("--lr_schedule", choices=["none", "cosine"], default="none",
                    help="LR schedule. 'cosine' anneals --lr down to --lr_min "
                         "over the full --epochs window using CosineAnnealingLR.")
    ap.add_argument("--lr_min", type=float, default=1e-5,
                    help="floor LR for the cosine schedule.")
    ap.add_argument("--init_ckpt", type=str, default=None,
                    help="path to a temporal_mlp_2d_reproj ckpt to init weights "
                         "from. Architecture (ctx, hidden, n_hidden_layers) is "
                         "taken from the ckpt; CLI values ignored for those.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = session_split_multi(args.rats, seed=args.seed)
    print("Split sizes:")
    for which in ("train", "val", "test"):
        print(f"  {which}: " + str({r: len(splits[which][r]) for r in args.rats}),
              flush=True)

    print("\nLoading train sessions...", flush=True)
    train_ds = TemporalMLP2DReprojDataset(
        splits["train"], ctx=args.ctx,
        max_residual=args.max_residual,
        max_sessions_per_rat=args.max_sessions_per_rat)
    print("\nLoading val sessions...", flush=True)
    val_ds = TemporalMLP2DReprojDataset(
        splits["val"], ctx=args.ctx,
        max_residual=args.max_residual,
        max_sessions_per_rat=args.max_sessions_per_rat)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_ds)} val={len(val_ds)}")

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)

    if args.init_ckpt:
        init_ck = torch.load(args.init_ckpt, map_location="cpu",
                             weights_only=False)
        if init_ck.get("model_name") != "temporal_mlp_2d_reproj":
            raise ValueError(
                f"--init_ckpt model_name={init_ck.get('model_name')!r}, "
                "expected 'temporal_mlp_2d_reproj'")
        args.ctx = init_ck.get("ctx", args.ctx)
        args.hidden = init_ck.get("hidden", args.hidden)
        args.n_hidden_layers = init_ck.get("n_hidden_layers", args.n_hidden_layers)
        args.dropout = init_ck.get("dropout", args.dropout)
        print(f"\ninit from {args.init_ckpt}: ctx={args.ctx} hidden={args.hidden} "
              f"n_hidden_layers={args.n_hidden_layers}", flush=True)

    model = build_model("temporal_mlp_2d_reproj", ctx=args.ctx, hidden=args.hidden,
                        n_hidden_layers=args.n_hidden_layers,
                        dropout=args.dropout).to(device)
    if args.init_ckpt:
        model.load_state_dict(init_ck["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: temporal_mlp_2d_reproj  params: {n_params:,}  device: {device}",
          flush=True)
    print(f"input dim: {model.in_dim}  (pose={args.ctx*N_KP*3}, "
          f"2D={N_CAM*N_KP*2}, conf={N_CAM*N_KP}, vis={N_CAM*N_KP}, "
          f"reproj={N_CAM*N_KP*2})", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr_min)
    else:
        scheduler = None

    best_val = float("inf"); best_state = None; history = []
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_sse, train_n = 0.0, 0
        for batch in train_loader:
            x_pose = batch["x_pose"].to(device, non_blocking=True)
            x_2d = batch["x_2d"].to(device, non_blocking=True)
            x_conf = batch["x_conf"].to(device, non_blocking=True)
            x_vis = batch["x_vis"].to(device, non_blocking=True)
            x_reproj = batch["x_reproj"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            if args.noise_std_mm > 0 and args.noise_prob > 0:
                B = x_pose.shape[0]
                sample_mask = (torch.rand(B, device=device) < args.noise_prob)
                if sample_mask.any():
                    if args.noise_mode == "global":
                        # Original behavior: whole-skeleton, all-frames noise.
                        noise = torch.randn_like(x_pose) * args.noise_std_mm
                        noise = noise * sample_mask.view(-1, 1, 1, 1).to(x_pose.dtype)
                        x_pose = x_pose + noise
                    else:
                        # Targeted: pick K random kp in the LAST frame only,
                        # K ~ Uniform[1, targeted_noise_max_kp]. Models the
                        # real failure mode (1-3 keypoints misdetected at the
                        # current frame; temporal context still intact).
                        # Vectorized: per-sample random scores, then topk by K.
                        max_k = max(1, args.targeted_noise_max_kp)
                        # K per noisy sample, uniform in [1, max_k].
                        K = torch.randint(1, max_k + 1, (B,), device=device)
                        # Rank threshold: keypoints with rank < K get noised.
                        scores = torch.rand(B, N_KP, device=device)
                        ranks = scores.argsort(dim=1).argsort(dim=1)  # (B, N_KP)
                        kp_mask = (ranks < K.unsqueeze(1)) & sample_mask.unsqueeze(1)
                        noise_last = torch.randn(B, N_KP, 3, device=device,
                                                  dtype=x_pose.dtype) \
                                       * args.noise_std_mm
                        noise_last = noise_last * kp_mask.unsqueeze(-1).to(x_pose.dtype)
                        # In-place add to the last-frame slice — avoids cat.
                        x_pose = x_pose.clone()
                        x_pose[:, -1, :, :] = x_pose[:, -1, :, :] + noise_last

            pred = model(x_pose, x_2d, x_conf, x_vis, x_reproj)
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
        val_mse = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "val_mse": val_mse, "elapsed_s": elapsed,
                        "lr": cur_lr})
        print(f"epoch {epoch:3d}  train_mse={train_mse:.3f}  "
              f"val_mse={val_mse:.3f}  lr={cur_lr:.2e}  "
              f"({elapsed:.1f}s)", flush=True)
        if scheduler is not None:
            scheduler.step()
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
        "model_name": "temporal_mlp_2d_reproj",
        "ctx": args.ctx,
        "hidden": args.hidden,
        "n_hidden_layers": args.n_hidden_layers,
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
        "residual_clip_mm": RESIDUAL_CLIP_MM,
        "session_residuals_train": train_ds.session_residuals,
        "session_residuals_val": val_ds.session_residuals,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lr_min": args.lr_min,
        "init_ckpt": args.init_ckpt,
        "noise_std_mm": args.noise_std_mm,
        "noise_prob": args.noise_prob,
        "noise_mode": args.noise_mode,
        "targeted_noise_max_kp": args.targeted_noise_max_kp,
        "resid_norm_px": RESID_NORM_PX,
        "video_w": VIDEO_W, "video_h": VIDEO_H,
        "procrustes_direction": "dannce_to_sleap",
    }
    torch.save(save, ckpt)
    print(f"\nsaved {ckpt}  best_val_mse={best_val:.3f}", flush=True)


if __name__ == "__main__":
    main()
