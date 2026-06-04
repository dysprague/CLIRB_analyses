"""Trainer for the temporal 2D-input corrector (TemporalTriangulationRefiner).

Pulls per-processed-frame data from a `Paired2DTemporalDataset` (wraps the
per-session arrays from `Paired2DTrainDataset` and emits T_ctx-frame causal
windows). Per-frame features are the same 21-dim bundle as the single-frame
refiner; the model outputs a corrected pose for the LAST frame in the window.

Usage:
    python -m corrector.train_2d_temporal --rats R1 R2 R3 \\
        --tag R1R2R3_2d_temporal_v1 --ctx 5 --epochs 80 \\
        --early_stop_patience 15 --dropout 0.1 --weight_decay 1e-4 \\
        --bone_weight 0
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

# Re-use the per-frame feature pipeline + calibration helpers from the
# single-frame trainer.
from corrector.train_2d_input import (
    Paired2DTrainDataset, build_features,
    stack_session_calibs,
    VIDEO_W, VIDEO_H, N_CAM, N_KP, RESID_NORM_PX, OUTLIER_THRESHOLD_MM,
)

CKPT_DIR = _THIS.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# Inference-side residual clip, matching evaluate_all.correct_triangulation_refiner.
RESIDUAL_CLIP_MM = 200.0


# ---------------------------------------------------------------------------
# Temporal dataset: wrap per-session arrays into T_ctx-length causal windows.
# ---------------------------------------------------------------------------

class Paired2DTemporalDataset(Dataset):
    """Builds T_ctx-length causal windows from a `Paired2DTrainDataset`.

    A sample is indexed by (session_idx, end_frame); we return frames
    [end-ctx+1 ... end] for each per-session array. The base dataset has
    already dropped per-sample outliers and the warmup window, so per-session
    arrays are dense.

    Window-level outlier filter: any frame in the window failing the inlier
    check would already have been dropped by the base dataset's per-sample
    filter. The remaining failure mode is "the contiguous-frames assumption is
    invalid because the base dropped some interior frame." We tolerate that —
    the model treats consecutive samples as ~50 ms apart; a tiny number of
    invisible 1-sample gaps adds noise but doesn't change the optimization.
    """

    def __init__(self, base: Paired2DTrainDataset, ctx: int):
        self.base = base
        self.ctx = ctx
        # Flat (session_idx, end_frame_within_session) index, skipping the
        # first ctx-1 frames of each session.
        self._index = []
        for si, sess in enumerate(base.sessions):
            n = len(sess["x_2d"])
            for fi in range(ctx - 1, n):
                self._index.append((si, fi))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        si, end = self._index[idx]
        start = end - self.ctx + 1
        sess = self.base.sessions[si]
        return {
            "x_2d": sess["x_2d"][start:end + 1],               # (T, 3, 23, 2)
            "x_conf": sess["x_conf"][start:end + 1],           # (T, 3, 23)
            "x_triang_sleap": sess["x_triang_sleap"][start:end + 1],  # (T, 23, 3)
            "x_triang_dannce": sess["x_triang_dannce"][start:end + 1],
            # Target is the DANNCE pose at the LAST frame in the window.
            "y_dannce_last": sess["y_dannce"][end],            # (23, 3)
            "session_idx": si,
        }


def collate_temporal(batch: list[dict]) -> dict:
    """Stack T_ctx-windowed dicts into batched tensors."""
    out = {}
    keys_arr = ["x_2d", "x_conf", "x_triang_sleap", "x_triang_dannce"]
    for k in keys_arr:
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
    out["y_dannce_last"] = torch.from_numpy(
        np.stack([b["y_dannce_last"] for b in batch]))
    out["session_idx"] = torch.as_tensor([b["session_idx"] for b in batch],
                                          dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Feature builder for the temporal case: reshape (B, T, ...) -> (B*T, ...),
# call the existing per-frame build_features, then reshape back.
# ---------------------------------------------------------------------------

def build_features_temporal(x_triang_d, x_triang_s, x_2d, x_conf,
                             sess_calib_stacked, session_idx):
    """Returns (B, T, 23, 21) per-frame features.

    Inputs:
      x_triang_d : (B, T, 23, 3)
      x_triang_s : (B, T, 23, 3)
      x_2d       : (B, T, 3, 23, 2)
      x_conf     : (B, T, 3, 23)
      session_idx: (B,) long — one session per window; broadcast to per-frame.
    """
    B, T = x_triang_d.shape[:2]
    # Per-frame views — flatten the time dim into the batch.
    xd_f = x_triang_d.reshape(B * T, N_KP, 3)
    xs_f = x_triang_s.reshape(B * T, N_KP, 3)
    x2_f = x_2d.reshape(B * T, N_CAM, N_KP, 2)
    xc_f = x_conf.reshape(B * T, N_CAM, N_KP)
    # Repeat session_idx across the time dim (each frame in a window shares the
    # session's calibration).
    sidx_f = session_idx.repeat_interleave(T)

    feat_f = build_features(xd_f, xs_f, x2_f, xc_f,
                             sess_calib_stacked, sidx_f)        # (B*T, 23, 21)
    return feat_f.reshape(B, T, N_KP, 21)


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
                      num_workers=0, pin_memory=True,
                      collate_fn=collate_temporal)


def _apply_inference_guard(pred_last, x_triang_d_last, x_triang_s_last):
    """Match correct_triangulation_refiner's safety:
       - per-sample outlier: any frame coord |.|>OUTLIER_THRESHOLD_MM or
         non-finite -> replace pred with x_triang_d (identity / Procrustes-only).
       - per-kp residual clip to +/- RESIDUAL_CLIP_MM.
    Operates on the LAST-frame slices.
    """
    finite_in = torch.isfinite(x_triang_s_last).reshape(x_triang_s_last.shape[0], -1).all(dim=-1) \
                & torch.isfinite(x_triang_d_last).reshape(x_triang_d_last.shape[0], -1).all(dim=-1)
    max_abs_s = x_triang_s_last.reshape(x_triang_s_last.shape[0], -1).abs().max(dim=-1).values
    max_abs_d = x_triang_d_last.reshape(x_triang_d_last.shape[0], -1).abs().max(dim=-1).values
    inlier = finite_in & (max_abs_s < OUTLIER_THRESHOLD_MM) \
                       & (max_abs_d < OUTLIER_THRESHOLD_MM)

    delta = (pred_last - x_triang_d_last).clamp(min=-RESIDUAL_CLIP_MM,
                                                  max=RESIDUAL_CLIP_MM)
    clipped = x_triang_d_last + delta
    # Outliers fall back to identity (Procrustes-only) DANNCE-space pose.
    inlier3 = inlier.view(-1, 1, 1)
    return torch.where(inlier3, clipped, x_triang_d_last)


def evaluate(model, loader, sess_calib_torch, device):
    """Honest val MSE: apply the inference outlier guard + residual clip on
    predictions so a single broken sample can't dominate the aggregate."""
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_triang_d = batch["x_triang_dannce"].to(device, non_blocking=True)
            x_triang_s = batch["x_triang_sleap"].to(device, non_blocking=True)
            x_2d = batch["x_2d"].to(device, non_blocking=True)
            x_conf = batch["x_conf"].to(device, non_blocking=True)
            y_last = batch["y_dannce_last"].to(device, non_blocking=True)
            session_idx = batch["session_idx"].to(device, non_blocking=True)
            feat = build_features_temporal(x_triang_d, x_triang_s, x_2d, x_conf,
                                            sess_calib_torch, session_idx)
            pred = model(feat)                                  # (B, 23, 3)
            guarded = _apply_inference_guard(pred,
                                              x_triang_d[:, -1, :, :],
                                              x_triang_s[:, -1, :, :])
            sse += ((guarded - y_last) ** 2).sum().item()
            n += y_last.numel()
    return sse / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rats", nargs="+", required=True,
                    choices=["R1", "R2", "R3"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_per_kp_layers", type=int, default=3)
    ap.add_argument("--global_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--bone_weight", type=float, default=0.0,
                    help="bone-length loss weight on the LAST frame; default 0 "
                         "because we saw R3 instability with bone>0 last session")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_residual", type=float, default=60.0)
    ap.add_argument("--max_sessions_per_rat", type=int, default=None)
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_ckpt", type=str, default=None,
                    help="path to a temporal_triangulation_refiner ckpt to init from")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = session_split_multi(args.rats, seed=args.seed)
    print("Split sizes:")
    for which in ("train", "val", "test"):
        sizes = {r: len(splits[which][r]) for r in args.rats}
        print(f"  {which}: {sizes}", flush=True)

    print("\nLoading train sessions...", flush=True)
    train_base = Paired2DTrainDataset(
        splits["train"], max_residual=args.max_residual,
        max_sessions_per_rat=args.max_sessions_per_rat)
    print("\nLoading val sessions...", flush=True)
    val_base = Paired2DTrainDataset(
        splits["val"], max_residual=args.max_residual,
        max_sessions_per_rat=args.max_sessions_per_rat)

    train_ds = Paired2DTemporalDataset(train_base, ctx=args.ctx)
    val_ds = Paired2DTemporalDataset(val_base, ctx=args.ctx)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_ds)} val={len(val_ds)}")
    print(f"\nTemporal windows: train={len(train_ds):,}  val={len(val_ds):,}  "
          f"ctx={args.ctx}", flush=True)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)

    train_cal = stack_session_calibs(
        [s["calibration"] for s in train_base.sessions], device)
    val_cal = stack_session_calibs(
        [s["calibration"] for s in val_base.sessions], device)

    if args.init_ckpt:
        init_ck = torch.load(args.init_ckpt, map_location="cpu",
                             weights_only=False)
        if init_ck.get("model_name") != "temporal_triangulation_refiner":
            raise ValueError(f"--init_ckpt model_name={init_ck.get('model_name')!r}, "
                             "expected 'temporal_triangulation_refiner'")
        args.ctx = init_ck.get("ctx", args.ctx)
        args.hidden = init_ck.get("hidden", args.hidden)
        args.n_per_kp_layers = init_ck.get("n_per_kp_layers", args.n_per_kp_layers)
        args.global_dim = init_ck.get("global_dim", args.global_dim)
        args.dropout = init_ck.get("dropout", args.dropout)

    model = build_model("temporal_triangulation_refiner",
                        ctx=args.ctx,
                        hidden=args.hidden,
                        n_per_kp_layers=args.n_per_kp_layers,
                        global_dim=args.global_dim,
                        dropout=args.dropout).to(device)
    if args.init_ckpt:
        model.load_state_dict(init_ck["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: temporal_triangulation_refiner  params: {n_params:,}  "
          f"device: {device}", flush=True)

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
            y_last = batch["y_dannce_last"].to(device, non_blocking=True)
            session_idx = batch["session_idx"].to(device, non_blocking=True)

            feat = build_features_temporal(x_triang_d, x_triang_s, x_2d, x_conf,
                                            train_cal, session_idx)
            pred = model(feat)                                  # (B, 23, 3)
            loss_mse = ((pred - y_last) ** 2).mean()
            loss = loss_mse
            if args.bone_weight > 0:
                loss = loss + args.bone_weight * bone_length_loss(
                    pred, y_last, EDGES)

            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                 max_norm=args.grad_clip)
            opt.step()
            train_sse += ((pred.detach() - y_last) ** 2).sum().item()
            train_n += y_last.numel()
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
        "model_name": "temporal_triangulation_refiner",
        "ctx": args.ctx,
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
        "residual_clip_mm": RESIDUAL_CLIP_MM,
        "session_residuals_train": train_base.session_residuals,
        "session_residuals_val": val_base.session_residuals,
        "init_ckpt": args.init_ckpt,
        "lr": args.lr,
        "resid_norm_px": RESID_NORM_PX,
        "video_w": VIDEO_W, "video_h": VIDEO_H,
    }
    torch.save(save, ckpt)
    print(f"\nsaved {ckpt}  best_val_mse={best_val:.3f}", flush=True)


if __name__ == "__main__":
    main()
