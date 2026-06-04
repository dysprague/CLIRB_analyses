"""Per-frame inference benchmark for temporal_mlp_2d_reproj.

Online inference path (per video frame at ~50 ms cadence) needs:
  1. project_3d_to_2d on 3 cams x 23 kp (small reprojection)
  2. compute (detected - reproj) / RESID_NORM_PX, normalize 2D, build feature vec
  3. forward pass through the MLP

This script measures each step on CPU and GPU, with batch size 1 to simulate
the online case. Also measures a batched run as a sanity check.

Usage:
    python scripts/bench_reproj_v1.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from corrector.data_world_2d_from_saved import load_session_2d
from corrector.data_world_2d import reproject_all_cams
from corrector.models import build_model
from corrector.train_temporal_mlp_2d_reproj import (
    VIDEO_W, VIDEO_H, N_CAM, N_KP, RESID_NORM_PX,
    OUTLIER_THRESHOLD_MM, RESIDUAL_CLIP_MM)


def time_block(fn, n_warmup=10, n_runs=200):
    """Returns mean/median latency in microseconds."""
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples = np.array(samples) * 1e6  # to us
    return float(np.mean(samples)), float(np.median(samples)), float(np.std(samples))


def main():
    ckpt_path = "corrector/checkpoints/R1R2R3_temporal_mlp_2d_reproj_v1.pt"
    rat = "R2"
    session = "2026_02_09_1"
    ctx = 5

    print(f"Loading {ckpt_path}...", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = ck["model_name"]
    print(f"model: {name}, ctx={ck.get('ctx')}, hidden={ck.get('hidden')}")

    print(f"\nLoading session {rat}/{session}...", flush=True)
    sd = load_session_2d(rat, session, smooth_dannce=True)
    print(f"  P={len(sd.x_triang_3d)} processed frames")

    # Pick a representative frame.
    P = len(sd.x_triang_3d)
    fi = P // 2
    # Pull a (ctx, 23, 3) window in SLEAP world (un-aligned — for bench we just
    # want the shape).
    pose_win_sleap = sd.x_triang_3d[max(0, fi - ctx + 1): fi + 1]
    # Pad if short
    while len(pose_win_sleap) < ctx:
        pose_win_sleap = np.concatenate([pose_win_sleap[:1], pose_win_sleap], 0)
    pose_win_sleap = pose_win_sleap.astype(np.float32)
    detected = sd.x_2d[fi].astype(np.float32)         # (3, 23, 2)
    conf = sd.x_conf[fi].astype(np.float32)            # (3, 23)
    calib = sd.calibration
    triang_curr = sd.x_triang_3d[fi].astype(np.float32)  # (23, 3)

    # For inference, we need pose_win in DANNCE-world (Procrustes-applied). For
    # the benchmark we don't have a Procrustes ready, so we use the raw window
    # — the operations time the same way regardless.

    # --- Step 1: reprojection (numpy, since calib is numpy dicts).
    def step_reproj():
        # Project a single frame's triangulated 3D through 3 cams.
        return reproject_all_cams(triang_curr[None, :, :], calib)
    mean_us, med_us, std_us = time_block(step_reproj, n_runs=500)
    print(f"\n[reproject_all_cams, single frame]  "
          f"mean={mean_us:.1f}us  med={med_us:.1f}us  std={std_us:.1f}us")
    reproj_curr = step_reproj()           # (1, 3, 23, 2)

    # --- Step 2: build features (CPU-side ops).
    def step_features():
        vis = ((conf > 0) & np.isfinite(detected).all(axis=-1)
               & np.isfinite(reproj_curr[0]).all(axis=-1))
        resid = np.where(vis[..., None],
                         (detected - reproj_curr[0]) / RESID_NORM_PX,
                         0.0).astype(np.float32)
        det_safe = np.where(vis[..., None], detected, 0.0).astype(np.float32)
        conf_safe = np.where(vis, conf, 0.0).astype(np.float32)
        scale = np.array([VIDEO_W, VIDEO_H], dtype=np.float32).reshape(1, 1, 2)
        xy_norm = (det_safe / scale - 0.5) * 2.0
        return xy_norm, conf_safe, vis.astype(np.float32), resid
    mean_us, med_us, std_us = time_block(step_features, n_runs=500)
    print(f"[feature build (numpy)]              "
          f"mean={mean_us:.1f}us  med={med_us:.1f}us  std={std_us:.1f}us")
    xy_norm, conf_safe, vis_f, resid = step_features()

    # --- Step 3: model forward pass, CPU and GPU.
    model_kwargs = dict(ctx=ck.get("ctx", 5),
                        hidden=ck.get("hidden", 128),
                        n_hidden_layers=ck.get("n_hidden_layers", 2),
                        dropout=0.0)
    model = build_model(name, **model_kwargs)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel params: {n_params:,}")

    pose_t = torch.from_numpy(pose_win_sleap).unsqueeze(0)        # (1, ctx, 23, 3)
    x2_t = torch.from_numpy(xy_norm).unsqueeze(0)                 # (1, 3, 23, 2)
    xc_t = torch.from_numpy(conf_safe).unsqueeze(0)               # (1, 3, 23)
    xv_t = torch.from_numpy(vis_f).unsqueeze(0)                   # (1, 3, 23)
    xr_t = torch.from_numpy(resid).unsqueeze(0)                   # (1, 3, 23, 2)

    @torch.no_grad()
    def step_forward_cpu():
        return model(pose_t, x2_t, xc_t, xv_t, xr_t)
    mean_us, med_us, std_us = time_block(step_forward_cpu, n_warmup=20, n_runs=300)
    print(f"[forward CPU, B=1]                   "
          f"mean={mean_us:.1f}us  med={med_us:.1f}us  std={std_us:.1f}us")

    if torch.cuda.is_available():
        import copy
        device = torch.device("cuda")
        model_cpu = copy.deepcopy(model)  # keep a CPU model for the CPU e2e block
        model_g = model.to(device)
        model = model_cpu  # rebind so subsequent CPU calls work
        pose_g = pose_t.to(device)
        x2_g = x2_t.to(device); xc_g = xc_t.to(device)
        xv_g = xv_t.to(device); xr_g = xr_t.to(device)

        @torch.no_grad()
        def step_forward_gpu():
            out = model_g(pose_g, x2_g, xc_g, xv_g, xr_g)
            torch.cuda.synchronize()
            return out

        mean_us, med_us, std_us = time_block(step_forward_gpu, n_warmup=50,
                                              n_runs=500)
        print(f"[forward GPU, B=1, with sync]        "
              f"mean={mean_us:.1f}us  med={med_us:.1f}us  std={std_us:.1f}us")

        # Bonus: also test B=64 (batched) on GPU for throughput context.
        pose_b = pose_g.expand(64, -1, -1, -1).contiguous()
        x2_b = x2_g.expand(64, -1, -1, -1).contiguous()
        xc_b = xc_g.expand(64, -1, -1).contiguous()
        xv_b = xv_g.expand(64, -1, -1).contiguous()
        xr_b = xr_g.expand(64, -1, -1, -1).contiguous()

        @torch.no_grad()
        def step_forward_gpu_b64():
            out = model_g(pose_b, x2_b, xc_b, xv_b, xr_b)
            torch.cuda.synchronize()
            return out
        mean_us, med_us, std_us = time_block(step_forward_gpu_b64, n_warmup=30,
                                              n_runs=200)
        print(f"[forward GPU, B=64, per-frame]       "
              f"mean={mean_us/64:.1f}us  med={med_us/64:.1f}us  "
              f"(total batch mean {mean_us:.1f}us)")

    # --- Total end-to-end estimate (single frame, CPU pipeline + CPU model).
    print("\n--- End-to-end single-frame estimate (CPU) ---")

    def step_e2e_cpu():
        r = reproject_all_cams(triang_curr[None, :, :], calib)
        vis = ((conf > 0) & np.isfinite(detected).all(axis=-1)
               & np.isfinite(r[0]).all(axis=-1))
        resid = np.where(vis[..., None],
                         (detected - r[0]) / RESID_NORM_PX,
                         0.0).astype(np.float32)
        det_safe = np.where(vis[..., None], detected, 0.0).astype(np.float32)
        conf_safe = np.where(vis, conf, 0.0).astype(np.float32)
        scale = np.array([VIDEO_W, VIDEO_H], dtype=np.float32).reshape(1, 1, 2)
        xy_norm = (det_safe / scale - 0.5) * 2.0
        pose_t = torch.from_numpy(pose_win_sleap).unsqueeze(0)
        x2_t = torch.from_numpy(xy_norm).unsqueeze(0)
        xc_t = torch.from_numpy(conf_safe).unsqueeze(0)
        xv_t = torch.from_numpy(vis.astype(np.float32)).unsqueeze(0)
        xr_t = torch.from_numpy(resid).unsqueeze(0)
        with torch.no_grad():
            return model(pose_t, x2_t, xc_t, xv_t, xr_t)
    mean_us, med_us, std_us = time_block(step_e2e_cpu, n_warmup=20, n_runs=200)
    print(f"[end-to-end CPU, B=1]                "
          f"mean={mean_us:.1f}us ({mean_us/1000:.3f} ms)  "
          f"med={med_us:.1f}us ({med_us/1000:.3f} ms)")

    if torch.cuda.is_available():
        def step_e2e_gpu():
            r = reproject_all_cams(triang_curr[None, :, :], calib)
            vis = ((conf > 0) & np.isfinite(detected).all(axis=-1)
                   & np.isfinite(r[0]).all(axis=-1))
            resid = np.where(vis[..., None],
                             (detected - r[0]) / RESID_NORM_PX,
                             0.0).astype(np.float32)
            det_safe = np.where(vis[..., None], detected, 0.0).astype(np.float32)
            conf_safe = np.where(vis, conf, 0.0).astype(np.float32)
            scale = np.array([VIDEO_W, VIDEO_H], dtype=np.float32).reshape(1, 1, 2)
            xy_norm = (det_safe / scale - 0.5) * 2.0
            pose_t = torch.from_numpy(pose_win_sleap).unsqueeze(0).to(device)
            x2_t = torch.from_numpy(xy_norm).unsqueeze(0).to(device)
            xc_t = torch.from_numpy(conf_safe).unsqueeze(0).to(device)
            xv_t = torch.from_numpy(vis.astype(np.float32)).unsqueeze(0).to(device)
            xr_t = torch.from_numpy(resid).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model_g(pose_t, x2_t, xc_t, xv_t, xr_t)
                torch.cuda.synchronize()
            return out
        mean_us, med_us, std_us = time_block(step_e2e_gpu, n_warmup=30, n_runs=200)
        print(f"[end-to-end GPU, B=1, incl. h2d sync]"
              f" mean={mean_us:.1f}us ({mean_us/1000:.3f} ms)  "
              f"med={med_us:.1f}us ({med_us/1000:.3f} ms)")

    print("\nSLEAP budget: ~30 ms per frame (production cadence).")


if __name__ == "__main__":
    main()
