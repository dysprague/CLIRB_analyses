"""
Render a long video with overlay panels on top and PC1/PC2 trajectories on the
bottom.

Top-left   : raw Camera0 frame + raw SLEAP (cyan) + DANNCE (magenta)
Top-right  : raw Camera0 frame + corrected SLEAP (cyan) + DANNCE (magenta)
Bottom-left: PC1 trajectory  with raw SLEAP, raw DANNCE, corrected SLEAP curves
             plus a vertical cursor at the current frame.
Bottom-right: PC2 trajectory, same three curves.

Usage:
    python -m corrector.render_long_combined \
        --ckpt corrector/checkpoints/R2R3_world_mlp.pt \
        --rat R3 --session 2026_02_06_1 --camera 0 --n_frames 10000
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from config import EDGES, sleap_path, calibration_path
from data_io import load_template
from projection import project_3d_to_2d_for_camera

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.evaluate_world import RAT_TEMPLATE, project_to_template_pcs
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model

OUT_DIR = _THIS.parent / "videos"
OUT_DIR.mkdir(exist_ok=True)


def correct_world(model, x, device, batch=8192, ctx: int = 1):
    out = np.empty_like(x, dtype=np.float32)
    if ctx <= 1:
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xt = torch.from_numpy(x[i:i + batch].astype(np.float32)).to(device)
                out[i:i + batch] = model(xt).cpu().numpy()
        return out
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


def draw_skel(img, pts2d, color, edges=EDGES, radius=4, thickness=1):
    pts = pts2d.astype(np.int32)
    h, w = img.shape[:2]
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    for e in edges:
        if valid[e[0]] and valid[e[1]]:
            cv2.line(img, tuple(pts[e[0]]), tuple(pts[e[1]]), color, thickness, cv2.LINE_AA)
    for i in range(len(pts)):
        if valid[i]:
            cv2.circle(img, tuple(pts[i]), radius, color, -1, cv2.LINE_AA)


def make_pc_panels(pc_raw, pc_corr, pc_dn, panel_w, panel_h):
    """Render two static (no cursor) PC panels at given panel_w x panel_h pixels.

    Returns (panel_pc1_img, panel_pc2_img) as uint8 BGR arrays. We add the
    cursor per-frame in fast cv2 code instead of redrawing matplotlib for
    every frame.
    """
    n = len(pc_raw); t = np.arange(n) / SLEAP_HZ
    panels = []
    titles = ("PC1", "PC2")
    for j, title in enumerate(titles):
        dpi = 100
        fig = plt.figure(figsize=(panel_w / dpi, panel_h / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.plot(t, pc_raw[:, j], color="#1f77b4", lw=0.7, label="SLEAP raw", alpha=0.85)
        ax.plot(t, pc_corr[:, j], color="#2ca02c", lw=0.7, label="SLEAP corrected", alpha=0.85)
        ax.plot(t, pc_dn[:, j], color="#d62728", lw=0.7, label="DANNCE", alpha=0.85)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
        ax.set_xlim(t[0], t[-1])
        fig.tight_layout(pad=0.7)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())[:, :, :3]
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
        if bgr.shape[0] != panel_h or bgr.shape[1] != panel_w:
            bgr = cv2.resize(bgr, (panel_w, panel_h))
        panels.append(bgr)
        plt.close(fig)
    return panels[0], panels[1], t


def overlay_cursor(panel, time_arr, current_t, color=(0, 255, 255)):
    """Draw a vertical line on a precomputed plot panel at x corresponding to current_t."""
    h, w = panel.shape[:2]
    # Approximate axes margin; matplotlib default tight_layout leaves ~12% margin
    # on the left and ~3% on the right at this dpi. Use a single ratio to map
    # time → pixel x. Recompute from the full time range.
    t0 = time_arr[0]; t1 = time_arr[-1]
    x_left_frac = 0.12; x_right_frac = 0.97
    px = int(round((x_left_frac + (current_t - t0) / (t1 - t0) *
                    (x_right_frac - x_left_frac)) * w))
    px = max(0, min(w - 1, px))
    out = panel.copy()
    cv2.line(out, (px, 0), (px, h - 1), color, 1, cv2.LINE_AA)
    return out


def render(ckpt_path: str, rat: str, session: str,
           start_frame: int, n_frames: int, camera: int,
           fps: int = 20, calibration_minutes: float = 5.0,
           calibration_n_sample: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_kwargs = dict(hidden=ck.get("hidden", 128),
                        n_hidden_layers=ck.get("n_hidden_layers", 2))
    if ck["model_name"] == "temporal_mlp":
        model_kwargs["ctx"] = ck.get("ctx", 5)
    model = build_model(ck["model_name"], **model_kwargs)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    eval_ctx = ck.get("ctx", 1)

    print(f"Loading {rat}/{session} ...", flush=True)
    sl, dn = load_paired_world(rat, session)

    idx = calibration_indices(len(sl), calibration_minutes, SLEAP_HZ,
                               calibration_n_sample, seed=0)
    tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
    print(f"  Procrustes residual={tx['residual']:.2f}  scale={tx['s']:.4f}",
          flush=True)

    sl_aligned = tx["apply"](sl).astype(np.float32)
    sl_corr_dannce_space = correct_world(model, sl_aligned, device, ctx=eval_ctx)
    dn_in_sleap = tx["apply_inverse"](dn).astype(np.float32)
    sl_corr_in_sleap = tx["apply_inverse"](sl_corr_dannce_space).astype(np.float32)

    cal_folder = calibration_path(rat, session)
    end_frame = min(start_frame + n_frames, len(sl), len(dn))
    rng = slice(start_frame, end_frame)
    print(f"Projecting {end_frame - start_frame} frames to Camera{camera}...",
          flush=True)
    sleap_2d = project_3d_to_2d_for_camera(sl[rng], cal_folder, camera_idx=camera)
    sleap_corr_2d = project_3d_to_2d_for_camera(
        sl_corr_in_sleap[rng], cal_folder, camera_idx=camera)
    dannce_2d = project_3d_to_2d_for_camera(
        dn_in_sleap[rng], cal_folder, camera_idx=camera)

    # PC trajectories for the windowed range
    tmpl = dict(load_template(rat, RAT_TEMPLATE[rat]))
    pcs_to_use = tmpl["pcs_to_use"].ravel().astype(int)
    pc_raw = project_to_template_pcs(sl[rng], tmpl, pcs_to_use)
    pc_corr = project_to_template_pcs(sl_corr_in_sleap[rng], tmpl, pcs_to_use)
    pc_dn = project_to_template_pcs(dn_in_sleap[rng], tmpl, pcs_to_use)

    sp = sleap_path(rat, session)
    cam_name = f"Camera{camera}"
    video_file = Path(sp) / cam_name / "0.mp4"
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_file}")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Layout: top is two video panels (vid_w x vid_h each), bottom is two PC
    # panels (vid_w x ~vid_h*0.5 each).
    top_h = vid_h
    bot_h = int(round(vid_h * 0.55))
    out_w = vid_w * 2
    out_h = top_h + bot_h

    run_tag = Path(ckpt_path).stem
    out_subdir = OUT_DIR / run_tag
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{rat}_{session}_cam{camera}_long_f{start_frame}-{end_frame}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_path}")

    print("Pre-rendering PC panels...", flush=True)
    panel_pc1, panel_pc2, t_arr = make_pc_panels(pc_raw, pc_corr, pc_dn,
                                                  vid_w, bot_h)

    CYAN = (255, 255, 0); MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255); YELLOW = (0, 255, 255)
    print(f"Rendering {out_path.name} ({end_frame - start_frame} frames)...",
          flush=True)
    t0 = time.time()
    n = end_frame - start_frame
    for fi in range(n):
        ok, frame = cap.read()
        if not ok: break
        left = frame.copy(); right = frame.copy()
        draw_skel(left, sleap_2d[fi], CYAN)
        draw_skel(left, dannce_2d[fi], MAGENTA)
        draw_skel(right, sleap_corr_2d[fi], CYAN)
        draw_skel(right, dannce_2d[fi], MAGENTA)
        cv2.putText(left, "SLEAP raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN, 2, cv2.LINE_AA)
        cv2.putText(left, "DANNCE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(right, "SLEAP corrected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN, 2, cv2.LINE_AA)
        cv2.putText(right, "DANNCE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(left, f"frame {start_frame + fi}", (10, vid_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
        cv2.putText(right, f"{rat}/{session}  Camera{camera}",
                    (10, vid_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2,
                    cv2.LINE_AA)
        top = np.hstack([left, right])

        # Bottom: PC panels with current-time cursor
        cur_t = t_arr[fi] if fi < len(t_arr) else t_arr[-1]
        bot_left = overlay_cursor(panel_pc1, t_arr, cur_t, color=(0, 255, 255))
        bot_right = overlay_cursor(panel_pc2, t_arr, cur_t, color=(0, 255, 255))
        bot = np.hstack([bot_left, bot_right])

        writer.write(np.vstack([top, bot]))
        if (fi + 1) % 500 == 0 or fi == n - 1:
            print(f"  {fi + 1}/{n} ({time.time() - t0:.1f}s)", flush=True)
    writer.release(); cap.release()
    print(f"saved {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=10000)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    render(args.ckpt, args.rat, args.session, args.start_frame,
           args.n_frames, args.camera, args.fps)


if __name__ == "__main__":
    main()
