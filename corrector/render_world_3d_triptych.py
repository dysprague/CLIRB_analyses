"""Render a 3D skeleton triptych: SLEAP raw, SLEAP corrected, DANNCE side-by-
side on the same 3D axes with an X-axis offset between each, so all three
animate together but never overlap.

Each skeleton is centered to its per-frame mean (so the rat doesn't translate
across the arena) and then shifted by 0 / +spacing / +2*spacing along X.

Camera view: fixed isometric (azim=45, elev=30). Axis limits fixed across all
frames from the windowed data so the bones don't pop in/out of view.

Usage:
    python -m corrector.render_world_3d_triptych \\
        --ckpt corrector/checkpoints/R1R2R3_world_temporal_mlp.pt \\
        --rat R3 --session 2026_02_10_1 --n_frames 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  side-effect register
import numpy as np
import torch
import cv2

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from config import EDGES

from corrector.data_world import load_paired_world, SLEAP_HZ
from corrector.models import build_model
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.render_world_overlay import correct_world

OUT_DIR = _THIS.parent / "videos"
OUT_DIR.mkdir(exist_ok=True)


def _center_per_frame(arr: np.ndarray) -> np.ndarray:
    """Subtract per-frame mean keypoint position. arr: (T, 23, 3) -> (T, 23, 3)."""
    return arr - arr.mean(axis=1, keepdims=True)


def _build_corrected(ck, model, name, rat, session, sl, dn, device,
                      calibration_minutes=5.0, calibration_n_sample=1000,
                      smooth_size=11):
    """Returns sl_corr_in_sleap aligned to the SLEAP timeline (shape == sl)."""
    if name in ("triangulation_refiner", "temporal_triangulation_refiner"):
        raise NotImplementedError(
            "3D triptych is wired for 3D-input and 'temporal_mlp_2d*' "
            "correctors only; pure saved-2D-Procrustes refiners aren't.")
    # Procrustes on (sl, dn) — shared by all 3D-input + temporal_mlp_2d* models.
    idx = calibration_indices(len(sl), calibration_minutes, SLEAP_HZ,
                               calibration_n_sample, seed=0)
    tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
    print(f"  Procrustes residual={tx['residual']:.2f}  scale={tx['s']:.4f}  "
          f"z_flipped={tx['z_flipped']}", flush=True)
    sl_aligned_dn = tx["apply"](sl).astype(np.float32)
    if name == "temporal_mlp_2d":
        from corrector.evaluate_all import correct_temporal_mlp_2d
        sl_corr_dannce = correct_temporal_mlp_2d(
            model, rat, session, sl_aligned_dn, device,
            ctx=ck.get("ctx", 5))
    elif name == "temporal_mlp_2d_reproj":
        from corrector.evaluate_all import correct_temporal_mlp_2d_reproj
        sl_corr_dannce = correct_temporal_mlp_2d_reproj(
            model, rat, session, sl_aligned_dn, device,
            ctx=ck.get("ctx", 5),
            smooth_size=smooth_size)
    else:
        eval_ctx = ck.get("ctx", 1)
        eval_vel_acc = (name == "velacc_mlp")
        sl_corr_dannce = correct_world(model, sl_aligned_dn, device,
                                         ctx=eval_ctx, vel_acc=eval_vel_acc)
    sl_corr_in_sleap = tx["apply_inverse"](sl_corr_dannce).astype(np.float32)
    return sl_corr_in_sleap, tx


def _draw_skeleton_3d(ax, pts, color, edges=EDGES, marker_size=14,
                        line_width=1.6):
    """pts: (23, 3). Draws bones + joints on the given mpl 3D axes."""
    valid = np.isfinite(pts).all(axis=1)
    for i, j in edges:
        if valid[i] and valid[j]:
            ax.plot([pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    [pts[i, 2], pts[j, 2]],
                    color=color, linewidth=line_width, solid_capstyle="round")
    vp = pts[valid]
    if len(vp):
        ax.scatter(vp[:, 0], vp[:, 1], vp[:, 2], color=color, s=marker_size,
                   depthshade=False)


def render(ckpt_path: str, rat: str, session: str, start_frame: int,
           n_frames: int, fps: int, spacing_mm: float, azim: float, elev: float,
           out_subdir_name: str | None, smooth_size: int = 11):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ck["model_name"]

    # Model build (same dispatch as render_world_overlay for 3D-input models)
    if name == "temporal_mlp":
        model_kwargs = dict(hidden=ck.get("hidden", 128),
                            n_hidden_layers=ck.get("n_hidden_layers", 2),
                            ctx=ck.get("ctx", 5))
    elif name == "velacc_mlp":
        model_kwargs = dict(hidden=ck.get("hidden", 128),
                            n_hidden_layers=ck.get("n_hidden_layers", 2))
    elif name == "mlp":
        model_kwargs = dict(hidden=ck.get("hidden", 128),
                            n_hidden_layers=ck.get("n_hidden_layers", 2))
    elif name == "linear":
        model_kwargs = {}
    elif name == "perrat_head":
        model_kwargs = dict(base_ckpt=ck.get("base_ckpt"),
                            hidden=ck.get("hidden", 64),
                            n_hidden_layers=ck.get("n_hidden_layers", 2))
    elif name in ("temporal_mlp_2d", "temporal_mlp_2d_reproj"):
        model_kwargs = dict(ctx=ck.get("ctx", 5),
                            hidden=ck.get("hidden", 128),
                            n_hidden_layers=ck.get("n_hidden_layers", 2),
                            dropout=ck.get("dropout", 0.0))
    else:
        raise NotImplementedError(f"3D triptych: model {name!r} not wired")
    model = build_model(name, **model_kwargs)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()

    print(f"Loading {rat}/{session} ...", flush=True)
    sl, dn = load_paired_world(rat, session)
    sl_corr_in_sleap, tx = _build_corrected(ck, model, name, rat, session,
                                              sl, dn, device,
                                              smooth_size=smooth_size)
    dn_in_sleap = tx["apply_inverse"](dn).astype(np.float32)

    end_frame = min(start_frame + n_frames, len(sl), len(dn))
    rng = slice(start_frame, end_frame)
    raw_c = _center_per_frame(sl[rng].astype(np.float32))
    cor_c = _center_per_frame(sl_corr_in_sleap[rng])
    dnn_c = _center_per_frame(dn_in_sleap[rng])
    # SLEAP world convention is z<0 = in-front-of-camera; after centering this
    # leaves the spine pointing down in matplotlib's default z-up axes. Negate
    # z so the rat sits upright (head/spine top, paws bottom).
    for a in (raw_c, cor_c, dnn_c):
        a[..., 2] *= -1.0
    n = end_frame - start_frame
    print(f"Window: {n} frames, spacing={spacing_mm:.0f} mm  (z-flipped)",
          flush=True)

    # Offsets along +X.
    offsets = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([spacing_mm, 0.0, 0.0], dtype=np.float32),
        np.array([2 * spacing_mm, 0.0, 0.0], dtype=np.float32),
    ]
    arrs_centered = [raw_c, cor_c, dnn_c]
    arrs_offset = [a + o for a, o in zip(arrs_centered, offsets)]

    # Tight axis limits from the windowed data (percentile-based crop so a
    # single far-flung outlier keypoint doesn't shrink the rest of the frame).
    flat = np.concatenate([a.reshape(-1, 3) for a in arrs_offset], axis=0)
    flat = flat[np.isfinite(flat).all(axis=1)]
    p_lo = np.percentile(flat, 1.0, axis=0)
    p_hi = np.percentile(flat, 99.0, axis=0)
    # X needs to span all three skeletons; widen X to absolute min/max but use
    # percentile on Y and Z so we crop tightly around the rat body.
    x_lo = flat[:, 0].min()
    x_hi = flat[:, 0].max()
    pad_x = (x_hi - x_lo) * 0.04
    pad_yz = 30.0
    lo = np.array([x_lo - pad_x, p_lo[1] - pad_yz, p_lo[2] - pad_yz],
                  dtype=np.float32)
    hi = np.array([x_hi + pad_x, p_hi[1] + pad_yz, p_hi[2] + pad_yz],
                  dtype=np.float32)
    print(f"axis lims: X[{lo[0]:.0f}, {hi[0]:.0f}]  "
          f"Y[{lo[1]:.0f}, {hi[1]:.0f}]  Z[{lo[2]:.0f}, {hi[2]:.0f}]", flush=True)

    run_tag = Path(ckpt_path).stem
    sub_tag = run_tag if out_subdir_name is None else out_subdir_name
    out_subdir = OUT_DIR / sub_tag
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = (out_subdir /
                f"{rat}_{session}_3dtriptych_f{start_frame}-{end_frame}.mp4")

    # Render with a single matplotlib figure, rasterized to numpy per frame.
    # We want the skeletons to fill the frame. matplotlib's 3D axes reserve
    # default padding for ticks even when hidden, so we shrink the figure
    # vertically and use an aggressive subplots_adjust to push the axes to
    # the edges.
    fig_w_in, fig_h_in, dpi = 14.0, 4.5, 130
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    x_span = hi[0] - lo[0]
    y_span = hi[1] - lo[1]
    z_span = hi[2] - lo[2]
    # Cap X:Y and X:Z to at most 3:1 so each skeleton retains depth.
    yz_max = max(y_span, z_span)
    box_x = min(x_span, 3.0 * yz_max)
    ax.set_box_aspect((box_x, y_span, z_span))
    ax.view_init(elev=elev, azim=azim)

    # Push the 3D axes hard to the figure edges and hide every axis decoration
    # so the skeletons own the frame.
    fig.subplots_adjust(left=-0.15, right=1.15, top=1.20, bottom=-0.20)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    # Hide the pane background and axis lines.
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))

    # Determine the final video size from the figure canvas.
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(str(out_path),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer: {out_path}")
    print(f"Rendering {out_path.name} ({w}x{h}) ...", flush=True)

    # Pretty colors: cyan (raw), green (corrected), magenta (DANNCE).
    color_raw = (0.0, 0.85, 1.0)
    color_cor = (0.2, 0.9, 0.2)
    color_dn = (1.0, 0.2, 0.85)
    labels_x = [0 + spacing_mm * k for k in range(3)]
    label_y = lo[1] - (hi[1] - lo[1]) * 0.05
    label_z = hi[2] + (hi[2] - lo[2]) * 0.02
    label_names = ["SLEAP raw", "SLEAP corrected", "DANNCE"]
    label_colors = [color_raw, color_cor, color_dn]

    t0 = time.time()
    for fi in range(n):
        ax.cla()
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((box_x, y_span, z_span))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_visible(False)
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))

        for a, c in zip(arrs_offset, [color_raw, color_cor, color_dn]):
            _draw_skeleton_3d(ax, a[fi], c)

        for x, name_lbl, c in zip(labels_x, label_names, label_colors):
            ax.text(x, label_y, label_z, name_lbl, color=c, fontsize=11,
                    ha="center", weight="bold")
        ax.set_title(f"{rat}/{session}  frame {start_frame + fi}",
                     fontsize=11, pad=4)

        fig.canvas.draw()
        # canvas to numpy
        try:
            img = np.asarray(fig.canvas.buffer_rgba())
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        except AttributeError:
            # older matplotlib
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(h, w, 3)[:, :, ::-1]
        writer.write(img)
        if (fi + 1) % 200 == 0 or fi == n - 1:
            print(f"  {fi + 1}/{n} ({time.time() - t0:.1f}s)", flush=True)
    writer.release()
    plt.close(fig)
    print(f"saved {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--spacing_mm", type=float, default=350.0,
                    help="X-axis offset between consecutive skeletons")
    ap.add_argument("--azim", type=float, default=45.0)
    ap.add_argument("--elev", type=float, default=30.0)
    ap.add_argument("--out_subdir", type=str, default=None)
    ap.add_argument("--smooth_size", type=int, default=11,
                    help="median-filter size inside the "
                         "temporal_mlp_2d_reproj corrector. Default 11; pass "
                         "15 for the production-recommended smoothing.")
    args = ap.parse_args()
    render(args.ckpt, args.rat, args.session, args.start_frame,
           args.n_frames, args.fps, args.spacing_mm,
           args.azim, args.elev, args.out_subdir,
           smooth_size=args.smooth_size)


if __name__ == "__main__":
    main()
