"""
Render world-space corrector videos (and a paired PC trajectory plot).

Pipeline:
  1. Load raw SLEAP_world and DANNCE_world (with the standard median filters).
  2. Fit Procrustes on the first 5 min calibration epoch.
  3. SLEAP_aligned = Procrustes(SLEAP) — pre-correction baseline (right side
     of the frame compares this against DANNCE).
  4. SLEAP_corrected = corrector(SLEAP_aligned) — what the network produces.
  5. To project onto Camera0 (a SLEAP camera), inverse-Procrustes both
     DANNCE and SLEAP_corrected back into SLEAP world space, project with
     SLEAP calibration. SLEAP_raw is projected directly.

Output:
  Left  panel : raw SLEAP (cyan)  + DANNCE-in-SLEAP-space (magenta)
  Right panel : SLEAP_corrected-in-SLEAP-space (cyan) + DANNCE-in-SLEAP-space (magenta)

Also writes a side-by-side PC trajectory plot for the same window.

Usage:
    python -m corrector.render_world_overlay \
        --ckpt corrector/checkpoints/R2R3_world_mlp.pt \
        --rat R3 --session 2026_02_06_1 --camera 0 --n_frames 1000
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

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from config import EDGES, sleap_path, calibration_path, DATA_ROOT
from data_io import load_template
from projection import project_3d_to_2d_for_camera
from skeleton import normalize_skeleton_batch

from corrector.data_world import (SLEAP_HZ, load_paired_world)
from corrector.evaluate_world import RAT_TEMPLATE, project_to_template_pcs
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model

OUT_DIR = _THIS.parent / "videos"
FIG_DIR = _THIS.parent / "figures"
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)


def correct_world(model, x, device, batch=8192, ctx: int = 1,
                  vel_acc: bool = False):
    """Apply a (possibly temporal or vel/acc) corrector. ctx=1 for single-frame
    models; ctx>1 uses a sliding causal window with first-frame padding.
    vel_acc=True feeds [pose, velocity, acceleration] (causal differencing)."""
    out = np.empty_like(x, dtype=np.float32)
    if vel_acc:
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


def render_pc_plot(rat, session, sl_raw, sl_corr_in_sleap, dn_in_sleap,
                   start_frame, n_frames, out_path):
    """Plot first 3 PCs over the same window for the three streams."""
    tmpl = dict(load_template(rat, RAT_TEMPLATE[rat]))
    pcs_to_use = tmpl["pcs_to_use"].ravel().astype(int)
    n_pcs = min(3, len(pcs_to_use))

    end = start_frame + n_frames
    pc_raw = project_to_template_pcs(sl_raw[start_frame:end], tmpl, pcs_to_use)
    pc_corr = project_to_template_pcs(sl_corr_in_sleap[start_frame:end], tmpl, pcs_to_use)
    pc_dn = project_to_template_pcs(dn_in_sleap[start_frame:end], tmpl, pcs_to_use)
    t = np.arange(end - start_frame) / SLEAP_HZ

    fig, axes = plt.subplots(n_pcs, 1, figsize=(10, 2.4 * n_pcs), sharex=True)
    if n_pcs == 1: axes = [axes]
    for j in range(n_pcs):
        ax = axes[j]
        ax.plot(t, pc_raw[:, j], color="#1f77b4", lw=0.9, label="SLEAP raw")
        ax.plot(t, pc_corr[:, j], color="#2ca02c", lw=0.9, label="SLEAP corrected")
        ax.plot(t, pc_dn[:, j], color="#d62728", lw=0.9, label="DANNCE")
        ax.set_ylabel(f"PC{j+1}")
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{rat}/{session} PC trajectories  "
                 f"(frames {start_frame}-{end})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render(ckpt_path: str, rat: str, session: str,
           start_frame: int, n_frames: int, camera: int,
           fps: int = 20, calibration_minutes: float = 5.0,
           calibration_n_sample: int = 1000,
           single_panel: bool = False,
           out_subdir_name: str | None = None,
           smooth_size: int = 11):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ck["model_name"]
    if name == "triangulation_refiner":
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
    eval_ctx = ck.get("ctx", 1)
    eval_vel_acc = (name == "velacc_mlp")

    print(f"Loading {rat}/{session} ...", flush=True)
    sl, dn = load_paired_world(rat, session)

    if name in ("triangulation_refiner", "temporal_triangulation_refiner"):
        # 2D-input path: fit Procrustes on saved triangulated SLEAP (matches
        # training-pipeline alignment exactly) and use the same per-processed-
        # frame correction the evaluator uses.
        from corrector.data_world_2d_from_saved import load_session_2d
        from corrector.evaluate_all import (correct_triangulation_refiner,
                                              correct_temporal_triangulation_refiner)
        sd_for_tx = load_session_2d(rat, session, smooth_dannce=True)
        idx = calibration_indices(len(sd_for_tx.x_triang_3d),
                                   calibration_minutes, SLEAP_HZ,
                                   calibration_n_sample, seed=0)
        tx = fit_procrustes(sd_for_tx.x_triang_3d[idx],
                            sd_for_tx.y_dannce_3d[idx], try_z_flip=True)
        print(f"  Procrustes residual={tx['residual']:.2f}  scale={tx['s']:.4f}  "
              f"z_flipped={tx['z_flipped']}", flush=True)
        sl_aligned = tx["apply"](sl).astype(np.float32)
        if name == "triangulation_refiner":
            sl_corr_dannce_space = correct_triangulation_refiner(
                model, rat, session, sl_aligned, tx, device)
        else:
            sl_corr_dannce_space = correct_temporal_triangulation_refiner(
                model, rat, session, sl_aligned, tx, device,
                ctx=ck.get("ctx", 5))
    else:
        idx = calibration_indices(len(sl), calibration_minutes, SLEAP_HZ,
                                   calibration_n_sample, seed=0)
        tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
        print(f"  Procrustes residual={tx['residual']:.2f}  scale={tx['s']:.4f}  "
              f"z_flipped={tx['z_flipped']}", flush=True)
        sl_aligned = tx["apply"](sl).astype(np.float32)
        if name == "temporal_mlp_2d":
            from corrector.evaluate_all import correct_temporal_mlp_2d
            sl_corr_dannce_space = correct_temporal_mlp_2d(
                model, rat, session, sl_aligned, device,
                ctx=ck.get("ctx", 5))
        elif name == "temporal_mlp_2d_reproj":
            from corrector.evaluate_all import correct_temporal_mlp_2d_reproj
            sl_corr_dannce_space = correct_temporal_mlp_2d_reproj(
                model, rat, session, sl_aligned, device,
                ctx=ck.get("ctx", 5),
                smooth_size=smooth_size)
        else:
            sl_corr_dannce_space = correct_world(model, sl_aligned, device,
                                                  ctx=eval_ctx, vel_acc=eval_vel_acc)

    # Bring everything back into SLEAP world space for projection onto Camera{N}.
    # SLEAP raw is already in SLEAP space (and was median-filtered upstream by
    # load_paired_world).
    # DANNCE -> SLEAP space  via apply_inverse.
    # SLEAP_corrected (lives in DANNCE space because that was the regression
    # target) -> SLEAP space via apply_inverse.
    dn_in_sleap = tx["apply_inverse"](dn).astype(np.float32)
    sl_corr_in_sleap = tx["apply_inverse"](sl_corr_dannce_space).astype(np.float32)

    # Cosmetic: the triangulation_refiner was trained on un-smoothed
    # triangulated SLEAP, so its raw output jitters compared to the velacc
    # baseline (which received median-11 input). Apply the same median-11
    # filter to the corrector output for visual comparison parity.
    if name in ("triangulation_refiner", "temporal_triangulation_refiner"):
        from scipy.ndimage import median_filter as _medfilt
        sl_corr_in_sleap = _medfilt(sl_corr_in_sleap, size=(11, 1, 1)).astype(
            np.float32)

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

    # Video lives in the SMB share — the local 2D cache only contains the
    # keypoint / calibration files, not the videos.
    cam_name = f"Camera{camera}"
    video_file = Path(DATA_ROOT) / rat / session / "sleap" / cam_name / "0.mp4"
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_file}")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_w = vid_w if single_panel else vid_w * 2
    out_h = vid_h
    run_tag = Path(ckpt_path).stem
    sub_tag = run_tag if out_subdir_name is None else out_subdir_name
    out_subdir = OUT_DIR / sub_tag
    fig_subdir = FIG_DIR / sub_tag
    out_subdir.mkdir(parents=True, exist_ok=True)
    fig_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{rat}_{session}_cam{camera}_f{start_frame}-{end_frame}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_path}")

    CYAN = (255, 255, 0); MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255); YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    print(f"Rendering {out_path.name} ...", flush=True)
    t0 = time.time()
    n = end_frame - start_frame
    for fi in range(n):
        ok, frame = cap.read()
        if not ok: break
        if single_panel:
            # Single panel: SLEAP raw (cyan) + SLEAP corrected (green) overlaid.
            panel = frame.copy()
            draw_skel(panel, sleap_2d[fi], CYAN)
            draw_skel(panel, sleap_corr_2d[fi], GREEN)
            cv2.putText(panel, "SLEAP raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, CYAN, 2, cv2.LINE_AA)
            cv2.putText(panel, "SLEAP corrected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
            cv2.putText(panel, f"{rat}/{session}  Camera{camera}",
                        (10, vid_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW,
                        2, cv2.LINE_AA)
            cv2.putText(panel, f"frame {start_frame + fi}", (10, vid_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
            writer.write(panel)
        else:
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
            writer.write(np.hstack([left, right]))
        if (fi + 1) % 200 == 0 or fi == n - 1:
            print(f"  {fi + 1}/{n} ({time.time() - t0:.1f}s)", flush=True)
    writer.release(); cap.release()
    print(f"saved {out_path}", flush=True)

    # Paired PC trajectory plot
    fig_path = fig_subdir / f"{rat}_{session}_f{start_frame}-{end_frame}_pc.png"
    render_pc_plot(rat, session, sl, sl_corr_in_sleap, dn_in_sleap,
                   start_frame, end_frame - start_frame, fig_path)
    print(f"saved {fig_path}", flush=True)
    return out_path, fig_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=1000)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--single_panel", action="store_true",
                    help="overlay SLEAP raw (cyan) and SLEAP corrected (green) "
                         "on the same frame instead of the default two-panel "
                         "raw|corrected vs DANNCE comparison.")
    ap.add_argument("--out_subdir", type=str, default=None,
                    help="override video subdir name (default: ckpt stem)")
    ap.add_argument("--smooth_size", type=int, default=11,
                    help="median-filter size inside the "
                         "temporal_mlp_2d_reproj corrector. Default 11; pass "
                         "15 for the production-recommended smoothing.")
    args = ap.parse_args()
    render(args.ckpt, args.rat, args.session, args.start_frame,
           args.n_frames, args.camera, args.fps,
           single_panel=args.single_panel,
           out_subdir_name=args.out_subdir,
           smooth_size=args.smooth_size)


if __name__ == "__main__":
    main()
