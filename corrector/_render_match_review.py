"""Render match-review videos: stitch 60-frame clips centered on each
template_1 match, with Camera0 video on top and the DANNCE-aligned PC1/PC2
trajectory on the bottom (template +- bounds overlaid). One video per
(session, match_source).

Match sources:
  - sleap : matches from raw SLEAP keypoints against template_1
  - dannce: matches from DANNCE keypoints against template_1 (these are
            the "ground-truth" template matches)

Both sets of indices live on the SLEAP-frame timeline. We use the same
DANNCE-on-SLEAP-timeline trajectory in both cases for the bottom plot
(so the x-axis is comparable).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_io import load_aligned_data, load_template
from exp_utils import run_template_matching
from config import sleap_path

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.evaluate_world import RAT_TEMPLATE, project_to_template_pcs
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.evaluate_all import DEFAULT_BOUNDS, WIN

CLIP = 60   # frames per match clip
HALF = CLIP // 2


def for_template(arr):
    out = arr.copy(); out[:, :, 2] = -out[:, :, 2]; return out


def make_pc_panel_image(pc_dn_window, template_pc, bounds_per_frame,
                         match_frame_in_window, win_w, win_h):
    """Render a PC1/PC2 trajectory panel as a BGR image of size (win_h, win_w).
    pc_dn_window: (CLIP, 2)
    template_pc: (WIN, 2)
    bounds_per_frame: (WIN, 2)  -- half-width per PC
    match_frame_in_window: index inside the clip where the match center sits
    """
    fig, axes = plt.subplots(1, 2, figsize=(win_w / 100, win_h / 100), dpi=100)
    pcs = ["PC1", "PC2"]
    for pi in range(2):
        ax = axes[pi]
        t_clip = np.arange(CLIP)
        ax.plot(t_clip, pc_dn_window[:, pi], color="#1f77b4", lw=1.4,
                label="DANNCE (SLEAP-time)")
        t_tmpl = np.arange(WIN) + match_frame_in_window - WIN + 1
        ax.plot(t_tmpl, template_pc[:, pi], color="#2ca02c", lw=1.6,
                label="template")
        upper = template_pc[:, pi] + bounds_per_frame[:, pi]
        lower = template_pc[:, pi] - bounds_per_frame[:, pi]
        ax.fill_between(t_tmpl, lower, upper, color="#2ca02c", alpha=0.15,
                        label="bounds")
        ax.axvline(match_frame_in_window, color="#d62728", lw=1.0, ls="--",
                   label="match end")
        ax.set_title(pcs[pi], fontsize=9)
        ax.set_xlabel("frame within clip", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.25)
        if pi == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    bgr = cv2.resize(bgr, (win_w, win_h))
    plt.close(fig)
    return bgr


def render(rat, session, source, sleap_matches, dannce_matches,
           pc_dn_full, template_pc, bounds_per_frame, eval_start,
           out_dir, max_clips=30, camera=0, fps=20):
    matches = sleap_matches if source == "sleap" else dannce_matches
    # convert eval-relative indices to absolute SLEAP frame indices
    abs_matches = [int(eval_start + f) for f in matches]

    # cap at max_clips, prioritize ones with enough room for the full window
    valid = []
    n_total_frames = pc_dn_full.shape[0]  # absolute frames available
    for f in abs_matches:
        # we want clip [f - HALF + 1 .. f + HALF]
        start = f - HALF + 1
        end = start + CLIP
        if start < eval_start or end > n_total_frames:
            continue
        valid.append((f, start, end))
        if len(valid) >= max_clips:
            break
    if not valid:
        print(f"  {rat}/{session} [{source}]: no clips fit the window — skip")
        return

    sp = sleap_path(rat, session)
    cam_name = f"Camera{camera}"
    video_file = Path(sp) / cam_name / "0.mp4"
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_file}")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    PANEL_H = max(280, vid_h // 2)

    out_path = out_dir / f"{rat}_{session}_template1_{source}_matches.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (vid_w, vid_h + PANEL_H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer: {out_path}")

    print(f"  {rat}/{session} [{source}]: rendering {len(valid)} clips -> "
          f"{out_path.name}", flush=True)
    t0 = time.time()
    WHITE = (255, 255, 255); YELLOW = (0, 255, 255)

    for ci, (f_abs, start, end) in enumerate(valid):
        # PC trajectory window for the bottom panel
        pc_window = pc_dn_full[start: end]
        match_frame_in_window = HALF - 1  # match end sits at index HALF-1
        panel_bgr = make_pc_panel_image(
            pc_window, template_pc, bounds_per_frame,
            match_frame_in_window, vid_w, PANEL_H
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for fi in range(CLIP):
            ok, frame = cap.read()
            if not ok: break
            top = frame.copy()
            cv2.putText(top, f"{rat}/{session}  Cam{camera}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
            cv2.putText(top, f"{source}-match {ci+1}/{len(valid)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
            cv2.putText(top, f"frame {start + fi}  match @ {f_abs}",
                        (10, vid_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE,
                        2, cv2.LINE_AA)
            # Highlight match-center frame
            if fi == match_frame_in_window:
                cv2.rectangle(top, (5, 5), (vid_w - 5, vid_h - 5),
                              (0, 0, 255), 4)
            combined = np.vstack([top, panel_bgr])
            writer.write(combined)
        if (ci + 1) % 5 == 0 or ci == len(valid) - 1:
            print(f"    {ci + 1}/{len(valid)} clips ({time.time() - t0:.1f}s)",
                  flush=True)

    writer.release(); cap.release()
    print(f"  saved {out_path}  ({time.time() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--max_clips", type=int, default=30)
    ap.add_argument("--out_dir", default=str(REPO / "corrector" / "videos"
                                              / "match_review"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rat, session = args.rat, args.session
    sl, dn = load_paired_world(rat, session)
    idx = calibration_indices(len(sl), 5.0, SLEAP_HZ, 1000, seed=0)
    tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)

    eval_start = int(5 * 60 * SLEAP_HZ)
    if eval_start >= len(sl) - 1000:
        eval_start = 0

    tmpl = dict(load_template(rat, RAT_TEMPLATE[rat]))
    pcu = tmpl["pcs_to_use"].ravel().astype(int)
    feat_stds = tmpl["feature_stds"]
    bnds_scalar = DEFAULT_BOUNDS[rat]
    bnds = np.tile(feat_stds[pcu] * bnds_scalar, (WIN, 1))
    template_pc = tmpl["template"][:, pcu]

    # Full-session SLEAP-frame-aligned DANNCE PC trajectory (for the panel)
    dn_in_sleap_full = tx["apply_inverse"](dn.astype(np.float32)).astype(np.float32)
    dn_t_full = for_template(dn_in_sleap_full)
    pc_dn_full = project_to_template_pcs(dn_t_full, tmpl, pcu)

    # Match indices on the eval window
    sl_t = for_template(sl[eval_start:].astype(np.float32))
    dn_t = for_template(tx["apply_inverse"](dn[eval_start:].astype(np.float32)))
    pc_sl = project_to_template_pcs(sl_t, tmpl, pcu)
    pc_dn = project_to_template_pcs(dn_t, tmpl, pcu)

    sleap_matches = run_template_matching(
        pc_sl, template_pc, bnds, max_outside=3, refractory_frames=WIN)
    dannce_matches = run_template_matching(
        pc_dn, template_pc, bnds, max_outside=3, refractory_frames=WIN)
    print(f"{rat}/{session}: sleap matches = {len(sleap_matches)},  "
          f"dannce matches = {len(dannce_matches)}")

    for source in ("sleap", "dannce"):
        render(rat, session, source, sleap_matches, dannce_matches,
                pc_dn_full, template_pc, bnds, eval_start,
                out_dir, max_clips=args.max_clips, camera=0)


if __name__ == "__main__":
    main()
