"""Side-by-side pre- vs post-lesion template-match comparison video -- REAL
CAMERA FOOTAGE version.

Same match selection and PC1/PC2 template+trace panels as
render_lesion_match_compare_video.py (imported directly from it, not
duplicated), but the top panel per side shows actual DANNCE camera video
frames instead of a rendered 3D skeleton.

Frame correspondence: a DANNCE camera's `0.mp4` has exactly as many frames
as `dannce_keys_3D` has rows (verified: 92500 == 92500 for a 36000-SLEAP-
frame session), and `frametimes.npy`'s row 0 is a plain 1..N counter with no
gaps -- unlike the online SLEAP rig, DANNCE prediction runs deterministically
over the whole video with no dropped frames. So native DANNCE frame index i
IS video frame i directly, no alignment table needed. Verified empirically
here too: extracting the exact native frame at a known template-match end
(where PC0/z-height say the animal should be at a rear apex) shows the rat
mid-rear in the real footage, and the frame 2s earlier shows it crouched.

Usage (debug subset first):
  .../envs/clirb_analysis/bin/python scripts/render_lesion_match_compare_video_camera.py \
      --n_select 3 --camera Camera1 --out data/variability_videos/lesion_compare_cam_debug.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_lesion_match_compare_video import (load_session, select_representative_matches,  # noqa: E402
                                               build_clip, SIDE_TITLE_COLOR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import dannce_video_path  # noqa: E402
from clirb.native_dannce import DANNCE_FPS  # noqa: E402
from clirb.events import TEMPLATE_LEN  # noqa: E402

SLEAP_HZ = 20.0
DISPLAY_SCALE = 0.5  # downsize 1920x1200 native frames for render speed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre_rat", default="R5")
    ap.add_argument("--pre_session", default="2026_07_27_1")
    ap.add_argument("--post_rat", default="R5")
    ap.add_argument("--post_session", default="2026_09_11_1")
    ap.add_argument("--n_select", type=int, default=30)
    ap.add_argument("--buffer_s", type=float, default=2.0)
    ap.add_argument("--camera", default="Camera1")
    ap.add_argument("--out", default="data/variability_videos/lesion_compare_camera_top30.mp4")
    args = ap.parse_args()

    sessions = {
        "pre": load_session(args.pre_rat, args.pre_session),
        "post": load_session(args.post_rat, args.post_session),
    }
    picked = {side: select_representative_matches(sessions[side], args.n_select)
             for side in ("pre", "post")}

    clip_lists = {}
    for side in ("pre", "post"):
        clips = []
        for e in picked[side]:
            c = build_clip(sessions[side], e, args.buffer_s)
            if c is not None:
                clips.append(c)
        clip_lists[side] = clips
        print(f"  {side}: {len(clips)}/{len(picked[side])} selected matches fit a full window")

    n_clips = min(len(clip_lists["pre"]), len(clip_lists["post"]))
    W = clip_lists["pre"][0]["n_frames"]
    assert all(c["n_frames"] == W for side in clip_lists.values() for c in side)
    t_axis_s = np.arange(W) / DANNCE_FPS

    video_paths = {side: dannce_video_path(sessions[side]["rat"], sessions[side]["session"],
                                           camera=args.camera) for side in ("pre", "post")}
    caps = {side: cv2.VideoCapture(video_paths[side]) for side in ("pre", "post")}
    for side, cap in caps.items():
        if not cap.isOpened():
            raise SystemExit(f"could not open {video_paths[side]}")
    disp_w = int(1920 * DISPLAY_SCALE)
    disp_h = int(1200 * DISPLAY_SCALE)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    panels = {}
    for col_i, side in enumerate(["pre", "post"]):
        gs_side = gs[0, col_i].subgridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)
        ax_vid = fig.add_subplot(gs_side[0, 0])
        ax_pc1 = fig.add_subplot(gs_side[1, 0])
        ax_pc2 = fig.add_subplot(gs_side[2, 0])
        ax_vid.set_title(f"{'PRE-lesion' if side == 'pre' else 'POST-lesion'}  "
                        f"{sessions[side]['rat']}/{sessions[side]['session']}",
                        color=SIDE_TITLE_COLOR[side], fontsize=12)
        ax_vid.set_xticks([]); ax_vid.set_yticks([])
        im = ax_vid.imshow(np.zeros((disp_h, disp_w, 3), dtype=np.uint8))
        ax_pc2.set_xlabel("time within clip (s)")
        panels[side] = dict(ax_vid=ax_vid, ax_pc1=ax_pc1, ax_pc2=ax_pc2, im=im)

    print(f"rendering {n_clips} lockstep clip pairs -> {args.out}")
    writer = FFMpegWriter(fps=DANNCE_FPS, metadata={"title": "Pre vs post lesion template matches (camera)"})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with writer.saving(fig, args.out, dpi=100):
        for clip_i in range(n_clips):
            time_markers = {}
            for side in ["pre", "post"]:
                clip = clip_lists[side][clip_i]
                template = sessions[side]["template"]
                p = panels[side]
                for ax, pidx, label in [(p["ax_pc1"], 0, "PC1"), (p["ax_pc2"], 1, "PC2")]:
                    for artist in list(ax.lines) + list(ax.collections):
                        artist.remove()
                    t_match = (clip["match_lo_local"] + np.arange(TEMPLATE_LEN) *
                              (DANNCE_FPS / SLEAP_HZ)) / DANNCE_FPS
                    center = template["template"][:, pidx]
                    band = template["bounds_to_use"][:, pidx]
                    ax.fill_between(t_match, center - band, center + band,
                                   color="0.85", zorder=1)
                    ax.plot(t_match, center, "k--", lw=1.3, zorder=2, label="template")
                    ax.plot(t_axis_s, clip["pcs"][:, pidx], color="#1f77b4", lw=1.6,
                           zorder=3, label="this match")
                    ax.axvspan(clip["match_lo_local"] / DANNCE_FPS,
                              clip["match_hi_local"] / DANNCE_FPS, color="gold", alpha=0.12, zorder=0)
                    m = ax.axvline(0, color="red", lw=1.5, zorder=5)
                    time_markers[(side, pidx)] = m
                    ax.relim(); ax.autoscale_view()
                    ax.set_ylabel(label)
                    if pidx == 0:
                        ax.set_title(f"match {clip_i + 1}/{n_clips}", fontsize=9)
                        ax.legend(fontsize=6, loc="upper right")
                caps[side].set(cv2.CAP_PROP_POS_FRAMES, clip["lo"])

            for j in range(W):
                for side in ["pre", "post"]:
                    ok, frame = caps[side].read()
                    if ok:
                        frame_small = cv2.resize(frame, (disp_w, disp_h))
                        panels[side]["im"].set_data(frame_small[:, :, ::-1])  # BGR -> RGB
                    for pidx in (0, 1):
                        time_markers[(side, pidx)].set_xdata([j / DANNCE_FPS, j / DANNCE_FPS])
                writer.grab_frame()

            print(f"  rendered match {clip_i + 1}/{n_clips} -> {args.out}")

    for cap in caps.values():
        cap.release()
    plt.close(fig)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
