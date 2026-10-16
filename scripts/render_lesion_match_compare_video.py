"""Side-by-side pre- vs post-lesion template-match comparison video.

Left = R5/2026_07_27_1 (pre-lesion, highest-performing pre-lesion session).
Right = R5/2026_09_11_1 (post-lesion, highest-performing post-lesion session).

For each session independently: replay the template-matching rule (standard
bounds, accepted_errors=3) on native-DANNCE-at-SLEAP-rate PCs to get every
match, take each match's 30-frame (1.5s) PC1/PC2 trajectory, rank by mean
squared distance to the mean trajectory across that session's own matches,
and keep the 30 closest -- the "most representative" matches, independently
per session. Both sessions have DANNCE 3D reconstruction, so this uses the
same validated native-DANNCE pipeline as everywhere else in this project
(clirb.native_dannce), not the SLEAP-only fallback used for the earlier,
not-yet-reconstructed lesion session.

Each stitched clip = a fixed-length window: 2s buffer + the match itself
(nominally 1.5s @ the online rule's 20 Hz) + 2s buffer, cut from full native
~50 Hz DANNCE keypoints. Every clip is exactly the same native-frame length
(by construction, from the buffer + a NOMINAL match duration, not the
per-match native span, which can drift by a frame or two under the
frame-correspondence mapping) so the two sides advance in lockstep: match N
on the left plays alongside match N on the right, even though N indexes a
different, independently-selected match in each session.

Per side, per frame: a 3D skeleton animates through a fixed virtual arena
(bounds from that session's own COM range); below it, two mini panels show
that session's own PC1 and PC2 template +/- band (drawn only across the
match's nominal span) with the clip's full PC1/PC2 trace overlaid and a
moving time marker.

Usage (debug subset first -- see NOTE in render_session_variability_video.py
for why):
  .../envs/clirb_analysis/bin/python scripts/render_lesion_match_compare_video.py \
      --n_select 3 --out data/variability_videos/lesion_compare_debug.mp4
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import session_cache as sc  # noqa: E402
from config import EDGES  # noqa: E402
from clirb import templates as tmpl  # noqa: E402
from clirb.events import (project_session_to_pcs, replay_rule_offline,  # noqa: E402
                          TEMPLATE_LEN, causal_median_5_kp)
from clirb.native_dannce import DANNCE_FPS, CAUSAL_MEDIAN_PAD  # noqa: E402
from skeleton import normalize_skeleton_batch, project_to_pcs  # noqa: E402

ACCEPTED_ERRORS = 3
SLEAP_HZ = 20.0
CURRENT_COLOR = "#1f77b4"
SIDE_TITLE_COLOR = {"pre": "#1f77b4", "post": "#d62728"}


# --------------------------------------------------------------------------- #
# Per-session data + match selection
# --------------------------------------------------------------------------- #
def load_session(rat, session):
    raw = sc.get_raw_session(rat, session, want_dual=True, verbose=True)
    if raw is None or not raw["has_dual"]:
        raise SystemExit(f"{rat}/{session}: no dual SLEAP+DANNCE data cached")
    dn_native = raw["dannce_keys_3D"]
    dn_native = (dn_native.squeeze(1).transpose(0, 2, 1) if dn_native.ndim == 4
                else np.transpose(dn_native, [0, 2, 1]))
    di = raw["dannce_idx_for_sleap_cams"].astype(int).ravel()[1:]
    dn_at_sleap = dn_native[di]
    template = tmpl.load_template(rat, session)
    pcs_at_sleap, _ = project_session_to_pcs(dn_at_sleap, template, flip_z=False)
    match_frames, _ = replay_rule_offline(pcs_at_sleap, template, accepted_errors=ACCEPTED_ERRORS)
    print(f"  {rat}/{session}: {len(match_frames)} matches, template={template['_file']}")
    com_xy = dn_native.mean(axis=1)[:, :2]
    pad_xy = 0.1 * max(np.ptp(com_xy[:, 0]), np.ptp(com_xy[:, 1]))
    z_all = dn_native[:, :, 2]
    pad_z = 0.1 * np.ptp(z_all)
    return dict(rat=rat, session=session, dn_native=dn_native, di=di, template=template,
               pcs_at_sleap=pcs_at_sleap, match_frames=match_frames,
               arena_xlim=(com_xy[:, 0].min() - pad_xy, com_xy[:, 0].max() + pad_xy),
               arena_ylim=(com_xy[:, 1].min() - pad_xy, com_xy[:, 1].max() + pad_xy),
               arena_zlim=(z_all.min() - pad_z, z_all.max() + pad_z))


def select_representative_matches(sess, n_select):
    pcs, mf = sess["pcs_at_sleap"], sess["match_frames"]
    trajs = np.stack([pcs[e - TEMPLATE_LEN + 1:e + 1, :2] for e in mf])  # (n_match, 30, 2)
    mean_traj = trajs.mean(axis=0)
    mse = ((trajs - mean_traj[None]) ** 2).mean(axis=(1, 2))
    order = np.argsort(mse)
    n_select = min(n_select, len(mf))
    picked = np.sort(order[:n_select])  # chronological order among the picked
    print(f"  {sess['rat']}/{sess['session']}: picked {n_select}/{len(mf)} matches "
         f"(MSE-to-mean range {mse[order[0]]:.1f}..{mse[order[n_select - 1]]:.1f}, "
         f"full range up to {mse.max():.1f})")
    return mf[picked]


# --------------------------------------------------------------------------- #
# Native-rate buffered window: 2s buffer + nominal 1.5s match + 2s buffer
# --------------------------------------------------------------------------- #
def build_clip(sess, match_end20, buffer_s, median_pad=CAUSAL_MEDIAN_PAD):
    dn_native, di, template = sess["dn_native"], sess["di"], sess["template"]
    match_start20 = match_end20 - (TEMPLATE_LEN - 1)
    native_start = int(di[match_start20])
    native_end = int(di[match_end20])
    buffer_frames = int(round(buffer_s * DANNCE_FPS))
    nominal_match_frames = int(round(TEMPLATE_LEN / SLEAP_HZ * DANNCE_FPS))
    total_frames = 2 * buffer_frames + nominal_match_frames

    lo = native_start - buffer_frames
    hi = lo + total_frames - 1
    lo_pad = lo - median_pad
    if lo_pad < 0 or hi >= dn_native.shape[0]:
        return None
    world = dn_native[lo:hi + 1]
    buf = dn_native[lo_pad:hi + 1].astype(np.float32)
    rotated, _, _ = normalize_skeleton_batch(buf)
    rotated_med = causal_median_5_kp(rotated)[median_pad:]
    pcs = project_to_pcs(rotated_med, template["pc_weights"], template["feature_means"])
    return dict(world=world, pcs=pcs, lo=lo, hi=hi,
               match_lo_local=native_start - lo, match_hi_local=native_end - lo,
               n_frames=total_frames)


# --------------------------------------------------------------------------- #
# Skeleton drawing (single Line3D per skeleton via NaN-separated edges)
# --------------------------------------------------------------------------- #
def _edge_segments(points):
    n_edges = len(EDGES)
    x = np.full(n_edges * 3, np.nan)
    y = np.full(n_edges * 3, np.nan)
    z = np.full(n_edges * 3, np.nan)
    for i, (a, b) in enumerate(EDGES):
        x[3 * i:3 * i + 2] = (points[a, 0], points[b, 0])
        y[3 * i:3 * i + 2] = (points[a, 1], points[b, 1])
        z[3 * i:3 * i + 2] = (points[a, 2], points[b, 2])
    return x, y, z


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render(sessions, clip_lists, out_path, buffer_s, fps=DANNCE_FPS, dpi=100):
    n_clips = min(len(clip_lists["pre"]), len(clip_lists["post"]))
    W = clip_lists["pre"][0]["n_frames"]
    assert all(c["n_frames"] == W for side in clip_lists.values() for c in side), \
        "all clips must be the same native-frame length for lockstep playback"
    t_axis_s = np.arange(W) / DANNCE_FPS

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    panels = {}
    for col_i, side in enumerate(["pre", "post"]):
        gs_side = gs[0, col_i].subgridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)
        ax3d = fig.add_subplot(gs_side[0, 0], projection="3d")
        ax_pc1 = fig.add_subplot(gs_side[1, 0])
        ax_pc2 = fig.add_subplot(gs_side[2, 0])
        sess = sessions[side]
        ax3d.set_xlim(*sess["arena_xlim"]); ax3d.set_ylim(*sess["arena_ylim"])
        ax3d.set_zlim(*sess["arena_zlim"])
        ax3d.view_init(elev=25, azim=-60)
        ax3d.set_title(f"{'PRE-lesion' if side == 'pre' else 'POST-lesion'}  "
                      f"{sess['rat']}/{sess['session']}", color=SIDE_TITLE_COLOR[side], fontsize=12)
        skel_line, = ax3d.plot([], [], [], "-", lw=1.8, color=CURRENT_COLOR, zorder=10)
        ax_pc2.set_xlabel("time within clip (s)")
        panels[side] = dict(ax3d=ax3d, ax_pc1=ax_pc1, ax_pc2=ax_pc2, skel_line=skel_line)

    writer = FFMpegWriter(fps=fps, metadata={"title": "Pre vs post lesion template matches"})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with writer.saving(fig, out_path, dpi=dpi):
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
                    ax.plot(t_axis_s, clip["pcs"][:, pidx], color=CURRENT_COLOR, lw=1.6,
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

            for j in range(W):
                for side in ["pre", "post"]:
                    clip = clip_lists[side][clip_i]
                    x, y, z = _edge_segments(clip["world"][j])
                    panels[side]["skel_line"].set_data(x, y)
                    panels[side]["skel_line"].set_3d_properties(z)
                    for pidx in (0, 1):
                        time_markers[(side, pidx)].set_xdata([j / DANNCE_FPS, j / DANNCE_FPS])
                writer.grab_frame()

            print(f"  rendered match {clip_i + 1}/{n_clips} -> {out_path}")

    plt.close(fig)
    print(f"done: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre_rat", default="R5")
    ap.add_argument("--pre_session", default="2026_07_27_1")
    ap.add_argument("--post_rat", default="R5")
    ap.add_argument("--post_session", default="2026_09_11_1")
    ap.add_argument("--n_select", type=int, default=30)
    ap.add_argument("--buffer_s", type=float, default=2.0)
    ap.add_argument("--out", default="data/variability_videos/lesion_compare.mp4")
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
    print(f"rendering {n_clips} lockstep clip pairs -> {args.out}")
    render(sessions, clip_lists, args.out, args.buffer_s)


if __name__ == "__main__":
    main()
