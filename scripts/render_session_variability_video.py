"""Render a sliding-window rear-variability video for one CLIRB session.

For each rear in a chosen range, plays its 4s native-rate (~50 Hz) window:

  left-top    : PC1/PC2/PC3 trajectories over the 4s window -- current rear
                highlighted, sliding-window mean +/- std (current rear + up
                to the previous LOOKBACK rears, shrinking at the session
                start) shaded behind it, moving time marker.
  left-bottom : session-time scrubber -- a dot per rear, a dot per reward
                event, a red bar over the rear currently playing.
  right       : world-frame skeleton in a fixed virtual-arena view -- the
                current rear at full alpha, up to LOOKBACK previous rears
                animating in LOCKSTEP (same relative frame-in-window) at low
                alpha ("ghost" rears).

Shares its rear-detection / alignment / windowing / variability logic with
session_variability.ipynb via clirb.native_dannce and clirb.events -- the two
should always agree on rear count, matched status, and variability values.

Usage (debug subset first -- rendering is slow, see NOTE below):
  .../envs/clirb_analysis/bin/python scripts/render_session_variability_video.py \
      --rat R5 --session 2026_07_27_1 --start 40 --n_rears 5 \
      --out data/variability_videos/R5_2026_07_27_1_debug.mp4

NOTE on runtime: each output frame is a full 3D-scene + line-plot redraw.
A LOOKBACK+1=5-skeleton lockstep scene at 201 frames/rear is slow to render
per-frame; a full-session run (e.g. ~250 rears) is ~250x a 1-rear debug run
and should be launched in the background, not run interactively.
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
from config import EDGES, DATA_ROOT  # noqa: E402
from data_io import load_aligned_data  # noqa: E402
from clirb import templates as tmpl  # noqa: E402
from clirb.events import (project_session_to_pcs, detect_rears, best_align_end,  # noqa: E402
                          replay_rule_offline, rear_matched, TEMPLATE_LEN)
from clirb.native_dannce import (extract_native_window, reward_native_frames,  # noqa: E402
                                 sliding_window_variability, sliding_window_mean_std,
                                 DANNCE_FPS)

PC_LABELS = ("PC1", "PC2", "PC3")
GHOST_ALPHA = 0.25
CURRENT_COLOR = "#1f77b4"
GHOST_COLOR = "#888888"
MATCHED_COLOR = "#2ca02c"
UNMATCHED_COLOR = "#d62728"


# --------------------------------------------------------------------------- #
# Shared session build (mirrors session_variability.ipynb)
# --------------------------------------------------------------------------- #
def build_session(rat, session, half_win_frames, lookback, accepted_errors=3):
    raw = sc.get_raw_session(rat, session, want_dual=True, verbose=True)
    if raw is None or not raw["has_dual"]:
        raise SystemExit(f"{rat}/{session}: no dual SLEAP+DANNCE data cached")
    if not raw["has_task_log"]:
        raise SystemExit(f"{rat}/{session}: no behavior_log -- no reward events")

    dn_native = raw["dannce_keys_3D"]
    dn_native = (dn_native.squeeze(1).transpose(0, 2, 1) if dn_native.ndim == 4
                else np.transpose(dn_native, [0, 2, 1]))
    di = raw["dannce_idx_for_sleap_cams"].astype(int).ravel()[1:]
    dn_at_sleap = dn_native[di]

    template = tmpl.load_template(rat, session)
    pcs_at_sleap, _ = project_session_to_pcs(dn_at_sleap, template, flip_z=False)
    rears = detect_rears(pcs_at_sleap[:, 0])
    rear_start = np.array([r[0] for r in rears])
    rear_end = np.array([r[1] for r in rears])
    print(f"{rat}/{session}: {len(rears)} rears detected")

    match_frames, _ = replay_rule_offline(pcs_at_sleap, template, accepted_errors=accepted_errors)
    rear_is_matched, _ = rear_matched(rear_start, rear_end, match_frames, template_len=TEMPLATE_LEN)

    windows, valid_rear_idx, anchor_native = [], [], []
    for k, (rs, re_, _n_above) in enumerate(rears):
        end20, _mse = best_align_end(rs, re_, pcs_at_sleap, template["template"][:, :2])
        if end20 < 0:
            continue
        a_native = int(di[end20])
        w = extract_native_window(dn_native, a_native, template, half_win_frames=half_win_frames)
        if w is None:
            continue  # +/-2s window runs off the session boundary -- excluded
        windows.append(w)
        valid_rear_idx.append(k)
        anchor_native.append(a_native)
    anchor_native = np.array(anchor_native)
    print(f"{len(windows)}/{len(rears)} rears kept ({len(rears) - len(windows)} "
          f"excluded: boundary-adjacent)")

    rear_is_matched_valid = rear_is_matched[valid_rear_idx]
    pcs_windows = [w["pcs"][:, :3] for w in windows]
    variability = np.array([sliding_window_variability(pcs_windows, i, lookback=lookback)
                            for i in range(len(pcs_windows))])

    aligned = load_aligned_data(rat, session)
    dannce_times_ms = aligned["dannce_times_ms"].ravel()
    t0_ms = dannce_times_ms.min()
    t_rear_min = (dannce_times_ms[anchor_native] - t0_ms) / 1000.0 / 60.0

    reward_frames = reward_native_frames(rat, session, di, data_root=DATA_ROOT)
    reward_frames = reward_frames[reward_frames < dn_native.shape[0]]
    t_reward_min = (dannce_times_ms[reward_frames] - t0_ms) / 1000.0 / 60.0
    print(f"{len(t_reward_min)} reward events")

    com_xy = dn_native.mean(axis=1)[:, :2]
    pad_xy = 0.1 * max(np.ptp(com_xy[:, 0]), np.ptp(com_xy[:, 1]))
    z_all = dn_native[:, :, 2]
    pad_z = 0.1 * np.ptp(z_all)

    return dict(
        windows=windows, rear_is_matched_valid=rear_is_matched_valid,
        pcs_windows=pcs_windows, variability=variability,
        t_rear_min=t_rear_min, t_reward_min=t_reward_min,
        arena_xlim=(com_xy[:, 0].min() - pad_xy, com_xy[:, 0].max() + pad_xy),
        arena_ylim=(com_xy[:, 1].min() - pad_xy, com_xy[:, 1].max() + pad_xy),
        arena_zlim=(z_all.min() - pad_z, z_all.max() + pad_z),
    )


# --------------------------------------------------------------------------- #
# Skeleton drawing helper -- one Line3D per skeleton via NaN-separated edges
# --------------------------------------------------------------------------- #
def _edge_segments(points):
    """points: (23, 3). Returns x, y, z arrays with a NaN gap between edges,
    suitable for a single ax.plot3D call / set_data_3d update."""
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
def render(sess, render_idx, out_path, fps=DANNCE_FPS, lookback=4, dpi=100):
    windows = sess["windows"]
    n_pcs = 3
    W = windows[0]["pcs"].shape[0]
    half_win = W // 2
    t_axis_s = (np.arange(W) - half_win) / DANNCE_FPS

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])
    gs_left = gs[0, 0].subgridspec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.45)
    pc_axes = [fig.add_subplot(gs_left[i, 0]) for i in range(3)]
    ax_scrub = fig.add_subplot(gs_left[3, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")

    # --- static scrubber content (rear + reward dots), drawn once ---
    ax_scrub.scatter(sess["t_rear_min"], np.zeros_like(sess["t_rear_min"]),
                     s=10, c="#666", label=f"rears (n={len(sess['t_rear_min'])})")
    ax_scrub.scatter(sess["t_reward_min"], np.ones_like(sess["t_reward_min"]),
                     s=40, c="gold", edgecolor="k", marker="v",
                     label=f"rewards (n={len(sess['t_reward_min'])})")
    ax_scrub.set_yticks([0, 1])
    ax_scrub.set_yticklabels(["rear", "reward"])
    ax_scrub.set_xlabel("session time (min)")
    ax_scrub.set_ylim(-0.6, 1.6)
    ax_scrub.legend(fontsize=7, loc="upper right", ncol=2)
    playhead = ax_scrub.axvline(sess["t_rear_min"][render_idx[0]], color="red", lw=2, zorder=5)

    # --- PC panels: static per-rear content rebuilt each segment; a moving
    # time marker updated every frame ---
    pc_time_markers = []
    for ax, label in zip(pc_axes, PC_LABELS):
        ax.set_ylabel(label)
        m = ax.axvline(0, color="k", lw=1.5, zorder=5)
        pc_time_markers.append(m)
    pc_axes[-1].set_xlabel("time from alignment anchor (s)")

    # --- 3D arena, fixed bounds for the whole video ---
    ax3d.set_xlim(*sess["arena_xlim"])
    ax3d.set_ylim(*sess["arena_ylim"])
    ax3d.set_zlim(*sess["arena_zlim"])
    ax3d.set_xlabel("x (world)"); ax3d.set_ylabel("y (world)"); ax3d.set_zlabel("z")
    ax3d.view_init(elev=25, azim=-60)
    n_slots = lookback + 1  # current + up to `lookback` ghosts
    skel_lines = [ax3d.plot([], [], [], "-", lw=1.5,
                            color=CURRENT_COLOR if s == 0 else GHOST_COLOR,
                            alpha=1.0 if s == 0 else GHOST_ALPHA, zorder=10 - s)[0]
                 for s in range(n_slots)]

    writer = FFMpegWriter(fps=fps, metadata={"title": "CLIRB session variability"})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with writer.saving(fig, out_path, dpi=dpi):
        for count, i in enumerate(render_idx):
            lo = max(0, i - lookback)
            slot_windows = windows[lo:i + 1][::-1]  # slot 0 = current, then ghosts back in time

            # --- rebuild the per-segment static PC content ---
            mean_t, std_t = sliding_window_mean_std(sess["pcs_windows"], i, lookback=lookback)
            matched = sess["rear_is_matched_valid"][i]
            band_color = MATCHED_COLOR if matched else UNMATCHED_COLOR
            for pidx, ax in enumerate(pc_axes):
                for artist in list(ax.lines) + list(ax.collections):
                    artist.remove()
                cur = windows[i]["pcs"][:, pidx]
                ax.plot(t_axis_s, mean_t[:, pidx], color=band_color, lw=1.5,
                       label=f"window mean (n={i - lo + 1})")
                ax.fill_between(t_axis_s, mean_t[:, pidx] - std_t[:, pidx],
                                mean_t[:, pidx] + std_t[:, pidx], color=band_color, alpha=0.18)
                ax.plot(t_axis_s, cur, color=CURRENT_COLOR, lw=2, label="current rear")
                pc_time_markers[pidx] = ax.axvline(t_axis_s[0], color="k", lw=1.5, zorder=5)
                ax.relim()
                ax.autoscale_view()
            pc_axes[0].legend(fontsize=7, loc="upper right")
            pc_axes[0].set_title(f"rear {i + 1}/{len(windows)} "
                                 f"({'matched' if matched else 'unmatched'})", fontsize=10)

            playhead.set_xdata([sess["t_rear_min"][i], sess["t_rear_min"][i]])

            for j in range(W):
                for pidx, m in enumerate(pc_time_markers):
                    m.set_xdata([t_axis_s[j], t_axis_s[j]])
                for slot, sw in enumerate(slot_windows):
                    x, y, z = _edge_segments(sw["world"][j])
                    skel_lines[slot].set_data(x, y)
                    skel_lines[slot].set_3d_properties(z)
                for slot in range(len(slot_windows), n_slots):
                    skel_lines[slot].set_data([], [])
                    skel_lines[slot].set_3d_properties([])
                writer.grab_frame()

            if (count + 1) % 5 == 0 or count == len(render_idx) - 1:
                print(f"  rendered {count + 1}/{len(render_idx)} rears -> {out_path}")

    plt.close(fig)
    print(f"done: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0, help="first valid-rear index to render")
    ap.add_argument("--n_rears", type=int, default=5, help="how many rears to render")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth valid rear")
    ap.add_argument("--half_win_s", type=float, default=2.0)
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--fps", type=float, default=DANNCE_FPS,
                    help="output fps; DANNCE_FPS (50) plays in real time")
    ap.add_argument("--accepted_errors", type=int, default=3)
    args = ap.parse_args()

    half_win_frames = int(round(args.half_win_s * DANNCE_FPS))
    sess = build_session(args.rat, args.session, half_win_frames, args.lookback,
                         accepted_errors=args.accepted_errors)

    n_valid = len(sess["windows"])
    end = min(args.start + args.n_rears * args.stride, n_valid)
    render_idx = list(range(args.start, end, args.stride))
    if not render_idx:
        raise SystemExit(f"no rears to render: start={args.start}, n_valid={n_valid}")
    print(f"rendering {len(render_idx)} of {n_valid} valid rears "
         f"(indices {render_idx[0]}..{render_idx[-1]}, stride {args.stride})")

    render(sess, render_idx, args.out, fps=args.fps, lookback=args.lookback)


if __name__ == "__main__":
    main()
