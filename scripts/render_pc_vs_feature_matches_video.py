"""Side-by-side video: the production PC0/PC1 template's top matches (left)
vs. the `ear_mean_z + cum_heading_change` corridor's top matches (right), in
ONE session, over real DANNCE camera footage.

Both sides match SLEAP (what the online rule actually sees); the camera panel
is DANNCE footage, indexed through `dannce_idx_for_sleap_cams`. Rendered at
SLEAP's own 20 Hz -- the rate the rule operates at -- so no resampling is
needed between the signal traces and the video.

LEFT  : production rule. replay_rule_offline on SLEAP PCs with the session's
        own template and bounds_to_use. Panels show PC0 and PC1 against the
        template corridor.
RIGHT : §8ab's candidate. A 2-D corridor in (ear_mean_z, cum_heading_change)
        built exactly as stage28 builds it -- DANNCE-only template from the
        rear+turn seed windows, re-matched to get ground truth, then a SLEAP
        template rebuilt from that -- then matched against SLEAP. Panels show
        each feature against its own corridor.

"Top" means most representative on both sides, by the same criterion the
lesion-compare renderer uses: lowest MSE of the event's own 30-frame
trajectory to the mean trajectory of all that side's matches. Apples to
apples -- neither side is ranked by a score the other does not have.

Usage:
  .../envs/clirb_analysis/bin/python scripts/render_pc_vs_feature_matches_video.py \
      --rat R5 --session 2026_07_27_1 --n_select 20 \
      --out data/variability_videos/R5_pc_vs_feature_top20.mp4
"""
import argparse, os, sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))

import session_cache as sc
from config import dannce_video_path
from clirb import events as ev, templates as tp, features as cf
from clirb.spatial import dual as du
import stage24_multi_template_features as s24

W = ev.TEMPLATE_LEN
SLEAP_HZ = 20.0
FEATURES = ("ear_mean_z", "cum_heading_change")
SIDE_COLOR = {"pc": "#1f77b4", "feat": "#d62728"}
DISPLAY_SCALE = 0.5


def feature_corridor(sd):
    """stage28's construction, returning the SLEAP template + bounds + matches."""
    s_traj = np.stack([sd["feat_sl"][f] for f in FEATURES], axis=1)
    d_traj = np.stack([sd["feat_dn"][f] for f in FEATURES], axis=1)
    acc = max(1, round(1.5 * len(FEATURES)))
    seeds = sd["seeds"]["rear_turn"]
    s1, _ = s24.windows_at(s_traj, seeds); d1, _ = s24.windows_at(d_traj, seeds)
    dt0, _, b1 = s24.rebuild_and_bounds(d1, s1)
    gt, _ = ev.get_template_match_indices(d_traj, dt0, b1, refractory_frames=W, accepted_errors=acc)
    gt = ev.correct_matches(d_traj, dt0, gt)
    sseg, _ = s24.windows_at(s_traj, gt); dseg, _ = s24.windows_at(d_traj, gt)
    st, _, bounds = s24.rebuild_and_bounds(sseg, dseg)
    m, _ = ev.get_template_match_indices(s_traj, st, bounds, refractory_frames=W, accepted_errors=acc)
    m = ev.correct_matches(s_traj, st, m) if len(m) else np.array([], dtype=int)
    return dict(traj=s_traj, template=st, bounds=bounds, matches=np.asarray(m))


def top_matches(traj, matches, n_select):
    """Most representative: lowest MSE of own window to the mean window."""
    matches = np.asarray([e for e in matches if e >= W - 1 and e < len(traj)])
    if len(matches) == 0:
        return matches
    wins = np.stack([traj[e - W + 1:e + 1] for e in matches])
    mse = ((wins - wins.mean(axis=0)[None]) ** 2).mean(axis=(1, 2))
    order = np.argsort(mse)[:min(n_select, len(matches))]
    return np.sort(matches[order])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rat", default="R5")
    ap.add_argument("--session", default="2026_07_27_1")
    ap.add_argument("--n_select", type=int, default=20)
    ap.add_argument("--buffer_s", type=float, default=1.5)
    ap.add_argument("--camera", default="Camera1")
    ap.add_argument("--out", default="data/variability_videos/R5_pc_vs_feature_top20.mp4")
    args = ap.parse_args()

    rat, session = args.rat, args.session
    sd = s24.session_data(rat, session)
    if sd is None:
        raise SystemExit(f"{rat}/{session}: no dual data or failed the world-space fit guard")
    raw = sc.get_raw_session(rat, session, want_dual=True)
    tm = tp.load_template(rat, session)
    di = raw["dannce_idx_for_sleap_cams"].astype(int).ravel()[1:]

    d3 = du.dannce_aligned(raw)
    n = min(len(raw["triang_3d"]), len(d3), len(di))
    s_pcs, _ = ev.project_session_to_pcs(raw["triang_3d"][:n], tm, flip_z=True)
    pcu = np.asarray(tm["pcs_to_use"]).ravel().astype(int)[:2]

    pc_matches, _ = ev.replay_rule_offline(s_pcs, tm)
    pc_traj = s_pcs[:, pcu]
    fc = feature_corridor(sd)
    print(f"{rat}/{session}  template={tm['_file']}")
    print(f"  PC0/PC1 production rule : {len(pc_matches)} matches")
    print(f"  ear_mean_z+cum_heading  : {len(fc['matches'])} matches")

    a, b = np.asarray(pc_matches), np.asarray(fc["matches"])
    if len(a) and len(b):
        shared = sum(np.min(np.abs(b - e)) < 10 for e in a)
        print(f"  overlap: {shared}/{len(a)} of the production rule's matches are also "
              f"found by the feature corridor ({shared / len(a):.1%}); "
              f"the corridor finds {len(b) - shared} the rule does not")

    pick = {"pc": top_matches(pc_traj, pc_matches, args.n_select),
            "feat": top_matches(fc["traj"], fc["matches"], args.n_select)}
    n_clips = min(len(pick["pc"]), len(pick["feat"]))
    print(f"  rendering top {n_clips} per side")

    buf = int(round(args.buffer_s * SLEAP_HZ))
    clip_len = 2 * buf + W

    def clip_range(e):
        lo = e - W + 1 - buf
        return (lo, lo + clip_len - 1) if lo >= 0 and lo + clip_len - 1 < n else None

    clips = {s: [c for c in (clip_range(e) for e in pick[s][:n_clips]) if c] for s in pick}
    n_clips = min(len(clips["pc"]), len(clips["feat"]))

    vid = dannce_video_path(rat, session, camera=args.camera)
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise SystemExit(f"could not open {vid}")
    dw, dh = int(1920 * DISPLAY_SCALE), int(1200 * DISPLAY_SCALE)

    def grab(native_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(native_idx))
        ok, fr = cap.read()
        if not ok:
            return np.zeros((dh, dw, 3), np.uint8)
        return cv2.cvtColor(cv2.resize(fr, (dw, dh)), cv2.COLOR_BGR2RGB)

    # NB: features are computed on RAW SLEAP keypoints (the convention stage21-28
    # use, with DANNCE Procrustes-aligned into that same frame), and SLEAP's raw
    # z axis points DOWN -- hence ear_mean_z is negative and a rear makes it more
    # negative. Labelled rather than silently flipped, to stay identical to the
    # analysis convention.
    SIG = {"pc": [("PC0", pc_traj[:, 0], tm["template"][:, 0], tm["bounds_to_use"][:, 0]),
                  ("PC1", pc_traj[:, 1], tm["template"][:, 1], tm["bounds_to_use"][:, 1])],
           "feat": [(FEATURES[0], fc["traj"][:, 0], fc["template"][:, 0], fc["bounds"][:, 0]),
                    (FEATURES[1], fc["traj"][:, 1], fc["template"][:, 1], fc["bounds"][:, 1])]}
    LABEL = {"ear_mean_z": "ear_mean_z\n(raw SLEAP z: down +)",
             "cum_heading_change": "cum_heading_change\n(deg, + = left)"}
    TITLE = {"pc": "PRODUCTION RULE — PC0/PC1 template",
             "feat": "CANDIDATE — ear_mean_z + cum_heading_change"}

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 2, wspace=0.12)
    panel = {}
    for col, side in enumerate(("pc", "feat")):
        sub = gs[0, col].subgridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.42)
        ax_v = fig.add_subplot(sub[0, 0])
        ax_v.set_title(f"{TITLE[side]}\n{rat}/{session}", color=SIDE_COLOR[side], fontsize=11)
        ax_v.set_xticks([]); ax_v.set_yticks([])
        panel[side] = dict(v=ax_v, im=ax_v.imshow(np.zeros((dh, dw, 3), np.uint8)),
                           a=[fig.add_subplot(sub[1, 0]), fig.add_subplot(sub[2, 0])])
        panel[side]["a"][1].set_xlabel("time within clip (s)")

    t = np.arange(clip_len) / SLEAP_HZ
    t_match = (buf + np.arange(W)) / SLEAP_HZ
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = FFMpegWriter(fps=SLEAP_HZ, metadata={"title": "PC vs feature matches"})

    with writer.saving(fig, args.out, dpi=100):
        for ci in range(n_clips):
            cursors = []
            for side in ("pc", "feat"):
                lo, hi = clips[side][ci]
                for ax, (label, sig, ctr, bnd) in zip(panel[side]["a"], SIG[side]):
                    for art in list(ax.lines) + list(ax.collections):
                        art.remove()
                    ax.fill_between(t_match, ctr - bnd, ctr + bnd, color="0.85", zorder=1)
                    ax.plot(t_match, ctr, "k--", lw=1.3, zorder=2, label="template")
                    ax.plot(t, sig[lo:hi + 1], color=SIDE_COLOR[side], lw=1.7, zorder=3,
                            label="this match")
                    ax.axvspan(t_match[0], t_match[-1], color="gold", alpha=0.12, zorder=0)
                    cursors.append(ax.axvline(0, color="red", lw=1.5, zorder=5))
                    ax.relim(); ax.autoscale_view()
                    ax.set_ylabel(LABEL.get(label, label), fontsize=8)
                panel[side]["a"][0].set_title(f"match {ci + 1}/{n_clips}  (frame {hi - buf})",
                                              fontsize=9)
                panel[side]["a"][0].legend(fontsize=6, loc="upper right")
            for j in range(clip_len):
                for side in ("pc", "feat"):
                    lo, _ = clips[side][ci]
                    panel[side]["im"].set_data(grab(di[lo + j]))
                for c in cursors:
                    c.set_xdata([t[j], t[j]])
                writer.grab_frame()
            print(f"  clip {ci + 1}/{n_clips} written", flush=True)
    cap.release()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
