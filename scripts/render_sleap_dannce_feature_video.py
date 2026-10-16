"""Render a SLEAP-vs-DANNCE comparison video for one session: three camera
panels on top (SLEAP keypoints in green, DANNCE keypoints in magenta,
overlaid on the raw video), and a scrolling grid of feature line plots below
-- one panel per stage21/22/23 feature, each showing SLEAP's and DANNCE's
value over the trailing 30-second window.

Pipeline (identical to stage20-23, applied to the rendered window):
  1. Load SLEAP's corrected triang_3d and DANNCE resampled onto the SLEAP
     frame grid (session_cache + clirb.spatial.dual.dannce_aligned).
  2. Fit ONE session-level rigid+scale Procrustes (sector-balanced sample,
     outlier-trimmed) mapping DANNCE -> SLEAP's world/calibration frame.
  3. Apply the online rule's own 5-frame causal median filter to both.
  4. Project both modalities' filtered keypoints onto each SLEAP camera via
     the session's own calibration -- this is what gets drawn on the video.
  5. Compute all 14 features from the same filtered keypoints for the line
     plots below.

Usage:
    python scripts/render_sleap_dannce_feature_video.py --rat R6 --session 2026_08_25_1
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO)
import os
os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb import features as feat
from clirb.spatial import dual as du
from clirb.spatial.geometry import ARENA, sector_index
from corrector.world_alignment import fit_procrustes
from corrector.data_world_2d import load_session_calibration, project_3d_to_2d_batch
from config import NODE_IDX, EDGES, sleap_path

OUT_ROOT = Path(REPO) / "figures" / "session_videos"

FPS = ev.FPS  # 20
SECTOR_W = 30
N_PER_SECTOR = 300
RNG = np.random.default_rng(0)

SNOUT, EARL, EARR = NODE_IDX['Snout'], NODE_IDX['EarL'], NODE_IDX['EarR']
SPINEF, SPINEM, SPINEL, TAILBASE = (NODE_IDX['SpineF'], NODE_IDX['SpineM'],
                                    NODE_IDX['SpineL'], NODE_IDX['TailBase'])
HIPL, HIPR, SHOULDERL, SHOULDERR = (NODE_IDX['HipL'], NODE_IDX['HipR'],
                                    NODE_IDX['ShoulderL'], NODE_IDX['ShoulderR'])

FEATURE_ORDER = ['snout_z', 'ear_snout_mean_z', 'ear_mean_z',
                 'angle_lower_xy', 'angle_upper_xy', 'angle_mid_xy', 'angle_skip_xy',
                 'angle_torsion_xy', 'angle_body_head_xy',
                 'dist_snout_tb', 'dist_ears_tb', 'dist_spineF_tb', 'dist_earmean_hipmean',
                 'speed']
UNITS = dict(snout_z='mm', ear_snout_mean_z='mm', ear_mean_z='mm',
            angle_lower_xy='deg', angle_upper_xy='deg', angle_mid_xy='deg', angle_skip_xy='deg',
            angle_torsion_xy='deg', angle_body_head_xy='deg',
            dist_snout_tb='mm', dist_ears_tb='mm', dist_spineF_tb='mm', dist_earmean_hipmean='mm',
            speed='mm/s')
SHORT_LABEL = dict(snout_z='snout Z', ear_snout_mean_z='snout+ears Z', ear_mean_z='ears Z',
                   angle_lower_xy='ang: lower', angle_upper_xy='ang: upper', angle_mid_xy='ang: mid',
                   angle_skip_xy='ang: skip', angle_torsion_xy='ang: torsion',
                   angle_body_head_xy='ang: body-head', dist_snout_tb='dist: snout-tb',
                   dist_ears_tb='dist: ears-tb', dist_spineF_tb='dist: spineF-tb',
                   dist_earmean_hipmean='dist: ears-hips', speed='speed')


def robust_proc(A, B, iters=5, keep=0.7):
    """apply(A) ~ B. robust_proc(dannce_pts, sleap_pts) -> apply() maps
    DANNCE -> SLEAP (validated convention, see SPATIAL_CONFOUND_RESULTS.md)."""
    idx = np.arange(len(A)); f = None
    for _ in range(iters):
        f = fit_procrustes(A[idx], B[idx], try_z_flip=True)
        r = np.linalg.norm(f['apply'](A) - B, axis=-1).mean(axis=1)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 200:
            break
    return f


def sector_balanced_sample(com_xy, n_per_sector=N_PER_SECTOR, rng=RNG):
    sec = sector_index(ARENA.theta(com_xy), SECTOR_W)
    samp = []
    for k in range(12):
        ii = np.where(sec == k)[0]
        if len(ii):
            samp.extend(rng.choice(ii, size=min(len(ii), n_per_sector), replace=False))
    return np.array(sorted(samp), dtype=int)


signed_angle_xy = feat.signed_angle_xy


# Feature definitions live in clirb.features (the single source of truth, §8s-§8z).
compute_all_features = feat.compute_features


def draw_two_skeletons(img, sleap_pts, dannce_pts, scale):
    def _draw(pts, color):
        p = np.rint(pts * scale).astype(np.int32)
        h, w = img.shape[:2]
        valid = (np.isfinite(pts).all(axis=-1) & (p[:, 0] >= 0) & (p[:, 0] < w)
                & (p[:, 1] >= 0) & (p[:, 1] < h))
        for e in EDGES:
            if valid[e[0]] and valid[e[1]]:
                cv2.line(img, tuple(p[e[0]]), tuple(p[e[1]]), color, 1, cv2.LINE_AA)
        for i in range(len(p)):
            if valid[i]:
                cv2.circle(img, tuple(p[i]), 3, color, -1, cv2.LINE_AA)
    _draw(dannce_pts, (255, 0, 255))  # magenta, BGR
    _draw(sleap_pts, (0, 255, 0))     # green


class FeaturePanel:
    """Grid of scrolling line plots, one per feature, SLEAP (green) vs.
    DANNCE (magenta), trailing `window_s`-second window. Built once; each
    frame only updates line data + xlim (fast, no artist rebuilding)."""

    def __init__(self, width, height, features, feat_sl, feat_dn, fps,
                window_s=30.0, dpi=100, ncols=4):
        self.fps = fps
        self.window_s = window_s
        self.window_fr = int(window_s * fps)
        self.features = features
        nrows = int(np.ceil(len(features) / ncols))

        self.fig, axes = plt.subplots(nrows, ncols, figsize=(width / dpi, height / dpi), dpi=dpi)
        self.fig.patch.set_facecolor("white")
        self.axes_flat = np.atleast_1d(axes).ravel()
        self.ax_of = {}
        self.lines_sl, self.lines_dn = {}, {}

        for ax, feat in zip(self.axes_flat, features):
            both = np.concatenate([feat_sl[feat], feat_dn[feat]])
            both = both[np.isfinite(both)]
            lo, hi = np.percentile(both, [1, 99])
            pad = 0.1 * max(hi - lo, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{SHORT_LABEL[feat]} ({UNITS[feat]})", fontsize=8, color="0.1")
            ax.tick_params(labelsize=6, colors="0.3")
            for spine in ax.spines.values():
                spine.set_color("0.7")
            ax.grid(alpha=0.4, lw=0.5)
            ln_s, = ax.plot([], [], '-', color='#1baf1b', lw=1.3, label='SLEAP')
            ln_d, = ax.plot([], [], '-', color='#d81bd8', lw=1.3, label='DANNCE')
            self.lines_sl[feat] = ln_s
            self.lines_dn[feat] = ln_d
            self.ax_of[feat] = ax

        for ax in self.axes_flat[len(features):]:
            ax.axis('off')
        self.axes_flat[0].legend(fontsize=7, loc='upper left', ncol=2)
        self.fig.tight_layout(pad=0.6, h_pad=1.0, w_pad=0.8)

    def update(self, frame_idx, feat_sl, feat_dn):
        t_now = frame_idx / self.fps
        i0 = max(0, frame_idx - self.window_fr + 1)
        i1 = frame_idx + 1
        x = np.arange(i0, i1) / self.fps
        for feat in self.features:
            self.lines_sl[feat].set_data(x, feat_sl[feat][i0:i1])
            self.lines_dn[feat].set_data(x, feat_dn[feat][i0:i1])
            self.ax_of[feat].set_xlim(max(0.0, t_now - self.window_s), max(t_now, self.window_s))
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def close(self):
        plt.close(self.fig)


def render(rat, session, n_frames, fps, panel_width, feature_panel_height,
          window_s, h264, out_path):
    print(f"loading {rat}/{session} ...", flush=True)
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw is None or not raw.get('has_dual', False):
        raise RuntimeError(f"{rat}/{session} has no dual SLEAP+DANNCE data")
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        raise RuntimeError(f"{rat}/{session}: dannce_aligned returned None")
    n_total = min(len(raw['triang_3d']), len(d3))
    n = min(n_frames, n_total)
    print(f"  session has {n_total} dual frames; rendering first {n} "
         f"({n / fps / 60:.1f} min at {fps} fps)", flush=True)

    sl_raw_full = np.asarray(raw['triang_3d'][:n_total], dtype=np.float32)
    dn_raw_full = np.asarray(d3[:n_total], dtype=np.float32)

    # --- one session-level rigid+scale Procrustes, DANNCE -> SLEAP frame ---
    print("fitting world-space Procrustes alignment (sector-balanced sample)...", flush=True)
    samp = sector_balanced_sample(sl_raw_full.mean(axis=1)[:, :2])
    fit = robust_proc(dn_raw_full[samp], sl_raw_full[samp])
    print(f"  fit residual={fit['residual']:.2f}mm  scale={fit['s']:.4f}  "
         f"z_flipped={fit['z_flipped']}", flush=True)
    dn_aligned_full = fit['apply'](dn_raw_full)
    overall_rmse = float(np.sqrt(((sl_raw_full - dn_aligned_full) ** 2).sum(axis=-1).mean()))
    print(f"  whole-session world-space RMSE (sanity check): {overall_rmse:.1f}mm", flush=True)

    # --- slice to the rendered window, then causal-median filter (online rule's own filter) ---
    sl_raw = sl_raw_full[:n]
    dn_aligned = dn_aligned_full[:n]
    sl = ev.causal_median_5_kp(sl_raw, warmup='online')
    dn = ev.causal_median_5_kp(dn_aligned, warmup='online')

    print("computing features ...", flush=True)
    feat_sl = compute_all_features(sl)
    feat_dn = compute_all_features(dn)

    # --- project both modalities onto each SLEAP camera ---
    print("loading calibration + projecting to cameras 0,1,2 ...", flush=True)
    calib, cal_date = load_session_calibration(rat, session)
    print(f"  calibration date: {cal_date}", flush=True)
    sl_2d = {c: project_3d_to_2d_batch(sl, calib[c]) for c in range(3)}
    dn_2d = {c: project_3d_to_2d_batch(dn, calib[c]) for c in range(3)}

    # --- video captures ---
    sp = Path(sleap_path(rat, session))
    caps, vid_dims = {}, None
    for c in range(3):
        vf = sp / f"Camera{c}" / "0.mp4"
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {vf}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if vid_dims is None:
            vid_dims = (w, h)
        caps[c] = cap
    vid_w, vid_h = vid_dims

    panel_w = panel_width - (panel_width % 2)
    panel_h = int(round(panel_w * vid_h / vid_w)); panel_h -= panel_h % 2
    scale = panel_w / vid_w
    top_w = panel_w * 3

    feat_h = feature_panel_height - (feature_panel_height % 2)
    out_w, out_h = top_w, panel_h + feat_h
    print(f"output: {out_w}x{out_h}  (top {panel_w}x{panel_h} x3, features {top_w}x{feat_h})",
         flush=True)

    print("building feature panel figure ...", flush=True)
    fpanel = FeaturePanel(top_w, feat_h, FEATURE_ORDER, feat_sl, feat_dn, fps,
                          window_s=window_s)

    if out_path is None:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = OUT_ROOT / f"{rat}_{session}_sleap_dannce_features_f0-{n}.mp4"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_path}")

    WHITE, YELLOW = (255, 255, 255), (0, 255, 255)
    GREEN, MAGENTA = (0, 255, 0), (255, 0, 255)
    print(f"rendering -> {out_path.name} ...", flush=True)
    t0 = time.time()
    written = 0
    for fi in range(n):
        panels, bad = [], False
        for c in range(3):
            ok, frame = caps[c].read()
            if not ok:
                bad = True
                break
            small = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            draw_two_skeletons(small, sl_2d[c][fi], dn_2d[c][fi], scale)
            cv2.putText(small, f"Camera{c}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, WHITE, 2, cv2.LINE_AA)
            panels.append(small)
        if bad:
            print(f"  video ended early at frame {fi}", flush=True)
            break
        top = np.hstack(panels)
        cv2.putText(top, "SLEAP", (8, panel_h - 34), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, GREEN, 2, cv2.LINE_AA)
        cv2.putText(top, "DANNCE", (8, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, MAGENTA, 2, cv2.LINE_AA)
        t_s = fi / fps
        cv2.putText(top, f"{rat}/{session}  frame {fi}  t={t_s // 60:.0f}:{t_s % 60:04.1f}",
                   (top_w - 430, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1,
                   cv2.LINE_AA)

        bottom = fpanel.update(fi, feat_sl, feat_dn)
        combined = np.vstack([top, bottom])
        writer.write(combined)
        written += 1
        if written % 200 == 0 or fi == n - 1:
            el = time.time() - t0
            print(f"  {written}/{n}  {el:.1f}s  ({written / max(el, 1e-6):.1f} fps)", flush=True)

    writer.release()
    for c in caps:
        caps[c].release()
    fpanel.close()

    if h264 and written:
        import shutil, subprocess
        if shutil.which("ffmpeg") is None:
            print("note: ffmpeg not found, leaving the mp4v file as-is", flush=True)
        else:
            tmp = out_path.with_name("_h264_" + out_path.name)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_path),
                  "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
            print("re-encoding to H.264 ...", flush=True)
            try:
                subprocess.run(cmd, check=True)
                before = out_path.stat().st_size / 1e6
                tmp.replace(out_path)
                after = out_path.stat().st_size / 1e6
                print(f"  {before:.0f} MB -> {after:.0f} MB", flush=True)
            except subprocess.CalledProcessError as exc:
                tmp.unlink(missing_ok=True)
                print(f"note: ffmpeg failed ({exc}); keeping the mp4v file", flush=True)

    print(f"\nwrote {out_path}  ({written} frames, {written / fps:.1f}s at {fps} fps)", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--n_frames", type=int, default=6000, help="default: first 5 min at 20fps")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--panel_width", type=int, default=640)
    ap.add_argument("--feature_panel_height", type=int, default=1000)
    ap.add_argument("--window_s", type=float, default=30.0)
    ap.add_argument("--no_h264", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    render(a.rat, a.session, a.n_frames, a.fps, a.panel_width,
          a.feature_panel_height, a.window_s, not a.no_h264, a.out)


if __name__ == '__main__':
    main()
