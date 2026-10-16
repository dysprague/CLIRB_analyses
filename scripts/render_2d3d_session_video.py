"""Two-tier session video: per-camera 2D keypoint overlays on top, triangulated
3D skeleton below.

Top row  : one panel per camera, the raw Camera<N>/0.mp4 frame with that
           camera's OWN 2D keypoints overlaid. By default these are the native
           SLEAP detections from sleap_keys_2D.npy -- i.e. what the network
           actually saw in that view, before triangulation. Pass
           --kp2d reproj to instead reproject triang_keys_3D.npy through the
           session calibration, which is what scripts/render_triang_overlay.py
           does and is the right mode for judging calibration/triangulation.
Bottom   : the triangulated 3D skeleton from triang_keys_3D.npy, rendered with
           matplotlib on fixed axes so the skeleton does not jump between
           frames. By default one 3D view is drawn per camera panel, at
           azimuths spread over 180 degrees, so the strip fills the same width
           as the top row and a single view's depth ambiguity is resolved by
           its neighbours (--views 1 for a single rendering). The axes track
           the animal by default (--view3d follow) so the skeleton stays large
           over a long clip; --view3d world fixes them to the whole window
           instead, which shows the path through the arena.

Keypoints are colored by body part (head / spine / each limb), consistently
between the 2D and 3D panels, so a limb can be followed between views.

Two data caveats this script handles explicitly, both real in R6/2026_09_08_1:
  * Native 2D detections carry a confidence channel and some cameras are much
    weaker than others. Keypoints below --conf_thresh are not drawn, and each
    panel prints its own per-frame mean confidence.
  * triang_keys_3D.npy contains triangulation blow-ups (coordinates in the
    millions on frames where the 2D input was bad). Axis limits are therefore
    taken from robust percentiles, and keypoints outside the plotted volume are
    dropped for that frame rather than dragging the view off.
  * SLEAP triangulation stores Z inverted relative to world "up" -- feet sit
    above the head in the raw array, which is why
    clirb.events.project_session_to_pcs flips it. The 3D panel flips it too, so
    the rat renders upright; --no_flip_z shows the raw array. Reprojection for
    --kp2d reproj always uses the raw, unflipped coordinates.

Usage:
    # 50 s clip from the best-tracked part of the session
    python scripts/render_2d3d_session_video.py \
        --rat R6 --session 2026_09_08_1 --start_frame 24000 --n_frames 1000

    # reprojected triangulation instead of native detections, single camera
    python scripts/render_2d3d_session_video.py \
        --rat R6 --session 2026_09_08_1 --kp2d reproj --cameras 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the projection

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from config import DATA_ROOT, EDGES, NODES, sleap_path                # noqa: E402
from corrector.data_world_2d import (load_session_calibration,        # noqa: E402
                                     project_3d_to_2d_batch)

OUT_ROOT = _THIS.parent.parent / "figures" / "session_videos"

# --- body-part colouring, shared by the 2D and 3D panels -------------------
GROUPS = {
    "head":       ([0, 1, 2],        (79, 213, 255)),    # amber-ish, BGR
    "spine":      ([3, 4, 5, 6],     (247, 195, 79)),    # light blue
    "fore-left":  ([7, 8, 9, 10],    (106, 187, 102)),   # green
    "fore-right": ([11, 12, 13, 14], (80, 83, 239)),     # red
    "hind-left":  ([15, 16, 17, 18], (188, 71, 171)),    # purple
    "hind-right": ([19, 20, 21, 22], (67, 112, 255)),    # deep orange
}
GROUP_OF = {}
NODE_BGR = [(255, 255, 255)] * len(NODES)
for _name, (_idx, _bgr) in GROUPS.items():
    for _i in _idx:
        GROUP_OF[_i] = _name
        NODE_BGR[_i] = _bgr
# an edge takes the colour of its distal (higher-index) node, so limb
# attachment bones read as part of the limb
EDGE_BGR = [NODE_BGR[max(e[0], e[1])] for e in EDGES]


def _bgr_to_mpl(bgr):
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


def causal_median(arr: np.ndarray, size: int) -> np.ndarray:
    """Causal median over time: current frame + (size-1) preceding frames.
    Matches the online convention used elsewhere in this repo."""
    if size <= 1:
        return arr
    pad = size - 1
    padded = np.concatenate([np.repeat(arr[:1], pad, axis=0), arr], axis=0)
    idx = np.arange(arr.shape[0])[:, None] + np.arange(size)[None, :]
    return np.median(padded[idx], axis=1).astype(arr.dtype)


def draw_skeleton_2d(img, pts2d, conf=None, conf_thresh=0.0,
                     radius=3, thickness=2):
    """Draw the 23-node skeleton on img. pts2d (23,2) already in img pixels.
    Points whose confidence is below conf_thresh (or which fall outside the
    image) are skipped, and an edge is drawn only if both ends survive."""
    h, w = img.shape[:2]
    pts = np.rint(pts2d).astype(np.int32)
    ok = np.isfinite(pts2d).all(axis=-1)
    ok &= (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if conf is not None:
        ok &= conf >= conf_thresh
    for e, col in zip(EDGES, EDGE_BGR):
        if ok[e[0]] and ok[e[1]]:
            cv2.line(img, tuple(pts[e[0]]), tuple(pts[e[1]]), col,
                     thickness, cv2.LINE_AA)
    for i in range(len(pts)):
        if ok[i]:
            cv2.circle(img, tuple(pts[i]), radius, NODE_BGR[i], -1, cv2.LINE_AA)
    return ok


class Panel3D:
    """One matplotlib figure holding `len(azims)` synchronized 3D views of the
    same skeleton, drawn at different azimuths.

    All views live in a single figure so each output frame costs exactly one
    canvas draw. The artists are built once and only their data is updated --
    rebuilding the figure per frame is what makes naive per-frame matplotlib
    rendering unusably slow.
    """

    def __init__(self, width, height, lo, hi, elev, azims, dpi=100):
        self.w, self.h = width, height
        self.lo, self.hi = lo, hi
        self.elev = elev
        self.azims = list(azims)
        nv = len(self.azims)

        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.fig.patch.set_facecolor("black")

        self.axes, self.edge_lines, self.group_pts = [], [], []
        for vi, az in enumerate(self.azims):
            ax = self.fig.add_subplot(1, nv, vi + 1, projection="3d")
            ax.set_facecolor("black")
            self.edge_lines.append([
                ax.plot([], [], [], "-", color=_bgr_to_mpl(c), lw=2.0)[0]
                for c in EDGE_BGR])
            gp = {}
            for name, (idx, bgr) in GROUPS.items():
                gp[name] = ax.plot([], [], [], "o", color=_bgr_to_mpl(bgr),
                                   ms=4.0, markeredgecolor="none",
                                   label=name)[0]
            self.group_pts.append(gp)

            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            try:  # keeps the rat from looking stretched; mpl >= 3.3
                ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
            except Exception:
                pass
            small = nv > 1
            ax.set_xlabel("X (mm)", color="0.7", fontsize=7 if small else 8)
            ax.set_ylabel("Y (mm)", color="0.7", fontsize=7 if small else 8)
            ax.set_zlabel("Z (mm)", color="0.7", fontsize=7 if small else 8)
            ax.tick_params(colors="0.5", labelsize=6 if small else 7)
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.set_pane_color((0.06, 0.06, 0.06, 1.0))
            ax.view_init(elev=elev, azim=az)
            ax.set_title(f"azim {az:.0f}\u00b0", color="0.6", fontsize=8)
            self.axes.append(ax)

        # one legend for the whole strip
        leg = self.axes[0].legend(loc="upper left", fontsize=7, ncol=2,
                                  facecolor="black", edgecolor="0.3",
                                  labelcolor="0.85",
                                  bbox_to_anchor=(-0.06, 1.02))
        leg.get_frame().set_alpha(0.6)

        self.suptitle = self.fig.suptitle("", color="0.9", fontsize=11, y=0.985)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                                 wspace=0.02)

    def set_limits(self, lo, hi):
        """Move the plotted volume (follow-cam). The span is expected to stay
        constant so the box aspect set in __init__ stays valid."""
        self.lo, self.hi = lo, hi
        for ax in self.axes:
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])

    def render(self, kp, title, azim_offset=None):
        """kp (23,3). Points outside the plotted volume are NaN'd out so a
        triangulation blow-up drops the affected bones instead of dragging the
        whole view off-axis."""
        p = np.asarray(kp, dtype=float).copy()
        inside = np.isfinite(p).all(axis=-1)
        for a in range(3):
            inside &= (p[:, a] >= self.lo[a]) & (p[:, a] <= self.hi[a])
        p[~inside] = np.nan

        for vi in range(len(self.azims)):
            for ln, e in zip(self.edge_lines[vi], EDGES):
                ln.set_data_3d(p[e, 0], p[e, 1], p[e, 2])
            for name, (idx, _bgr) in GROUPS.items():
                self.group_pts[vi][name].set_data_3d(
                    p[idx, 0], p[idx, 1], p[idx, 2])
            if azim_offset is not None:
                self.axes[vi].view_init(
                    elev=self.elev, azim=self.azims[vi] + azim_offset)
        self.suptitle.set_text(title)

        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.shape[0] != self.h or img.shape[1] != self.w:
            img = cv2.resize(img, (self.w, self.h))
        return img, int(inside.sum())

    def close(self):
        plt.close(self.fig)


def render(rat, session, cameras, start_frame, n_frames, fps, kp2d_mode,
           conf_thresh, panel_width, threed_height, elev, azim, spin,
           smooth_size, smooth_causal, view3d, follow_span, flip_z=True,
           n_views=None, h264=True, out_path=None):
    sp = Path(sleap_path(rat, session))

    triang = np.load(sp / "triang_keys_3D.npy").astype(np.float64)
    if triang.ndim != 3 or triang.shape[1:] != (len(NODES), 3):
        raise RuntimeError(f"unexpected triang shape {triang.shape}")
    print(f"triang_keys_3D : {triang.shape}", flush=True)

    kp2d_native = None
    if kp2d_mode == "native":
        f2d = sp / "sleap_keys_2D.npy"
        if not f2d.exists():
            raise FileNotFoundError(
                f"{f2d} not found; rerun with --kp2d reproj to project the "
                "triangulated 3D onto each camera instead")
        kp2d_native = np.load(f2d).astype(np.float64)
        # (T, n_cam, 23, 3) -> x, y, confidence
        if kp2d_native.ndim != 4 or kp2d_native.shape[2] != len(NODES):
            raise RuntimeError(f"unexpected 2D shape {kp2d_native.shape}")
        print(f"sleap_keys_2D  : {kp2d_native.shape} "
              f"(x, y, confidence)", flush=True)
        n_cam_avail = kp2d_native.shape[1]
    else:
        n_cam_avail = 3

    for c in cameras:
        if c < 0 or c >= n_cam_avail:
            raise ValueError(f"camera {c} out of range (have {n_cam_avail})")

    end_frame = min(start_frame + n_frames, len(triang))
    if kp2d_native is not None:
        end_frame = min(end_frame, len(kp2d_native))
    if end_frame <= start_frame:
        raise ValueError(f"empty frame range {start_frame}..{end_frame}")
    rng = slice(start_frame, end_frame)
    n = end_frame - start_frame
    print(f"window         : frames {start_frame}..{end_frame} "
          f"({n} frames, {n / fps:.1f}s at {fps} fps)", flush=True)

    # --- 3D keypoints for the bottom panel -------------------------------
    kp3d = triang[rng]
    if smooth_size > 1:
        which = "causal" if smooth_causal else "symmetric"
        print(f"smoothing 3D   : {which} median, size={smooth_size}", flush=True)
        if smooth_causal:
            kp3d = causal_median(kp3d, smooth_size)
        else:
            from scipy.ndimage import median_filter
            kp3d = median_filter(kp3d, size=(smooth_size, 1, 1))

    # --- 2D keypoints per camera -----------------------------------------
    # SLEAP triangulation output has Z inverted relative to world "up" --
    # clirb.events.project_session_to_pcs flips it before doing anything, and
    # feet sit above the head in the raw array. Flip it for the 3D panel so the
    # rat renders upright, but keep the RAW coordinates for reprojection: the
    # calibration maps the unflipped triangulation back onto the cameras.
    kp3d_disp = kp3d.copy()
    if flip_z:
        kp3d_disp[:, :, 2] = -kp3d_disp[:, :, 2]
        print("3D display     : Z flipped (SLEAP convention, renders upright)",
              flush=True)
    else:
        print("3D display     : raw Z (--no_flip_z; feet will appear above "
              "the head)", flush=True)

    kp2d, conf = {}, {}
    if kp2d_mode == "native":
        for c in cameras:
            kp2d[c] = kp2d_native[rng, c, :, :2]
            conf[c] = kp2d_native[rng, c, :, 2]
    else:
        calib, cal_date = load_session_calibration(rat, session)
        print(f"calibration    : {cal_date}", flush=True)
        for c in cameras:
            # raw (unflipped) coords -- see the flip_z note above
            kp2d[c] = project_3d_to_2d_batch(kp3d, calib[c])
            conf[c] = np.ones(kp2d[c].shape[:2])  # no confidence when reprojecting

    # --- robust 3D axis limits -------------------------------------------
    # triang_keys_3D carries blow-ups reaching millions of mm; percentiles keep
    # the view on the animal. Computed over the rendered window only.
    if view3d == "center":
        kp3d_disp = kp3d_disp - kp3d_disp.mean(axis=1, keepdims=True)

    # Per-frame animal centre, used by the follow-cam. The median over
    # keypoints is robust to the triangulation blow-ups; smoothing over time
    # keeps the view from jittering frame to frame.
    com = np.median(kp3d_disp, axis=1)
    com_track = None
    if view3d == "follow":
        from scipy.ndimage import median_filter
        com_track = median_filter(com, size=(21, 1), mode="nearest")
        half = follow_span / 2.0
        lo = com_track[0] - half
        hi = com_track[0] + half
        print(f"3D view        : follow-cam, {follow_span:.0f} mm box tracking "
              f"the animal centre", flush=True)
    else:
        flat = kp3d_disp.reshape(-1, 3)
        flat = flat[np.isfinite(flat).all(axis=1)]
        lo = np.percentile(flat, 1.0, axis=0)
        hi = np.percentile(flat, 99.0, axis=0)
        pad = 0.10 * np.maximum(hi - lo, 1.0)
        lo, hi = lo - pad, hi + pad
        # equalise the box so the skeleton keeps its proportions
        span = (hi - lo).max()
        mid = 0.5 * (lo + hi)
        lo, hi = mid - span / 2, mid + span / 2
        print(f"3D view        : {view3d}, fixed limits "
              f"X {lo[0]:.0f}..{hi[0]:.0f}  Y {lo[1]:.0f}..{hi[1]:.0f}  "
              f"Z {lo[2]:.0f}..{hi[2]:.0f}", flush=True)

    # --- video captures ---------------------------------------------------
    caps, vid_dims = {}, None
    for c in cameras:
        vf = sp / f"Camera{c}" / "0.mp4"
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {vf}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if vid_dims is None:
            vid_dims = (w, h)
        elif (w, h) != vid_dims:
            raise RuntimeError(f"Camera{c} is {w}x{h}, expected {vid_dims}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        caps[c] = cap
    vid_w, vid_h = vid_dims

    panel_w = panel_width - (panel_width % 2)
    panel_h = int(round(panel_w * vid_h / vid_w))
    panel_h -= panel_h % 2
    scale = panel_w / vid_w
    top_w = panel_w * len(cameras)
    threed_h = threed_height - (threed_height % 2)
    out_w, out_h = top_w, panel_h + threed_h
    print(f"output         : {out_w}x{out_h} "
          f"(top {panel_w}x{panel_h} x{len(cameras)}, 3D {top_w}x{threed_h})",
          flush=True)

    # One 3D view per camera panel by default, at azimuths spread over 180
    # degrees, so the bottom strip fills the same width as the top row and a
    # single viewpoint's depth ambiguity is resolved by its neighbours.
    if n_views is None:
        n_views = len(cameras)
    n_views = max(1, int(n_views))
    azims = ([azim] if n_views == 1
             else [azim + 180.0 * k / n_views for k in range(n_views)])
    print(f"3D views       : {n_views} "
          f"(azim {', '.join(f'{a:.0f}' for a in azims)}; elev {elev:.0f})",
          flush=True)
    panel3d = Panel3D(top_w, threed_h, lo, hi, elev, azims)

    if out_path is None:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        cam_tag = "cam" + "".join(str(c) for c in cameras)
        smooth_tag = (f"_sm{smooth_size}{'c' if smooth_causal else 's'}"
                      if smooth_size > 1 else "")
        out_path = (OUT_ROOT / f"{rat}_{session}_2d3d_{kp2d_mode}_{cam_tag}"
                               f"_v{n_views}_{view3d}{smooth_tag}"
                               f"_f{start_frame}-{end_frame}.mp4")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_path}")

    WHITE, GREY, YELLOW = (255, 255, 255), (170, 170, 170), (0, 255, 255)
    print(f"rendering -> {out_path.name}", flush=True)
    t0 = time.time()
    written = 0
    n_dropped_frames = 0

    for fi in range(n):
        frame_idx = start_frame + fi
        panels, bad = [], False
        for c in cameras:
            ok_read, frame = caps[c].read()
            if not ok_read:
                bad = True
                break
            small = cv2.resize(frame, (panel_w, panel_h),
                               interpolation=cv2.INTER_AREA)
            ok_kp = draw_skeleton_2d(small, kp2d[c][fi] * scale,
                                     conf=conf[c][fi], conf_thresh=conf_thresh)
            cv2.putText(small, f"Camera{c}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
            if kp2d_mode == "native":
                cv2.putText(small,
                            f"conf {conf[c][fi].mean():.2f}  kp {int(ok_kp.sum())}/{len(NODES)}",
                            (8, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, GREY, 1, cv2.LINE_AA)
            panels.append(small)
        if bad:
            print(f"  video ended early at frame {frame_idx}", flush=True)
            break
        top = panels[0] if len(panels) == 1 else np.hstack(panels)

        if com_track is not None:
            half = follow_span / 2.0
            panel3d.set_limits(com_track[fi] - half, com_track[fi] + half)
        az = (360.0 * fi / n) if spin else None
        t_s = frame_idx / fps
        title = (f"{rat}/{session}   triangulated 3D   "
                 f"frame {frame_idx}   t={t_s // 60:.0f}:{t_s % 60:04.1f}")
        bottom, n_in = panel3d.render(kp3d_disp[fi], title, azim_offset=az)
        if n_in < len(NODES):
            n_dropped_frames += 1
            cv2.putText(bottom, f"{len(NODES) - n_in} kp outside plotted volume",
                        (10, threed_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, YELLOW, 1, cv2.LINE_AA)

        combined = np.vstack([top, bottom])
        src = ("native SLEAP 2D detections" if kp2d_mode == "native"
               else "triang_keys_3D reprojected")
        cv2.putText(combined, f"top: {src}", (10, panel_h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
        writer.write(combined)
        written += 1

        if written % 100 == 0 or fi == n - 1:
            el = time.time() - t0
            print(f"  {written}/{n}  {el:.1f}s  "
                  f"({written / max(el, 1e-6):.1f} fps)", flush=True)

    writer.release()
    for c in caps:
        caps[c].release()
    panel3d.close()

    # OpenCV's mp4v is bulky (~1 MB/s at this size); H.264 cuts it ~3x and
    # plays in browsers and slide decks, which mp4v often will not.
    if h264 and written:
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            print("note: ffmpeg not found, leaving the mp4v file as-is",
                  flush=True)
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
                print(f"note: ffmpeg failed ({exc}); keeping the mp4v file",
                      flush=True)

    print(f"\nwrote {out_path}  ({written} frames, "
          f"{written / fps:.1f}s at {fps} fps)", flush=True)
    if n_dropped_frames:
        print(f"note: {n_dropped_frames}/{written} frames had at least one "
              f"keypoint outside the plotted volume (triangulation outliers)",
              flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--cameras", default="0,1,2",
                    help="comma-separated camera indices for the top row "
                         "(default 0,1,2)")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--kp2d", choices=("native", "reproj"), default="native",
                    help="top-row keypoints: 'native' = sleap_keys_2D.npy "
                         "per-camera detections (default); 'reproj' = "
                         "triang_keys_3D.npy projected through calibration")
    ap.add_argument("--conf_thresh", type=float, default=0.2,
                    help="hide native 2D keypoints below this confidence "
                         "(default 0.2; ignored for --kp2d reproj)")
    ap.add_argument("--panel_width", type=int, default=640,
                    help="width of each camera panel in px (default 640)")
    ap.add_argument("--threed_height", type=int, default=760,
                    help="height of the 3D panel in px (default 760)")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=45.0)
    ap.add_argument("--spin", action="store_true",
                    help="rotate the 3D view(s) once over the clip")
    ap.add_argument("--views", type=int, default=None,
                    help="number of 3D viewpoints side by side, at azimuths "
                         "spread over 180 degrees (default: one per camera "
                         "panel, so the bottom strip fills the width; use "
                         "--views 1 for a single 3D rendering)")
    ap.add_argument("--smooth_size", type=int, default=1,
                    help="median filter the 3D keypoints over time "
                         "(default 1 = off)")
    ap.add_argument("--smooth_causal", action="store_true",
                    help="make --smooth_size causal (online convention)")
    ap.add_argument("--no_flip_z", action="store_true",
                    help="do NOT negate Z for the 3D panel. SLEAP "
                         "triangulation stores Z inverted, so the default "
                         "flip is what makes the rat render upright; this "
                         "flag shows the raw array instead")
    ap.add_argument("--view3d", choices=("follow", "world", "center"),
                    default="follow",
                    help="how the 3D axes are framed. 'follow' (default) "
                         "keeps a fixed-size box centred on the animal, so "
                         "the skeleton stays large while the tick labels "
                         "still show where it is; 'world' fixes the box to "
                         "the whole window, showing the path through the "
                         "arena but shrinking the skeleton; 'center' "
                         "subtracts the per-frame mean, showing pose only")
    ap.add_argument("--follow_span", type=float, default=350.0,
                    help="edge length in mm of the --view3d follow box "
                         "(default 350; a rat spans roughly 220)")
    ap.add_argument("--no_h264", action="store_true",
                    help="skip the ffmpeg H.264 re-encode and leave OpenCV's "
                         "much larger mp4v output")
    ap.add_argument("--out", default=None, help="output .mp4 path")
    a = ap.parse_args()

    cameras = [int(x) for x in a.cameras.split(",") if x.strip() != ""]
    if not cameras:
        raise SystemExit("--cameras must name at least one camera")

    render(a.rat, a.session, cameras, a.start_frame, a.n_frames, a.fps,
           a.kp2d, a.conf_thresh, a.panel_width, a.threed_height,
           a.elev, a.azim, a.spin, a.smooth_size, a.smooth_causal,
           a.view3d, a.follow_span, not a.no_flip_z, a.views,
           not a.no_h264, a.out)


if __name__ == "__main__":
    main()
