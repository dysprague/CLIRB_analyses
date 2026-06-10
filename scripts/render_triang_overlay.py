"""Render triang_keys_3D.npy reprojected onto SLEAP camera videos.

No corrector inference — just project whatever is in triang_keys_3D.npy
through the per-session calibration onto Camera0/1/2 and overlay on the
source .mp4. Use for visual sanity-check of online corrector output without
re-running the network.

Optionally applies a causal median filter to the keypoints before projection
(default: size 5, causal — matches the online causal-median convention).

By default emits a three-panel side-by-side video with Cameras 0, 1, 2 from
left to right. Pass --camera N to render a single-camera view instead.

Usage:
    # Three-panel triptych, causal-5 smoothing (default)
    python scripts/render_triang_overlay.py --rat R4 --session 2026_06_09_1

    # Single camera, no smoothing
    python scripts/render_triang_overlay.py --rat R4 --session 2026_06_09_1 \
        --camera 0 --smooth_size 1

    # Single camera, causal-5
    python scripts/render_triang_overlay.py --rat R4 --session 2026_06_09_1 \
        --camera 0 --smooth_size 5 --smooth_causal
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from config import DATA_ROOT, calibration_path, EDGES                       # noqa: E402
from data_io import load_sleap_keys_3d                                       # noqa: E402
from corrector.data_world_2d import (load_session_calibration,               # noqa: E402
                                       project_3d_to_2d_batch)


OUT_ROOT = Path(__file__).resolve().parent.parent / "corrector" / "videos"


def _draw_skel(img, pts2d, color, edges=EDGES, radius=4, thickness=1):
    pts = pts2d.astype(np.int32)
    h, w = img.shape[:2]
    valid = (np.isfinite(pts2d).all(axis=-1)
             & (pts[:, 0] >= 0) & (pts[:, 0] < w)
             & (pts[:, 1] >= 0) & (pts[:, 1] < h))
    for e in edges:
        if valid[e[0]] and valid[e[1]]:
            cv2.line(img, tuple(pts[e[0]]), tuple(pts[e[1]]),
                     color, thickness, cv2.LINE_AA)
    for i in range(len(pts)):
        if valid[i]:
            cv2.circle(img, tuple(pts[i]), radius, color, -1, cv2.LINE_AA)


def _smooth_keypoints(triang: np.ndarray, smooth_size: int,
                       smooth_causal: bool) -> np.ndarray:
    """Apply a median filter along the time axis. Matches the convention used
    by `evaluate_all.correct_temporal_mlp_2d_reproj`: a causal filter looks
    only at the current + (smooth_size - 1) preceding frames, padded at the
    session start by repeating the first frame.
    """
    if smooth_size <= 1:
        return triang
    if smooth_causal:
        pad = smooth_size - 1
        padded = np.concatenate(
            [np.repeat(triang[:1], pad, axis=0), triang], axis=0)
        T = triang.shape[0]
        widx = np.arange(T)[:, None] + np.arange(smooth_size)[None, :]
        return np.median(padded[widx], axis=1).astype(np.float32)
    else:
        from scipy.ndimage import median_filter
        return median_filter(triang, size=(smooth_size, 1, 1)).astype(np.float32)


def render(rat: str, session: str, cameras: list[int], start_frame: int,
           n_frames: int, fps: int, smooth_size: int, smooth_causal: bool,
           out_subdir_name: str):
    triang_path = (Path(DATA_ROOT) / rat / session / "sleap" /
                   "triang_keys_3D.npy")
    print(f"loading {triang_path}", flush=True)
    triang = np.load(triang_path).astype(np.float32)
    if triang.ndim != 3 or triang.shape[1:] != (23, 3):
        raise RuntimeError(f"unexpected triang shape: {triang.shape}")
    print(f"  triang shape: {triang.shape}", flush=True)

    if smooth_size > 1:
        which = "causal" if smooth_causal else "symmetric"
        print(f"  applying {which} median filter, size={smooth_size}", flush=True)
        triang = _smooth_keypoints(triang, smooth_size, smooth_causal)

    calib, cal_date = load_session_calibration(rat, session)
    print(f"  calibration date: {cal_date}", flush=True)
    for c in cameras:
        if c < 0 or c >= len(calib):
            raise ValueError(f"camera {c} out of range (have {len(calib)})")

    end_frame = min(start_frame + n_frames, len(triang))
    rng = slice(start_frame, end_frame)

    # Project once per camera, up-front. Cheap relative to the per-frame
    # video read/draw/write.
    print(f"projecting {end_frame - start_frame} frames to cameras {cameras}...",
          flush=True)
    kp2d_per_cam = {}
    for c in cameras:
        kp2d_per_cam[c] = project_3d_to_2d_batch(triang[rng], calib[c])

    # Open one VideoCapture per requested camera.
    caps = {}
    vid_dims = None
    for c in cameras:
        video_file = (Path(DATA_ROOT) / rat / session / "sleap" /
                      f"Camera{c}" / "0.mp4")
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {video_file}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if vid_dims is None:
            vid_dims = (w, h)
        elif (w, h) != vid_dims:
            raise RuntimeError(f"camera {c} has dims {(w,h)} != expected {vid_dims}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        caps[c] = cap
    vid_w, vid_h = vid_dims

    out_subdir = OUT_ROOT / out_subdir_name
    out_subdir.mkdir(parents=True, exist_ok=True)
    cam_tag = ("cam" + "".join(str(c) for c in cameras)
               if len(cameras) > 1 else f"cam{cameras[0]}")
    smooth_tag = (
        f"_smooth{smooth_size}{'causal' if smooth_causal else 'sym'}"
        if smooth_size > 1 else "")
    out_path = (out_subdir /
                f"{rat}_{session}_{cam_tag}{smooth_tag}_"
                f"f{start_frame}-{end_frame}.mp4")
    out_w = vid_w * len(cameras)
    out_h = vid_h
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_path}")

    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    n = end_frame - start_frame
    print(f"rendering {out_path.name}  ({out_w}x{out_h}) ...", flush=True)
    t0 = time.time()
    for fi in range(n):
        panels = []
        bad_read = False
        for c in cameras:
            ok, frame = caps[c].read()
            if not ok:
                bad_read = True
                break
            _draw_skel(frame, kp2d_per_cam[c][fi], GREEN)
            label = f"Camera{c}"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)
            panels.append(frame)
        if bad_read:
            break
        combined = panels[0] if len(panels) == 1 else np.hstack(panels)
        # Overlay session info on the leftmost panel.
        title = f"{rat}/{session}  triang_keys_3D"
        if smooth_size > 1:
            title += (f"  ({'causal' if smooth_causal else 'sym'}"
                      f"-{smooth_size} median)")
        cv2.putText(combined, title, (10, vid_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(combined, f"frame {start_frame + fi}", (10, vid_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        writer.write(combined)
        if (fi + 1) % 200 == 0 or fi == n - 1:
            print(f"  {fi + 1}/{n} ({time.time() - t0:.1f}s)", flush=True)
    writer.release()
    for c in caps:
        caps[c].release()
    print(f"saved {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--camera", type=int, default=None,
                    help="single camera index (0/1/2). If omitted, defaults "
                         "to a 3-panel triptych (cameras 0,1,2 side-by-side).")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--smooth_size", type=int, default=5,
                    help="median filter size along time (default 5). "
                         "Set to 1 to disable.")
    ap.add_argument("--smooth_causal", action="store_true", default=True,
                    help="use causal (past+current only) median. Default True.")
    ap.add_argument("--no_smooth_causal", dest="smooth_causal",
                    action="store_false",
                    help="use a symmetric median instead of causal.")
    ap.add_argument("--out_subdir", type=str,
                    default="online_corrector_runs")
    args = ap.parse_args()
    cameras = [args.camera] if args.camera is not None else [0, 1, 2]
    render(args.rat, args.session, cameras, args.start_frame, args.n_frames,
           args.fps, args.smooth_size, args.smooth_causal, args.out_subdir)


if __name__ == "__main__":
    main()
