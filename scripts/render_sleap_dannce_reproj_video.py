"""Overlay SLEAP and DANNCE 3D keypoints, reprojected into all three SLEAP
camera spaces, on the SLEAP camera videos.

Both skeletons are drawn on the same frame of the same camera:
  GREEN   SLEAP  -- `sleap/triang_keys_3D.npy`, the online 3-camera
                    triangulation, projected with this session's own SLEAP
                    calibration (the cameras it was triangulated from).
  MAGENTA DANNCE -- `processed/sleap_dannce_keys.mat`, resampled onto the SLEAP
                    frame grid, then mapped into SLEAP's world frame by ONE
                    session-level rigid + isotropic-scale Procrustes fit
                    (spatially balanced sample, outlier-trimmed -- the
                    stage20 / section-8q convention) and projected through the
                    same SLEAP calibration.

The Procrustes step removes only the rigid SLEAP-rig-vs-DANNCE-rig calibration
offset; it is a single transform for the whole session and cannot absorb any
per-frame pose disagreement.  Everything still visible between the two
skeletons is genuine tracking disagreement.

Per-frame HUD shows that frame's SLEAP-vs-DANNCE RMSE over the 23 keypoints.

Usage:
    python scripts/render_sleap_dannce_reproj_video.py \
        --rat R5 --session 2026_10_14_1 --start_frame 0 --n_frames 1000
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import session_cache as sc                                     # noqa: E402
from clirb.spatial import dual as du                           # noqa: E402
from config import DATA_ROOT, EDGES                            # noqa: E402
from corrector.data_world_2d import (load_session_calibration,  # noqa: E402
                                      project_3d_to_2d_batch)
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample  # noqa: E402

OUT_ROOT = REPO / 'results' / 'newcal' / 'videos'

SLEAP_COLOR = (0, 255, 0)        # BGR green
DANNCE_COLOR = (255, 0, 255)     # BGR magenta


def _draw_skel(img, pts2d, color, radius=3, thickness=1):
    pts = pts2d.astype(np.int32)
    h, w = img.shape[:2]
    valid = (np.isfinite(pts2d).all(axis=-1)
             & (pts[:, 0] >= 0) & (pts[:, 0] < w)
             & (pts[:, 1] >= 0) & (pts[:, 1] < h))
    for a, b in EDGES:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    for i in range(len(pts)):
        if valid[i]:
            cv2.circle(img, tuple(pts[i]), radius, color, -1, cv2.LINE_AA)


def causal_median(kp, size):
    if size <= 1:
        return kp
    pad = size - 1
    padded = np.concatenate([np.repeat(kp[:1], pad, axis=0), kp], axis=0)
    widx = np.arange(len(kp))[:, None] + np.arange(size)[None, :]
    return np.median(padded[widx], axis=1).astype(np.float32)


def h264_transcode(path, crf=23):
    """Re-encode the mp4v draft to H.264 (much smaller, plays everywhere)."""
    import shutil, subprocess
    if shutil.which('ffmpeg') is None:
        return path
    tmp = path.with_suffix('.h264.mp4')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(path),
           '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', str(tmp)]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f'  ffmpeg transcode failed ({e!r}); keeping mp4v', flush=True)
        return path
    os.replace(tmp, path)
    return path


def render(rat, session, cameras, start_frame, n_frames, fps, smooth_size,
           out_dir=None, tag='', scale=1.0):
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw is None or not raw.get('has_dual', False):
        raise RuntimeError(f'{rat}/{session}: no dual SLEAP+DANNCE data')
    d3 = du.dannce_aligned(raw)
    n_tot = min(len(raw['triang_3d']), len(d3))
    sl = np.asarray(raw['triang_3d'][:n_tot], dtype=np.float32)
    dn = np.asarray(d3[:n_tot], dtype=np.float32)

    # One session-level Procrustes: DANNCE -> SLEAP world frame.
    samp, _ = balanced_sample(sl.mean(axis=1)[:, :2],
                              rng=np.random.default_rng(0))
    fit = robust_proc(dn[samp], sl[samp])
    dn_w = fit['apply'](dn).astype(np.float32)
    print(f'  Procrustes DANNCE->SLEAP: scale={fit["s"]:.4f} '
          f'z_flip={fit["z_flipped"]} residual={fit["residual"]:.2f}mm '
          f'(fit on {len(samp)} frames)', flush=True)

    if smooth_size > 1:
        sl = causal_median(sl, smooth_size)
        dn_w = causal_median(dn_w, smooth_size)

    frame_rmse = np.sqrt((((sl - dn_w) ** 2).sum(axis=-1)).mean(axis=1))
    print(f'  session RMSE={np.sqrt((((sl-dn_w)**2).sum(axis=-1)).mean()):.2f}mm  '
          f'per-frame median={np.median(frame_rmse):.2f}mm', flush=True)

    calib, cal_date = load_session_calibration(rat, session)
    print(f'  calibration: {cal_date}', flush=True)

    end_frame = min(start_frame + n_frames, n_tot)
    rng = slice(start_frame, end_frame)
    sl2d = {c: project_3d_to_2d_batch(sl[rng], calib[c]) for c in cameras}
    dn2d = {c: project_3d_to_2d_batch(dn_w[rng], calib[c]) for c in cameras}

    caps, dims = {}, None
    for c in cameras:
        vf = Path(DATA_ROOT) / rat / session / 'sleap' / f'Camera{c}' / '0.mp4'
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            raise FileNotFoundError(vf)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dims = dims or (w, h)
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        caps[c] = cap
    vw, vh = dims

    out_dir = Path(out_dir) if out_dir else OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_tag = 'cam' + ''.join(str(c) for c in cameras)
    sm = f'_med{smooth_size}' if smooth_size > 1 else '_raw'
    out_path = out_dir / f'{rat}_{session}_{cam_tag}{sm}_f{start_frame}-{end_frame}{tag}.mp4'
    ow, oh = int(vw * len(cameras) * scale), int(vh * scale)
    ow -= ow % 2; oh -= oh % 2
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (ow, oh))
    if not writer.isOpened():
        raise RuntimeError(f'cannot open writer {out_path}')

    n = end_frame - start_frame
    print(f'  rendering {out_path.name} ({ow}x{oh}, {n} frames)...', flush=True)
    t0 = time.time()
    for fi in range(n):
        panels = []
        for c in cameras:
            ok, frame = caps[c].read()
            if not ok:
                break
            _draw_skel(frame, dn2d[c][fi], DANNCE_COLOR)
            _draw_skel(frame, sl2d[c][fi], SLEAP_COLOR)
            panels.append(frame)
        if len(panels) != len(cameras):
            break
        comb = panels[0] if len(panels) == 1 else np.hstack(panels)
        if (comb.shape[1], comb.shape[0]) != (ow, oh):
            comb = cv2.resize(comb, (ow, oh), interpolation=cv2.INTER_AREA)
        # HUD is drawn AFTER the resize so it stays legible at any scale.
        pw = ow // len(cameras)
        for i, c in enumerate(cameras):
            cv2.putText(comb, f'Camera{c}', (i * pw + 8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        gf = start_frame + fi
        cv2.putText(comb, f'{rat}/{session}   cal {cal_date}', (8, oh - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(comb, 'SLEAP (3 cam)', (8, oh - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, SLEAP_COLOR, 1, cv2.LINE_AA)
        cv2.putText(comb, 'DANNCE (6 cam, Procrustes to SLEAP world)', (150, oh - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, DANNCE_COLOR, 1, cv2.LINE_AA)
        cv2.putText(comb, f'frame {gf}   RMSE {frame_rmse[gf]:6.1f} mm', (8, oh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(comb)
        if (fi + 1) % 250 == 0 or fi == n - 1:
            print(f'    {fi+1}/{n} ({time.time()-t0:.1f}s)', flush=True)
    writer.release()
    for c in caps:
        caps[c].release()
    h264_transcode(out_path)
    print(f'  saved {out_path} '
          f'({out_path.stat().st_size/1e6:.1f} MB)', flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rat', required=True)
    ap.add_argument('--session', required=True)
    ap.add_argument('--camera', type=int, default=None,
                    help='single camera; default is the 0/1/2 triptych')
    ap.add_argument('--start_frame', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=1000)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--smooth_size', type=int, default=1,
                    help='causal median window applied to BOTH modalities '
                         '(1 = raw, the default: show tracking as stored)')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--scale', type=float, default=0.5,
                    help='output scale factor (default 0.5: 2880x600 triptych)')
    a = ap.parse_args()
    cams = [a.camera] if a.camera is not None else [0, 1, 2]
    print(f'{a.rat}/{a.session}', flush=True)
    render(a.rat, a.session, cams, a.start_frame, a.n_frames, a.fps,
           a.smooth_size, a.out_dir, a.tag, a.scale)


if __name__ == '__main__':
    main()
