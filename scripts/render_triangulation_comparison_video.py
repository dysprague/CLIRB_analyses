"""Three-way overlay: DANNCE, the rig's equal-weight triangulation, and the
conf^2 confidence-weighted triangulation, all reprojected into the three SLEAP
cameras.

  MAGENTA  DANNCE, Procrustes-mapped into SLEAP's world frame
  CYAN     SLEAP, equal-weight DLT -- exactly what `triang_keys_3D` contains
  YELLOW   SLEAP, conf^2-weighted DLT -- the proposed change

Magenta / cyan / yellow are the three subtractive primaries: maximally distinct
from one another, and none of them is green, which the arena floor and the
backdrop are.

The Procrustes fit maps DANNCE onto the EQUAL-weight SLEAP, and that single
transform is used for both SLEAP variants, so neither variant gets an alignment
tuned to itself and the comparison between them is fair.

The weighted variant is re-triangulated on the processed frames and its gaps
filled by the same linear interpolation the rig applies, so both SLEAP series
are contiguous and on the same acquisition grid.

SMOOTHING.  By default all three skeletons get `ev.causal_median_5_kp(...,
warmup='online')` in world space before projection -- the same 5-frame causal
median the online rule applies inside `TemplateMatch.corrections`/`HeadHold`
and that every offline feature and matching script applies via
`ev.project_session_to_pcs`.  So what is drawn is what the rule actually acts
on, not the raw triangulation.  `--smooth none` shows the raw series instead.
The per-frame RMSE in the HUD is computed on whichever series is drawn.

By default the window is chosen automatically as the stretch where the two
variants differ most, since a window where they agree shows nothing.

Usage:
    python scripts/render_triangulation_comparison_video.py \
        --rat R7 --session 2026_10_26_1 --n_frames 1200
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import session_cache as sc                                        # noqa: E402
from clirb.spatial import dual as du                              # noqa: E402
from clirb.spatial import quality as ql                           # noqa: E402
from config import DATA_ROOT, EDGES                               # noqa: E402
from corrector.data_world_2d import (load_session_calibration,     # noqa: E402
                                      project_3d_to_2d_batch)
from clirb import events as ev                                     # noqa: E402
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample  # noqa: E402
from scripts.newcal_weighted_reanalysis import dlt_chunked, fill_gaps      # noqa: E402

OUT_ROOT = REPO / 'results' / 'newcal' / 'videos'
C_DANNCE = (255, 0, 255)     # magenta
C_EQUAL = (255, 255, 0)      # cyan
C_WEIGHTED = (0, 255, 255)   # yellow
EXPONENT = 2.0


def _draw(img, pts2d, color, radius=3, thickness=1):
    pts = np.nan_to_num(pts2d, nan=-1e6).astype(np.int32)
    h, w = img.shape[:2]
    ok = (np.isfinite(pts2d).all(axis=-1) & (pts[:, 0] >= 0) & (pts[:, 0] < w)
          & (pts[:, 1] >= 0) & (pts[:, 1] < h))
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    for i in range(len(pts)):
        if ok[i]:
            cv2.circle(img, tuple(pts[i]), radius, color, -1, cv2.LINE_AA)


def smooth(kp):
    """The online rule's own filter: 5-frame causal median in keypoint space,
    first four frames left unfiltered exactly as the rule leaves them while its
    buffer fills."""
    return ev.causal_median_5_kp(np.asarray(kp, np.float32), warmup='online')


def build(rat, session, smoothing='causal5'):
    raw = sc.get_raw_session(rat, session, want_dual=True)
    d3 = du.dannce_aligned(raw)
    sl_eq = np.asarray(raw['triang_3d'], np.float32)
    n = min(len(sl_eq), len(d3))
    sl_eq = sl_eq[:n]
    dn = np.asarray(d3[:n], np.float32)

    k2 = np.asarray(raw['sleap_keys_2D'])
    fm = np.asarray(raw['frame_mapping_cam0'], int)
    nproc = min(len(fm), len(k2))
    P, K, D, R, T = ql.load_calib(ql.calib_dir(rat, session))
    und = ql.undistort_all(k2[:nproc, :, :, :2].astype(np.float64), K, D)
    conf = np.clip(k2[:nproc, :, :, 2].astype(np.float64), 0, None)
    w = conf ** EXPONENT
    under = (w > 0).sum(axis=1) < 2
    w[np.broadcast_to(under[:, None, :], w.shape)] = 1.0      # on_underdetermined: equal
    kp = dlt_chunked(und, P, w)

    sl_wt = np.full_like(sl_eq, np.nan)
    fmp = fm[:nproc]
    ok = (fmp >= 0) & (fmp < n) & np.isfinite(kp).all(axis=(1, 2))[:len(fmp)]
    sl_wt[fmp[ok]] = kp[ok]
    known = np.zeros(n, bool); known[fmp[ok]] = True
    sl_wt = fill_gaps(sl_wt, known)

    # one alignment, fitted on the equal-weight variant, used for both
    samp, _ = balanced_sample(np.median(sl_eq, axis=1)[:, :2],
                              rng=np.random.default_rng(0))
    fit = robust_proc(dn[samp], sl_eq[samp])
    dn_w = fit['apply'](dn).astype(np.float32)
    print(f'  Procrustes DANNCE->SLEAP: scale={fit["s"]:.4f} '
          f'residual={fit["residual"]:.2f} mm', flush=True)

    if smoothing == 'causal5':
        sl_eq, sl_wt, dn_w = smooth(sl_eq), smooth(sl_wt), smooth(dn_w)

    e_rmse = np.sqrt(((sl_eq - dn_w) ** 2).sum(-1).mean(axis=1))
    w_rmse = np.sqrt(((sl_wt - dn_w) ** 2).sum(-1).mean(axis=1))
    print(f'  whole session: equal median {np.nanmedian(e_rmse):.1f} mm, '
          f'conf^2 median {np.nanmedian(w_rmse):.1f} mm', flush=True)
    return sl_eq, sl_wt, dn_w, e_rmse, w_rmse, known


def pick_window(e_rmse, w_rmse, n_frames):
    """The stretch where the two variants differ most -- a window where they
    agree would show nothing."""
    d = np.nan_to_num(e_rmse - w_rmse, nan=0.0)
    k = np.ones(n_frames) / n_frames
    sm = np.convolve(d, k, mode='valid')
    return int(np.argmax(sm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rat', default='R7')
    ap.add_argument('--session', default='2026_10_26_1')
    ap.add_argument('--start_frame', type=int, default=None,
                    help='omit to pick the most informative window automatically')
    ap.add_argument('--n_frames', type=int, default=1200)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--smooth', choices=['causal5', 'none'], default='causal5',
                    help="'causal5' (default) applies the online rule's own "
                         "5-frame causal median to all three skeletons")
    a = ap.parse_args()
    cams = [0, 1, 2]

    print(f'{a.rat}/{a.session}', flush=True)
    sl_eq, sl_wt, dn_w, e_rmse, w_rmse, known = build(a.rat, a.session, a.smooth)
    n = len(sl_eq)
    start = (a.start_frame if a.start_frame is not None
             else pick_window(e_rmse, w_rmse, a.n_frames))
    end = min(start + a.n_frames, n)
    rng = slice(start, end)
    print(f'  window {start}-{end} ({start/a.fps:.0f}-{end/a.fps:.0f} s): '
          f'equal {np.nanmedian(e_rmse[rng]):.1f} mm vs '
          f'conf^2 {np.nanmedian(w_rmse[rng]):.1f} mm', flush=True)

    calib, cal_date = load_session_calibration(a.rat, a.session)
    proj = {c: {k: project_3d_to_2d_batch(v[rng], calib[c])
                for k, v in [('dn', dn_w), ('eq', sl_eq), ('wt', sl_wt)]}
            for c in cams}

    caps, dims = {}, None
    for c in cams:
        vf = Path(DATA_ROOT) / a.rat / a.session / 'sleap' / f'Camera{c}' / '0.mp4'
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            raise FileNotFoundError(vf)
        dims = dims or (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        caps[c] = cap
    vw, vh = dims
    ow, oh = int(vw * len(cams) * a.scale), int(vh * a.scale)
    ow -= ow % 2; oh -= oh % 2

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tag = 'med5' if a.smooth == 'causal5' else 'raw'
    out = OUT_ROOT / (f'{a.rat}_{a.session}_triangcompare_{tag}_cam012_'
                      f'f{start}-{end}.mp4')
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'),
                             a.fps, (ow, oh))
    print(f'  rendering {out.name} ({ow}x{oh})...', flush=True)
    t0 = time.time()
    for fi in range(end - start):
        panels = []
        for c in cams:
            ok, frame = caps[c].read()
            if not ok:
                break
            # DANNCE first and widest, so it reads as a halo under the two
            # SLEAP variants rather than being hidden by them; the variants go
            # on top, thinner, with the proposed one last.
            _draw(frame, proj[c]['dn'][fi], C_DANNCE, radius=5, thickness=3)
            _draw(frame, proj[c]['eq'][fi], C_EQUAL, radius=3, thickness=2)
            _draw(frame, proj[c]['wt'][fi], C_WEIGHTED, radius=2, thickness=1)
            panels.append(frame)
        if len(panels) != len(cams):
            break
        comb = np.hstack(panels)
        comb = cv2.resize(comb, (ow, oh), interpolation=cv2.INTER_AREA)
        pw = ow // len(cams)
        for i, c in enumerate(cams):
            cv2.putText(comb, f'Camera{c}', (i * pw + 8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        gf = start + fi
        cv2.putText(comb, f'{a.rat}/{a.session}   cal {cal_date}   '
                          f'{"causal-5 median" if a.smooth == "causal5" else "raw"}',
                    (8, oh - 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(comb, 'DANNCE (6 cam, Procrustes to SLEAP world)', (8, oh - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DANNCE, 1, cv2.LINE_AA)
        cv2.putText(comb, f'SLEAP equal weight (the rig)   '
                          f'{e_rmse[gf]:5.1f} mm', (8, oh - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_EQUAL, 1, cv2.LINE_AA)
        cv2.putText(comb, f'SLEAP conf^2 weighted          '
                          f'{w_rmse[gf]:5.1f} mm', (8, oh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WEIGHTED, 1, cv2.LINE_AA)
        if not known[gf]:
            cv2.putText(comb, 'interpolated frame', (ow - 160, oh - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        writer.write(comb)
        if (fi + 1) % 250 == 0:
            print(f'    {fi+1}/{end-start} ({time.time()-t0:.0f}s)', flush=True)
    writer.release()
    for c in caps:
        caps[c].release()

    import shutil, subprocess
    if shutil.which('ffmpeg'):
        tmp = out.with_suffix('.h264.mp4')
        try:
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(out),
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                            '-pix_fmt', 'yuv420p', str(tmp)], check=True)
            os.replace(tmp, out)
        except Exception as e:
            print(f'  transcode failed ({e!r}); keeping mp4v', flush=True)
    print(f'  saved {out} ({out.stat().st_size/1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
