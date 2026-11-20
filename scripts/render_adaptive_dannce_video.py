"""Three-way overlay: baseline sDANNCE, an adaptive sDANNCE pass, and the
conf^2-weighted SLEAP triangulation, all reprojected into the three SLEAP
cameras.

  MAGENTA  baseline 6-camera sDANNCE (dannce/DANNCE/predict_results_finetune)
  CYAN     adaptive pass (default pass_c2d6_c5d4: Camera2<-Camera6, Camera5<-Camera4)
  YELLOW   SLEAP, conf^2-weighted DLT -- the reference

Colour convention matches render_triangulation_comparison_video.py: magenta is
the DANNCE family, yellow is the conf^2 SLEAP series.  Magenta / cyan / yellow
are the three subtractive primaries -- maximally distinct, and none of them is
green, which the arena floor and backdrop are.

ALIGNMENT.  Each DANNCE variant gets its OWN robust rigid+isotropic-scale
Procrustes onto the conf^2 SLEAP, estimated on a sector-balanced sample --
the same convention as scripts/newcal_adaptive_dannce_r9.py, so the per-frame
RMSE in the HUD reproduces the numbers in that appendix.  Fitting each variant
separately is the fair comparison here: the two predictions have different
world-frame registrations and neither should inherit the other's.

SMOOTHING.  By default all three skeletons get `ev.causal_median_5_kp(...,
warmup='online')` in world space before projection -- the 5-frame causal median
the online rule applies and every offline feature script applies through
`ev.project_session_to_pcs`.  `--smooth none` draws the raw series.

The default window is chosen automatically as the stretch where the baseline
most needs fixing (largest mean baseline-minus-adaptive per-frame RMSE), since
a window where they agree shows nothing.  That window is by construction the
most flattering minute in the session -- pass --start_frame for a
representative one and judge the method on that.

Usage:
    python scripts/render_adaptive_dannce_video.py --rat R9 --session 2026_10_28_1
    python scripts/render_adaptive_dannce_video.py --start_frame 18000 --n_frames 1200
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import session_cache as sc                                        # noqa: E402
from clirb.spatial import quality as ql                           # noqa: E402
from config import DATA_ROOT, EDGES                               # noqa: E402
from corrector.data_world_2d import (load_session_calibration,     # noqa: E402
                                      project_3d_to_2d_batch)
from clirb import events as ev                                     # noqa: E402
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample  # noqa: E402
from scripts.newcal_weighted_reanalysis import dlt_chunked, fill_gaps      # noqa: E402

OUT_ROOT = REPO / 'results' / 'newcal' / 'videos'
C_BASE = (255, 0, 255)       # magenta -- baseline DANNCE
C_ADAPT = (255, 255, 0)      # cyan    -- adaptive DANNCE
C_SLEAP = (0, 255, 255)      # yellow  -- conf^2 SLEAP
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
    return ev.causal_median_5_kp(np.asarray(kp, np.float32), warmup='online')


def load_pred(path):
    m = sio.loadmat(path, simplify_cells=True)
    return np.asarray(m['pred'], np.float64).transpose(0, 2, 1)   # (T, 23, 3)


def build(rat, session, pass_name, smoothing='causal5'):
    sess = Path(DATA_ROOT) / rat / session
    raw = sc.get_raw_session(rat, session, want_dual=True)
    idx = np.asarray(raw['dannce_idx_for_sleap_cams'], int).ravel()[1:]
    sl_eq = np.asarray(raw['triang_3d'], np.float32)
    n = min(len(sl_eq), len(idx))
    idx = idx[:n]

    # conf^2-weighted SLEAP on the contiguous acquisition grid
    k2 = np.asarray(raw['sleap_keys_2D'])
    fm = np.asarray(raw['frame_mapping_cam0'], int)
    nproc = min(len(fm), len(k2))
    P, K, D, R, T = ql.load_calib(ql.calib_dir(rat, session))
    und = ql.undistort_all(k2[:nproc, :, :, :2].astype(np.float64), K, D)
    conf = np.clip(k2[:nproc, :, :, 2].astype(np.float64), 0, None)
    w = conf ** EXPONENT
    under = (w > 0).sum(axis=1) < 2
    w[np.broadcast_to(under[:, None, :], w.shape)] = 1.0
    kp = dlt_chunked(und, P, w)
    sl_wt = np.full((n, kp.shape[1], 3), np.nan, np.float32)
    fmp = fm[:nproc]
    ok = (fmp >= 0) & (fmp < n) & np.isfinite(kp).all(axis=(1, 2))[:len(fmp)]
    sl_wt[fmp[ok]] = kp[ok]
    known = np.zeros(n, bool); known[fmp[ok]] = True
    sl_wt = fill_gaps(sl_wt, known)

    samp, _ = balanced_sample(np.median(sl_wt, axis=1)[:, :2],
                              rng=np.random.default_rng(0))
    paths = {
        'base': sess / 'dannce' / 'DANNCE' / 'predict_results_finetune' / 'save_data_AVG.mat',
        'adapt': sess / 'dannce_adaptive' / pass_name / 'DANNCE' / 'predict_results' / 'save_data_AVG.mat',
    }
    out = {}
    for tag, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(p)
        d = load_pred(str(p))
        d = d[np.clip(idx, 0, len(d) - 1)].astype(np.float32)
        fit = robust_proc(d[samp], sl_wt[samp])
        out[tag] = fit['apply'](d).astype(np.float32)
        print(f'  Procrustes {tag:5s} -> conf^2 SLEAP: scale={fit["s"]:.4f} '
              f'residual={fit["residual"]:.2f} mm', flush=True)

    if smoothing == 'causal5':
        sl_wt = smooth(sl_wt)
        out = {k: smooth(v) for k, v in out.items()}

    b_rmse = np.sqrt(((out['base'] - sl_wt) ** 2).sum(-1).mean(axis=1))
    a_rmse = np.sqrt(((out['adapt'] - sl_wt) ** 2).sum(-1).mean(axis=1))
    print(f'  whole session vs conf^2 SLEAP: baseline median '
          f'{np.nanmedian(b_rmse):.1f} mm, adaptive median '
          f'{np.nanmedian(a_rmse):.1f} mm', flush=True)
    return sl_wt, out['base'], out['adapt'], b_rmse, a_rmse, known


def pick_window(b_rmse, a_rmse, n_frames):
    d = np.nan_to_num(b_rmse - a_rmse, nan=0.0)
    sm = np.convolve(d, np.ones(n_frames) / n_frames, mode='valid')
    return int(np.argmax(sm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rat', default='R9')
    ap.add_argument('--session', default='2026_10_28_1')
    ap.add_argument('--pass_name', default='pass_c2d6_c5d4')
    ap.add_argument('--start_frame', type=int, default=None,
                    help='omit to pick the window where the fix matters most')
    ap.add_argument('--n_frames', type=int, default=1200)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--smooth', choices=['causal5', 'none'], default='causal5')
    a = ap.parse_args()
    cams = [0, 1, 2]

    print(f'{a.rat}/{a.session}  {a.pass_name}', flush=True)
    sl, base, adapt, b_rmse, a_rmse, known = build(a.rat, a.session, a.pass_name,
                                                   a.smooth)
    n = len(sl)
    start = (a.start_frame if a.start_frame is not None
             else pick_window(b_rmse, a_rmse, a.n_frames))
    end = min(start + a.n_frames, n)
    rng = slice(start, end)
    print(f'  window {start}-{end} ({start/a.fps:.0f}-{end/a.fps:.0f} s): '
          f'baseline {np.nanmedian(b_rmse[rng]):.1f} mm vs '
          f'adaptive {np.nanmedian(a_rmse[rng]):.1f} mm', flush=True)

    calib, cal_date = load_session_calibration(a.rat, a.session)
    proj = {c: {k: project_3d_to_2d_batch(v[rng], calib[c])
                for k, v in [('base', base), ('adapt', adapt), ('sl', sl)]}
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
    out = OUT_ROOT / (f'{a.rat}_{a.session}_adaptdannce_{a.pass_name}_{tag}_'
                      f'cam012_f{start}-{end}.mp4')
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
            # SLEAP reference first and widest, so it reads as a halo under the
            # two DANNCE variants being compared against it.
            _draw(frame, proj[c]['sl'][fi], C_SLEAP, radius=5, thickness=3)
            _draw(frame, proj[c]['base'][fi], C_BASE, radius=3, thickness=2)
            _draw(frame, proj[c]['adapt'][fi], C_ADAPT, radius=2, thickness=1)
            panels.append(frame)
        if len(panels) != len(cams):
            break
        comb = cv2.resize(np.hstack(panels), (ow, oh), interpolation=cv2.INTER_AREA)
        pw = ow // len(cams)
        for i, c in enumerate(cams):
            cv2.putText(comb, f'Camera{c}', (i * pw + 8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        gf = start + fi
        cv2.putText(comb, f'{a.rat}/{a.session}   cal {cal_date}   '
                          f'{"causal-5 median" if a.smooth == "causal5" else "raw"}',
                    (8, oh - 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(comb, 'SLEAP conf^2 (reference)', (8, oh - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_SLEAP, 1, cv2.LINE_AA)
        cv2.putText(comb, f'DANNCE baseline 6-cam          {b_rmse[gf]:5.1f} mm',
                    (8, oh - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BASE, 1, cv2.LINE_AA)
        cv2.putText(comb, f'DANNCE {a.pass_name:<22s}{a_rmse[gf]:5.1f} mm',
                    (8, oh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ADAPT, 1, cv2.LINE_AA)
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
