"""Grant figure 2: one well-matched and one poorly-matched skeleton.

Both rows show the SAME two reconstructions:
  GREEN    SLEAP, the online 3-camera triangulation
  MAGENTA  sDANNCE, 6-camera, mapped into SLEAP's world frame by ONE
           session-level rigid + isotropic-scale Procrustes fit

Columns 1-3 reproject both into the three SLEAP cameras and draw them on the
actual video frame.  Column 4 shows the same frame in the EGOCENTRIC space the
error metric lives in (SpineM-centred, heading-aligned, 5-frame causal median),
so the picture and the number in F1 are the same quantity.

Frames are chosen by percentile of the egocentric per-frame RMSE -- by default
the session median, the 90th and the 99th percentile.  Percentiles rather than
the single best and single worst frame: the extremes are unrepresentative in
both directions, whereas these three say "here is a typical frame, here is the
frame you hit one time in ten, here is the frame you hit one time in a hundred".
Picks are forced at least 200 frames apart so they are independent moments
rather than three views of the same event.

Usage:
    python scripts/grant_sept_examples.py --rat R5 --session 2026_09_22_1
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); os.chdir(REPO)

import session_cache as sc                                      # noqa: E402
from clirb import events as ev                                  # noqa: E402
from clirb.spatial import dual as du                            # noqa: E402
from skeleton import normalize_skeleton_batch                   # noqa: E402
from config import DATA_ROOT, EDGES, NODE_IDX                   # noqa: E402
from corrector.data_world_2d import (load_session_calibration,   # noqa: E402
                                      project_3d_to_2d_batch)
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample  # noqa: E402

OUT = 'results/grant_sept'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)

# The arena floor and backdrop are green/brown, so the two overlays are cyan
# and magenta -- subtractive primaries, maximally distinct from each other and
# from the scene. (The older reproj renderer used green for SLEAP, which is
# unreadable against this arena.)
C_SLEAP_BGR, C_DANNCE_BGR = (255, 230, 0), (255, 0, 255)
C_SLEAP, C_DANNCE = '#00e6ff', '#ff00ff'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
SPINEM = NODE_IDX['SpineM']
CROP = 300                      # half-width of the crop around the animal, px
GAMMA = 0.55                    # the arena is dimly lit; lift it for print


def brighten(img, gamma=GAMMA):
    """Gamma-lift the frame so the animal is legible in print. Applied only to
    the photograph, never to the overlays, which are drawn afterwards."""
    lut = (np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255)
           ).astype(np.uint8)
    return cv2.LUT(img, lut)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'figure.facecolor': 'white',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42})


def draw(img, pts, color, radius=4, thickness=2):
    p = np.nan_to_num(pts, nan=-1e6).astype(np.int32)
    h, w = img.shape[:2]
    ok = (np.isfinite(pts).all(axis=-1) & (p[:, 0] >= 0) & (p[:, 0] < w)
          & (p[:, 1] >= 0) & (p[:, 1] < h))
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(p[a]), tuple(p[b]), color, thickness, cv2.LINE_AA)
    for i in range(len(p)):
        if ok[i]:
            cv2.circle(img, tuple(p[i]), radius, color, -1, cv2.LINE_AA)


def ego_pair(raw):
    """The F1 pipeline, verbatim: egocentric + causal median, both modalities."""
    sl = np.asarray(raw['sleap_keys_3D'], np.float32)
    dn = np.asarray(raw['dannce_keys_3D'])
    dn = (dn.squeeze(axis=1).transpose(0, 2, 1) if dn.ndim == 4
          else np.transpose(dn, [0, 2, 1]))
    idx = np.asarray(raw['dannce_idx_for_sleap_cams'], int).ravel()[1:]
    dn = dn[np.clip(idx, 0, len(dn) - 1)].astype(np.float32)
    n = min(len(sl), len(dn))
    sl, dn = sl[:n], dn[:n]
    f = sl.copy(); f[:, :, 2] = -f[:, :, 2]
    a = ev.causal_median_5_kp(normalize_skeleton_batch(f)[0], warmup='online')
    b = ev.causal_median_5_kp(normalize_skeleton_batch(dn)[0], warmup='online')
    return np.asarray(a, np.float32), np.asarray(b, np.float32), n


def ego_panel(ax, s_ego, d_ego, view=(0, 2), label=('x (mm)', 'z (mm)')):
    i, j = view
    for pts, col in ((d_ego, C_DANNCE), (s_ego, C_SLEAP)):
        for a, b in EDGES:
            ax.plot([pts[a, i], pts[b, i]], [pts[a, j], pts[b, j]],
                    color=col, lw=1.6, zorder=2, solid_capstyle='round')
        ax.scatter(pts[:, i], pts[:, j], s=16, color=col, zorder=3,
                   edgecolor='white', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel(label[0], fontsize=7.6); ax.set_ylabel(label[1], fontsize=7.6)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MUTED); ax.spines['bottom'].set_color(MUTED)
    ax.grid(color='#eeede9', lw=0.6); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rat', default='R5')
    ap.add_argument('--session', default='2026_09_22_1')
    ap.add_argument('--pcts', type=float, nargs='+', default=[50.0, 90.0, 99.0],
                    help='one row per percentile of the per-frame RMSE')
    a = ap.parse_args()
    cams = [0, 1, 2]

    raw = sc.get_raw_session(a.rat, a.session, want_dual=True)
    s_ego, d_ego, n = ego_pair(raw)
    err = np.linalg.norm(s_ego - d_ego, axis=-1)
    frame_rmse = np.sqrt((np.delete(err, SPINEM, axis=1) ** 2).mean(axis=1))

    # world space for the video overlay
    sl_w = np.asarray(raw['triang_3d'], np.float32)[:n]
    dn_w0 = np.asarray(du.dannce_aligned(raw)[:n], np.float32)
    samp, _ = balanced_sample(np.median(sl_w, axis=1)[:, :2],
                              rng=np.random.default_rng(0))
    fit = robust_proc(dn_w0[samp], sl_w[samp])
    dn_w = fit['apply'](dn_w0).astype(np.float32)
    print(f'{a.rat}/{a.session}: n={n}, Procrustes scale={fit["s"]:.4f} '
          f'residual={fit["residual"]:.2f} mm', flush=True)

    # pick frames near each percentile, requiring the animal to be fully in view
    calib, cal_date = load_session_calibration(a.rat, a.session)
    picks = []
    for pct in a.pcts:
        target = np.percentile(frame_rmse, pct)
        cand = np.argsort(np.abs(frame_rmse - target))[:400]
        chosen = None
        for f in cand:
            if f < 10 or f >= n - 10 or any(abs(int(f) - q) < 200 for q, _ in picks):
                continue
            p2 = {c: project_3d_to_2d_batch(sl_w[f:f + 1], calib[c])[0] for c in cams}
            if all(np.isfinite(v).all() and (v[:, 0] > 60).all() and (v[:, 1] > 60).all()
                   for v in p2.values()):
                chosen = int(f); break
        picks.append((chosen if chosen is not None else int(cand[0]), pct))
        print(f'  p{pct:g}: frame {picks[-1][0]} '
              f'(ego RMSE {frame_rmse[picks[-1][0]]:.1f} mm, '
              f'target {target:.1f} mm)', flush=True)

    caps = {}
    for c in cams:
        vf = Path(DATA_ROOT) / a.rat / a.session / 'sleap' / f'Camera{c}' / '0.mp4'
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            raise FileNotFoundError(vf)
        caps[c] = cap

    nrow = len(picks)
    fig = plt.figure(figsize=(13.6, 3.6 * nrow))
    gs = fig.add_gridspec(nrow, 4, width_ratios=[1, 1, 1, 0.95],
                          hspace=0.16, wspace=0.17)
    ROWLAB = {50.0: 'typical frame\n(session median)',
              90.0: 'moderately mismatched\n(90th percentile)',
              99.0: 'poorly matched\n(99th percentile)'}

    for row, (f, pct) in enumerate(picks):
        for col, c in enumerate(cams):
            caps[c].set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = caps[c].read()
            if not ok:
                continue
            frame = brighten(frame)
            ps = project_3d_to_2d_batch(sl_w[f:f + 1], calib[c])[0]
            pd_ = project_3d_to_2d_batch(dn_w[f:f + 1], calib[c])[0]
            draw(frame, pd_, C_DANNCE_BGR, radius=5, thickness=3)
            draw(frame, ps, C_SLEAP_BGR, radius=3, thickness=2)
            ctr = np.nanmedian(np.vstack([ps, pd_]), axis=0)
            h, w = frame.shape[:2]
            x0 = int(np.clip(ctr[0] - CROP, 0, max(0, w - 2 * CROP)))
            y0 = int(np.clip(ctr[1] - CROP, 0, max(0, h - 2 * CROP)))
            crop = frame[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP]
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(MUTED); sp.set_linewidth(0.8)
            if row == 0:
                ax.set_title(f'Camera{c}', loc='left', fontsize=9.5, color=INK2)
            if col == 0:
                ax.set_ylabel(ROWLAB.get(pct, f'{pct:g}th percentile'),
                              fontsize=9.5, color=INK)

        ax = fig.add_subplot(gs[row, 3])
        ax.set_anchor('N')          # equal aspect shrinks it; keep it top-aligned
        ego_panel(ax, s_ego[f], d_ego[f])
        ax.set_title(f'per-frame RMSE {frame_rmse[f]:.1f} mm', loc='left',
                     fontsize=9.5, color=INK, pad=6)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, lo + max(hi - lo, 260))
        if row == 0:
            ax.annotate('egocentric frame — where the metric lives',
                        xy=(0, 1.22), xycoords='axes fraction', fontsize=9.5,
                        color=INK2, va='bottom')

    for c in caps:
        caps[c].release()

    fig.legend(handles=[
        Line2D([], [], color=C_SLEAP, lw=3.0, label='SLEAP — 3 camera, online'),
        Line2D([], [], color=C_DANNCE, lw=3.0, label='sDANNCE — 6 camera, offline')],
        loc='upper right', bbox_to_anchor=(0.995, 1.0 + 0.055 * 2 / nrow),
        fontsize=9.5,
        ncol=1, handlelength=1.7, labelspacing=0.35)
    fig.suptitle('Example reconstructions — SLEAP (cyan) against sDANNCE '
                 '(magenta), same frame, all three cameras',
                 x=0.012, ha='left', fontsize=13, y=1.0 + 0.015 * 2 / nrow)
    fig.text(0.012, 1.0 - 0.035 * 2 / nrow,
             f'{a.rat} {a.session}, calibration {cal_date}. DANNCE is mapped into '
             "SLEAP's world frame by one session-level rigid + isotropic-scale fit, "
             'so everything still visible between the two skeletons is genuine '
             'tracking disagreement.\nThe right-hand column is the egocentric space '
             'the error metric lives in; its number is the quantity summarised in '
             'the aggregate figure.',
             fontsize=8.2, color=INK2, ha='left', va='top')
    stem = f'{FIG}/F2_examples_{a.rat}_{a.session}'
    fig.savefig(f'{stem}.png', bbox_inches='tight')
    fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {stem}.png/.pdf')


if __name__ == '__main__':
    main()
