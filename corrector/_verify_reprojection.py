"""Sanity-check reprojection residuals on a handful of sessions.

For each (rat, session), reproject the *raw* triangulated SLEAP 3D back through
each camera using the per-session calibration, compare to the actual SLEAP 2D
detections, and report:

  - per-camera median |residual| in pixels
  - residual vs. SLEAP confidence (binned)
  - fraction of detections "behind camera" (NaN reprojections)
  - residual when reprojecting the *smoothed* 3D vs raw 3D (expect smoothed
    to have small extra residual on fast motion frames)
  - residual when reprojecting DANNCE 3D (expect LARGER — DANNCE pose differs
    from SLEAP pose, especially extremities)

Also writes a figure at
  corrector/figures/reprojection_residuals.png
showing the per-keypoint residual distribution for one canonical session.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from config import NODES  # noqa: E402

from corrector.data_world_2d import (  # noqa: E402
    load_paired_world_with_2d, reproject_all_cams)


SESSIONS = [
    ("R1", "2025_11_01_1"),
    ("R2", "2025_11_01_1"),
    ("R3", "2025_11_02_1"),
    ("R3", "2026_02_10_1"),   # post-cutover; should be visibly worse
]

CONF_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.5)]


def residual_stats(reproj, detected, conf, name):
    """Returns dict of per-cam, conf-binned, behind-cam stats."""
    n_cam = reproj.shape[1]
    diff = reproj - detected  # (T, 3, 23, 2)
    dist = np.linalg.norm(diff, axis=-1)  # (T, 3, 23) px
    behind = np.isnan(dist)
    valid = ~behind

    out = {"name": name, "per_cam": [], "by_conf": [],
            "frac_behind": float(behind.mean())}
    for ci in range(n_cam):
        d = dist[:, ci][valid[:, ci]]
        out["per_cam"].append({
            "cam": ci,
            "n": int(d.size),
            "median_px": float(np.median(d)),
            "p90_px": float(np.percentile(d, 90)),
            "mean_px": float(d.mean()),
        })
    for lo, hi in CONF_BINS:
        m = valid & (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            out["by_conf"].append({"lo": lo, "hi": hi, "n": 0})
            continue
        d = dist[m]
        out["by_conf"].append({
            "lo": float(lo), "hi": float(hi),
            "n": int(m.sum()),
            "median_px": float(np.median(d)),
            "p90_px": float(np.percentile(d, 90)),
        })
    return out


def fmt_stats(s):
    lines = [f"  {s['name']}"]
    lines.append("    per-camera median (p90) px:")
    for c in s["per_cam"]:
        lines.append(f"      cam{c['cam']}: median={c['median_px']:6.2f}  "
                     f"p90={c['p90_px']:7.2f}  n={c['n']:>7d}")
    lines.append(f"    fraction behind camera (NaN): {s['frac_behind']:.4f}")
    lines.append("    by SLEAP confidence:")
    for c in s["by_conf"]:
        if c["n"] == 0:
            lines.append(f"      conf [{c['lo']:.2f},{c['hi']:.2f}): n=0")
        else:
            lines.append(f"      conf [{c['lo']:.2f},{c['hi']:.2f}): "
                         f"median={c['median_px']:6.2f}  p90={c['p90_px']:7.2f}  "
                         f"n={c['n']:>7d}")
    return "\n".join(lines)


def main():
    plot_data = None
    for rat, sess in SESSIONS:
        print(f"\n=== {rat}/{sess} ===")
        try:
            sl_smooth, dn, xy, conf, sl_raw, calib, cal_date = \
                load_paired_world_with_2d(rat, sess)
        except Exception as e:
            print(f"  load err: {e}")
            continue
        print(f"  cal_date={cal_date}  T={len(sl_smooth)}  "
              f"n_cam={len(calib)}  conf_median={np.median(conf):.3f}  "
              f"conf>0.5_frac={(conf > 0.5).mean():.3f}")

        rep_raw = reproject_all_cams(sl_raw, calib)
        rep_smooth = reproject_all_cams(sl_smooth, calib)
        rep_dannce = reproject_all_cams(dn, calib)

        s_raw = residual_stats(rep_raw, xy, conf, "RAW triang 3D -> 2D")
        s_smo = residual_stats(rep_smooth, xy, conf, "SMOOTHED triang 3D -> 2D")
        s_dnc = residual_stats(rep_dannce, xy, conf, "DANNCE 3D -> 2D")
        print(fmt_stats(s_raw))
        print(fmt_stats(s_smo))
        print(fmt_stats(s_dnc))

        if plot_data is None and rat == "R2":
            plot_data = (rat, sess, rep_raw, rep_smooth, rep_dannce, xy, conf)

    if plot_data is None:
        return
    rat, sess, rep_raw, rep_smooth, rep_dannce, xy, conf = plot_data
    diff_raw = np.linalg.norm(rep_raw - xy, axis=-1)         # (T, 3, 23)
    diff_smooth = np.linalg.norm(rep_smooth - xy, axis=-1)
    diff_dnc = np.linalg.norm(rep_dannce - xy, axis=-1)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    n_kp = diff_raw.shape[-1]
    x = np.arange(n_kp)
    for ax, d, label in [(axes[0], diff_raw, "RAW triang"),
                          (axes[1], diff_smooth, "SMOOTHED triang (median-11)"),
                          (axes[2], diff_dnc, "DANNCE 3D")]:
        for ci, color in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c"]):
            per_kp_med = np.nanmedian(d[:, ci], axis=0)
            per_kp_p90 = np.nanpercentile(d[:, ci], 90, axis=0)
            ax.plot(x, per_kp_med, "-o", color=color,
                     label=f"cam{ci} median", ms=4, lw=1.0)
            ax.plot(x, per_kp_p90, "--", color=color,
                     label=f"cam{ci} p90", lw=0.9, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylabel("reproj |Δ| px (log)")
        ax.set_title(f"{rat}/{sess} — {label}")
        ax.grid(alpha=0.3, which="both")
        if ax is axes[0]:
            ax.legend(fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(NODES, rotation=60, ha="right", fontsize=7)
    axes[-1].set_xlabel("keypoint")
    fig.tight_layout()
    out_path = REPO / "corrector/figures/reprojection_residuals.png"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
