"""
QC-style summary report for a world-space corrector eval.

Reads the JSON produced by evaluate_world.py and writes:
  - per-rat keypoint MSE table (aligned vs corrected, % improvement)
  - per-keypoint bar plot (worst keypoints first, raw vs corrected)
  - per-edge bone-length CV plot (DANNCE, raw-aligned-SLEAP, corrected-SLEAP)
  - PC MSE bar plot per rat

Usage:
    python -m corrector.summary_report --eval corrector/results/R2R3_world_mlp_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
from config import EDGES, NODES

FIG_DIR = _THIS.parent / "figures" / "summary"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_per_keypoint(rows):
    """Mean per-keypoint MSE (aligned, corrected) across rows."""
    if not rows:
        return None, None
    a = np.mean([r["per_keypoint_mse_aligned"] for r in rows], axis=0)
    c = np.mean([r["per_keypoint_mse_corrected"] for r in rows], axis=0)
    return a, c


def aggregate_pc(rows):
    if not rows: return None, None
    a = np.mean([r["pc_mse_aligned"] for r in rows], axis=0)
    c = np.mean([r["pc_mse_corrected"] for r in rows], axis=0)
    return a, c


def aggregate_bone_cv(rows):
    if not rows: return None, None, None
    a = np.mean([r["bone_cv_sleap_aligned"] for r in rows], axis=0)
    c = np.mean([r["bone_cv_sleap_corrected"] for r in rows], axis=0)
    d = np.mean([r["bone_cv_dannce"] for r in rows], axis=0)
    return d, a, c


def headline_table(by_rat):
    print(f"\n{'rat':<5} {'n':>3} {'kp aligned':>12} {'kp corr':>10} '{'%Δ':>5}' "
          f"{'PC1 a→c':>14} {'PC2 a→c':>14}")
    print("-" * 80)
    for rat, agg in sorted(by_rat.items()):
        if agg is None: continue
        ma = agg["keypoint_mse_aligned"]; mc = agg["keypoint_mse_corrected"]
        pa = agg["pc_mse_aligned"]; pc = agg["pc_mse_corrected"]
        pct = -100 * (ma - mc) / ma
        n = agg["n"]
        print(f"{rat:<5} {n:>3} {ma:>12.1f} {mc:>10.1f} {pct:>5.0f}% "
              f"{pa[0]:>6.0f}→{pc[0]:<6.0f} "
              f"{pa[1]:>6.0f}→{pc[1]:<6.0f}")


def per_keypoint_plot(by_rat_rows, out_path, title=""):
    rats = list(by_rat_rows.keys())
    fig, axes = plt.subplots(len(rats), 1, figsize=(14, 3 * len(rats)),
                              sharex=True)
    if len(rats) == 1: axes = [axes]
    for ax, rat in zip(axes, rats):
        a, c = aggregate_per_keypoint(by_rat_rows[rat])
        if a is None: continue
        x = np.arange(len(NODES))
        w = 0.4
        ax.bar(x - w/2, a, w, label="Procrustes only", color="#1f77b4", alpha=0.85)
        ax.bar(x + w/2, c, w, label="+ corrector",     color="#2ca02c", alpha=0.85)
        ax.set_ylabel(f"{rat}\nper-kp MSE")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(np.arange(len(NODES)))
    axes[-1].set_xticklabels(NODES, rotation=45, ha="right", fontsize=8)
    fig.suptitle(title or "Per-keypoint MSE (mean over test sessions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def bone_length_plot(by_rat_rows, out_path):
    rats = list(by_rat_rows.keys())
    edge_labels = [f"{NODES[i]}-{NODES[j]}" for (i, j) in EDGES]
    fig, axes = plt.subplots(len(rats), 1, figsize=(14, 3 * len(rats)),
                              sharex=True)
    if len(rats) == 1: axes = [axes]
    for ax, rat in zip(axes, rats):
        d, a, c = aggregate_bone_cv(by_rat_rows[rat])
        if d is None: continue
        x = np.arange(len(EDGES))
        w = 0.27
        ax.bar(x - w, d, w, label="DANNCE",           color="#d62728", alpha=0.85)
        ax.bar(x,     a, w, label="SLEAP (aligned)",  color="#1f77b4", alpha=0.85)
        ax.bar(x + w, c, w, label="SLEAP (+corrector)", color="#2ca02c", alpha=0.85)
        ax.set_ylabel(f"{rat}\nbone-length CV")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(np.arange(len(EDGES)))
    axes[-1].set_xticklabels(edge_labels, rotation=60, ha="right", fontsize=7)
    fig.suptitle("Per-edge bone-length coefficient of variation "
                 "(lower = more rigid)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def pc_plot(by_rat, out_path):
    rats = list(by_rat.keys())
    n_pcs = len(by_rat[rats[0]]["pc_mse_aligned"])
    fig, axes = plt.subplots(1, len(rats), figsize=(4 * len(rats), 4), sharey=True)
    if len(rats) == 1: axes = [axes]
    for ax, rat in zip(axes, rats):
        a = np.array(by_rat[rat]["pc_mse_aligned"])
        c = np.array(by_rat[rat]["pc_mse_corrected"])
        x = np.arange(n_pcs)
        w = 0.4
        ax.bar(x - w/2, a, w, label="Procrustes only", color="#1f77b4", alpha=0.85)
        ax.bar(x + w/2, c, w, label="+ corrector",     color="#2ca02c", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([f"PC{j+1}" for j in x])
        ax.set_title(f"{rat}  (n={by_rat[rat]['n']})")
        ax.set_ylabel("PC MSE")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
    fig.suptitle("PC-space MSE (lower = better) per rat", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    args = ap.parse_args()

    data = json.load(open(args.eval))
    rows = [r for r in data["rows"] if "error" not in r]
    by_rat_rows = {}
    for r in rows:
        by_rat_rows.setdefault(r["rat"], []).append(r)

    print(f"Loaded {len(rows)} valid sessions; rats: {list(by_rat_rows.keys())}")
    headline_table(data["by_rat"])

    stem = Path(args.eval).stem
    per_keypoint_plot(by_rat_rows, FIG_DIR / f"{stem}_per_keypoint.png",
                      title=f"{stem}: per-keypoint MSE")
    bone_length_plot(by_rat_rows, FIG_DIR / f"{stem}_bone_cv.png")
    pc_plot(data["by_rat"], FIG_DIR / f"{stem}_pc_mse.png")

    print(f"\nFigures written to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob(f"{stem}_*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
