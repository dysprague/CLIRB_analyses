"""
Per-edge bone-length sanity check.

Reads an evaluate_world.py JSON and reports which edges have CV that diverges
from DANNCE in the corrected output. A bone is flagged if:
    |cv_corrected - cv_dannce| > tol  AND |cv_corrected - cv_dannce| > |cv_aligned - cv_dannce|
i.e. the corrector made the bone less rigid than DANNCE without making it more
DANNCE-like.

Also prints any bone that the corrector clearly improves (closer to DANNCE
than the Procrustes-only baseline).

Usage:
    python -m corrector.bone_length_check --eval corrector/results/R2R3_world_mlp_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
from config import EDGES, NODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--tol", type=float, default=0.005,
                    help="absolute CV gap above which we flag")
    args = ap.parse_args()

    data = json.load(open(args.eval))
    rows = [r for r in data["rows"] if "error" not in r]
    by_rat = {}
    for r in rows:
        by_rat.setdefault(r["rat"], []).append(r)

    edge_labels = [f"{NODES[i]}-{NODES[j]}" for (i, j) in EDGES]
    n_edges = len(EDGES)

    for rat in sorted(by_rat):
        sub = by_rat[rat]
        d = np.mean([r["bone_cv_dannce"] for r in sub], axis=0)
        a = np.mean([r["bone_cv_sleap_aligned"] for r in sub], axis=0)
        c = np.mean([r["bone_cv_sleap_corrected"] for r in sub], axis=0)

        diffs_corr = c - d
        diffs_align = a - d

        flagged = []
        improved = []
        for k in range(n_edges):
            gap_c = abs(diffs_corr[k])
            gap_a = abs(diffs_align[k])
            if gap_c > args.tol and gap_c > gap_a:
                flagged.append(k)
            elif gap_a > args.tol and gap_c < 0.5 * gap_a:
                improved.append(k)

        print(f"\n{rat} (n={len(sub)} test sessions, tol={args.tol})")
        print(f"  mean CV  DANNCE={d.mean():.4f}  aligned={a.mean():.4f}  "
              f"corrected={c.mean():.4f}")
        if not flagged:
            print("  no edges flagged — corrector preserves rigidity")
        else:
            print(f"  edges where corrector hurts (corrected farther from DANNCE):")
            order = np.argsort(-np.abs(diffs_corr[flagged]))
            for k in [flagged[i] for i in order]:
                print(f"    {edge_labels[k]:<25s}  d={d[k]:.4f}  "
                      f"a={a[k]:.4f}  c={c[k]:.4f}  Δa={diffs_align[k]:+.4f}  "
                      f"Δc={diffs_corr[k]:+.4f}")
        if improved:
            order = np.argsort(np.abs(diffs_corr[improved]))
            top = [improved[i] for i in order[:5]]
            print(f"  edges where corrector clearly helps (top 5 by closeness "
                  f"to DANNCE):")
            for k in top:
                print(f"    {edge_labels[k]:<25s}  d={d[k]:.4f}  "
                      f"a={a[k]:.4f}  c={c[k]:.4f}  Δa={diffs_align[k]:+.4f}  "
                      f"Δc={diffs_corr[k]:+.4f}")


if __name__ == "__main__":
    main()
