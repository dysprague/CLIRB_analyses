"""Per-rat plots restricted to sessions BEFORE 2026_02_06.

Three figures per rat (one combined figure per rat, 3 panels):
  A) Stacked bar: # both, # sleap-only, # dannce-only offline matches per session
     (uses corrected_xyz at 300 ms tolerance for the R1R2R3_velacc model)
  B) F1 lines: raw_xyz, procrustes_xyz, corrected_xyz, plus corrected_groupO
  C) SLEAP/DANNCE alignment quality: kp_mse_align (pre-correction) and
     procrustes_residual over time (twin y-axes).
"""
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")

with open(REPO / "corrector/results/R1R2R3_velacc_all.json") as f:
    eval_data = json.load(f)
rows = eval_data["per_session"]

CUTOFF = date(2026, 2, 6)  # exclusive — only sessions strictly before this


def session_date(sess):
    parts = sess.split("_")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


# Filter and sort by date
per_rat = {"R1": [], "R2": [], "R3": []}
for r in rows:
    d = session_date(r["session"])
    if d >= CUTOFF:
        continue
    per_rat[r["rat"]].append((d, r))
for rat in per_rat:
    per_rat[rat].sort(key=lambda t: t[0])

print("Pre-2026_02_06 sessions per rat:")
for rat, items in per_rat.items():
    print(f"  {rat}: {[str(d) + ' / ' + r['session'] for d, r in items]}")


def counts_from_row(r, label="corrected_xyz", tol=300):
    n_gt = r.get("n_gt_xyz")
    n_sleap = r.get(f"{label}_n_sleap")
    recall = r.get(f"{label}_recall_{tol}")
    if any(v is None for v in (n_gt, n_sleap, recall)):
        return None
    n_both = int(round(recall * n_gt))
    n_sleap_only = max(int(n_sleap) - n_both, 0)
    n_dannce_only = max(int(n_gt) - n_both, 0)
    return n_both, n_sleap_only, n_dannce_only


COLORS = {
    "both": "#2ca02c",
    "sleap_only": "#1f77b4",
    "dannce_only": "#d62728",
}

LINE_COLORS = {
    "raw_xyz": "#7f7f7f",
    "procrustes_xyz": "#9467bd",
    "corrected_xyz": "#2ca02c",
    "corrected_groupO": "#1f77b4",
}

out_dir = REPO / "corrector" / "figures"
out_dir.mkdir(exist_ok=True)

for rat, items in per_rat.items():
    if not items:
        continue
    dates = [d for d, _ in items]
    sess_names = [r["session"] for _, r in items]
    xs = np.arange(len(items))

    fig, (ax_counts, ax_f1, ax_align) = plt.subplots(
        3, 1, figsize=(max(7, 1.5 * len(items)), 11), sharex=True
    )

    # ---- Panel A: stacked bars of match counts -----------------------------
    both_arr, sleap_only_arr, dannce_only_arr = [], [], []
    for _, r in items:
        c = counts_from_row(r, "corrected_xyz", 300)
        if c is None:
            both_arr.append(0); sleap_only_arr.append(0); dannce_only_arr.append(0)
        else:
            b, s, d = c
            both_arr.append(b); sleap_only_arr.append(s); dannce_only_arr.append(d)
    both_arr = np.array(both_arr)
    sleap_only_arr = np.array(sleap_only_arr)
    dannce_only_arr = np.array(dannce_only_arr)
    ax_counts.bar(xs, both_arr, color=COLORS["both"], label="both")
    ax_counts.bar(xs, sleap_only_arr, bottom=both_arr,
                  color=COLORS["sleap_only"], label="SLEAP-only")
    ax_counts.bar(xs, dannce_only_arr,
                  bottom=both_arr + sleap_only_arr,
                  color=COLORS["dannce_only"], label="DANNCE-only")
    for x, b, s, d in zip(xs, both_arr, sleap_only_arr, dannce_only_arr):
        total = b + s + d
        ax_counts.text(x, total + 1, str(total), ha="center", va="bottom",
                       fontsize=8)
    ax_counts.set_ylabel("# template matches (300 ms)")
    ax_counts.set_title(f"{rat} — offline match overlap, R1R2R3_velacc corrected")
    ax_counts.legend(fontsize=9, loc="upper right")
    ax_counts.grid(axis="y", alpha=0.25)

    # ---- Panel B: F1 across the pipeline -----------------------------------
    for label, color in LINE_COLORS.items():
        ys = [r.get(f"{label}_f1_300") for _, r in items]
        ax_f1.plot(xs, ys, "-o", color=color, label=label, lw=1.5, ms=6)
    ax_f1.axhline(0.7, color="gray", lw=0.6, ls=":")
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1@300 ms")
    ax_f1.set_title(f"{rat} — F1 across pipeline stages")
    ax_f1.legend(fontsize=9, loc="lower right")
    ax_f1.grid(alpha=0.25)

    # ---- Panel C: SLEAP/DANNCE alignment quality over time -----------------
    kp_align = [r.get("kp_mse_align") for _, r in items]
    proc_resid = [r.get("procrustes_residual") for _, r in items]
    ax_align.plot(xs, kp_align, "-o", color="#d62728",
                  label="kp_mse_align (mm²)", lw=1.5, ms=6)
    ax_align.set_ylabel("kp_mse_align (mm²)", color="#d62728")
    ax_align.tick_params(axis="y", labelcolor="#d62728")
    ax_align.grid(alpha=0.25)

    ax2 = ax_align.twinx()
    ax2.plot(xs, proc_resid, "-s", color="#1f77b4",
             label="Procrustes residual (mm)", lw=1.5, ms=6)
    ax2.set_ylabel("Procrustes residual (mm)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax_align.set_title(f"{rat} — SLEAP-DANNCE alignment quality")

    # x-axis: session labels
    ax_align.set_xticks(xs)
    ax_align.set_xticklabels([s for s in sess_names], rotation=40,
                              ha="right", fontsize=8)
    ax_align.set_xlabel("Session (chronological)")

    fig.tight_layout()
    out_path = out_dir / f"velacc_pre0206_{rat}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")
