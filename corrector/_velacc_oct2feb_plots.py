"""Per-rat plots for all sessions from 2025-10-01 through 2026-02-05.

Three panels per rat (vertically stacked):
  A) Stacked bar: # both, # SLEAP-only, # DANNCE-only offline template matches
     (corrected_xyz, 300 ms tol) using the R1R2R3_velacc model.
  B) F1@300 lines per pipeline stage (raw_xyz, procrustes_xyz,
     corrected_xyz, corrected_groupO).
  C) SLEAP–DANNCE alignment quality: kp_mse_align (mm², left) and
     Procrustes residual (mm, right) over time.
"""
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")

with open(REPO / "corrector/results/R1R2R3_velacc_all.json") as f:
    eval_data = json.load(f)
rows = eval_data["per_session"]

LO = date(2025, 10, 1)
HI = date(2026, 2, 5)


def session_date(sess):
    parts = sess.split("_")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


per_rat = {"R1": [], "R2": [], "R3": []}
n_errors = 0
for r in rows:
    if "error" in r:
        n_errors += 1
        continue
    d = session_date(r["session"])
    if not (LO <= d <= HI):
        continue
    per_rat[r["rat"]].append((d, r))
for rat in per_rat:
    per_rat[rat].sort(key=lambda t: t[0])

print(f"Skipped {n_errors} sessions with errors.")
for rat in per_rat:
    print(f"  {rat}: {len(per_rat[rat])} sessions in 2025-10-01..2026-02-05")


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


COLORS = {"both": "#2ca02c", "sleap_only": "#1f77b4", "dannce_only": "#d62728"}
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
    dates_ = [d for d, _ in items]
    sess_names = [r["session"] for _, r in items]
    xs = mdates.date2num(dates_)

    # Width approx 0.6 day; if too narrow set to 0.4
    bar_width = 0.6

    fig, (ax_counts, ax_f1, ax_align) = plt.subplots(
        3, 1, figsize=(max(11, 0.35 * len(items)), 11), sharex=True
    )

    # ---- A: stacked bars ---------------------------------------------------
    both_a, slop_a, dnop_a = [], [], []
    for _, r in items:
        c = counts_from_row(r)
        if c is None:
            both_a.append(0); slop_a.append(0); dnop_a.append(0)
        else:
            b, s, d = c
            both_a.append(b); slop_a.append(s); dnop_a.append(d)
    both_a = np.array(both_a)
    slop_a = np.array(slop_a)
    dnop_a = np.array(dnop_a)

    ax_counts.bar(xs, both_a, width=bar_width,
                  color=COLORS["both"], label="both")
    ax_counts.bar(xs, slop_a, width=bar_width, bottom=both_a,
                  color=COLORS["sleap_only"], label="SLEAP-only")
    ax_counts.bar(xs, dnop_a, width=bar_width, bottom=both_a + slop_a,
                  color=COLORS["dannce_only"], label="DANNCE-only")
    ax_counts.set_ylabel("# template matches (300 ms)")
    ax_counts.set_title(
        f"{rat} — offline match overlap (R1R2R3_velacc, corrected_xyz)"
    )
    ax_counts.legend(fontsize=9, loc="upper right")
    ax_counts.grid(axis="y", alpha=0.25)

    # ---- B: F1 lines -------------------------------------------------------
    for label, color in LINE_COLORS.items():
        ys = [r.get(f"{label}_f1_300") for _, r in items]
        ax_f1.plot(xs, ys, "-o", color=color, label=label, lw=1.2, ms=4)
    ax_f1.axhline(0.7, color="gray", lw=0.6, ls=":")
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1@300 ms")
    ax_f1.set_title(f"{rat} — F1 across pipeline stages")
    ax_f1.legend(fontsize=9, loc="lower right", ncol=4)
    ax_f1.grid(alpha=0.25)

    # ---- C: alignment quality ---------------------------------------------
    kp_align = [r.get("kp_mse_align") for _, r in items]
    proc_resid = [r.get("procrustes_residual") for _, r in items]
    ax_align.plot(xs, kp_align, "-o", color="#d62728",
                  label="kp_mse_align (mm²)", lw=1.2, ms=4)
    ax_align.set_ylabel("kp_mse_align (mm²)", color="#d62728")
    ax_align.tick_params(axis="y", labelcolor="#d62728")
    ax_align.grid(alpha=0.25)

    ax2 = ax_align.twinx()
    ax2.plot(xs, proc_resid, "-s", color="#1f77b4",
             label="Procrustes residual (mm)", lw=1.2, ms=4)
    ax2.set_ylabel("Procrustes residual (mm)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax_align.set_title(f"{rat} — SLEAP–DANNCE alignment quality")
    ax_align.set_xlabel("Session date")

    # Date formatting on x-axis
    ax_align.xaxis_date()
    ax_align.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_align.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")

    fig.tight_layout()
    out_path = out_dir / f"velacc_oct2feb_{rat}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")
