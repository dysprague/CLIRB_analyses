"""Per-rat plots comparing groupO GT vs xyz GT overlap across sessions.

Two panels per rat (vertically stacked):
  A) Stacked bars of n_both / n_xyz_only / n_groupO_only per session.
  B) Recall / precision / F1 of groupO-GT vs xyz-GT over time.
"""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")

df = pd.read_csv(REPO / "corrector/results/velacc_groupo_vs_xyz_gt.csv",
                 parse_dates=["date"])

COLORS = {"both": "#2ca02c", "xyz_only": "#d62728",
          "groupo_only": "#1f77b4"}

out_dir = REPO / "corrector/figures"

for rat in ["R1", "R2", "R3"]:
    sub = df[df.rat == rat].sort_values("date")
    if sub.empty:
        continue
    xs = mdates.date2num(sub.date)
    bar_width = 0.6

    fig, (ax_counts, ax_metric) = plt.subplots(
        2, 1, figsize=(max(11, 0.35 * len(sub)), 8), sharex=True
    )

    # A: stacked bars
    both = sub.n_both.values
    xyz_only = sub.n_xyz_only.values
    go_only = sub.n_groupo_only.values
    ax_counts.bar(xs, both, width=bar_width,
                  color=COLORS["both"], label="both (within 300 ms)")
    ax_counts.bar(xs, xyz_only, width=bar_width, bottom=both,
                  color=COLORS["xyz_only"], label="xyz-only")
    ax_counts.bar(xs, go_only, width=bar_width, bottom=both + xyz_only,
                  color=COLORS["groupo_only"], label="groupO-only")
    ax_counts.set_ylabel("# DANNCE GT template matches (300 ms tol)")
    ax_counts.set_title(
        f"{rat} — overlap of DANNCE GT matches: xyz space vs groupO space"
    )
    ax_counts.legend(fontsize=9, loc="upper right")
    ax_counts.grid(axis="y", alpha=0.25)

    # B: metrics over time
    ax_metric.plot(xs, sub.recall_xyz_in_groupo, "-o",
                   color="#d62728", label="recall (xyz GT found by groupO)",
                   lw=1.2, ms=4)
    ax_metric.plot(xs, sub.precision_xyz_in_groupo, "-s",
                   color="#1f77b4",
                   label="precision (groupO GT confirmed by xyz)",
                   lw=1.2, ms=4)
    ax_metric.plot(xs, sub.f1, "-^",
                   color="#2ca02c", label="F1", lw=1.6, ms=5)
    ax_metric.axhline(0.7, color="gray", lw=0.6, ls=":")
    ax_metric.set_ylim(0, 1.05)
    ax_metric.set_ylabel("metric (0-1)")
    ax_metric.set_title(f"{rat} — groupO GT vs xyz GT agreement")
    ax_metric.legend(fontsize=9, loc="lower right")
    ax_metric.grid(alpha=0.25)
    ax_metric.set_xlabel("Session date")

    ax_metric.xaxis_date()
    ax_metric.xaxis.set_major_locator(
        mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_metric.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")

    fig.tight_layout()
    out_path = out_dir / f"velacc_groupo_vs_xyz_gt_{rat}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")
