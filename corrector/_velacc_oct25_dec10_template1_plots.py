"""Per-rat plots for sessions from 2025-10-25 through 2025-12-10, using
TEMPLATE_1 (the SLEAP-defined template currently used by evaluate_all.py).

Two bars per session in the top panel:
  - raw_xyz                (stored SLEAP template_1)
  - corrected_xyz          (velacc-corrected SLEAP, stored SLEAP template_1)

GT for both variants: DANNCE matched against the stored SLEAP template_1.

Filters:
  - 2025-10-25 .. 2025-12-10
  - Drop sessions with <10 total matches (both + SLEAP-only + DANNCE-only on
    corrected_xyz @ 300 ms).
  - Same-day sessions are spread side-by-side via small day offsets.
"""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")

with open(REPO / "corrector/results/R1R2R3_velacc_all.json") as f:
    eval_data = json.load(f)
rows = eval_data["per_session"]

LO = date(2025, 10, 25)
HI = date(2025, 12, 10)
MIN_TOTAL_MATCHES = 10


def session_date(sess):
    parts = sess.split("_")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def counts_from_row(r, label, tol=300):
    n_gt = r.get("n_gt_xyz")
    n_sleap = r.get(f"{label}_n_sleap")
    recall = r.get(f"{label}_recall_{tol}")
    if any(v is None for v in (n_gt, n_sleap, recall)):
        return None
    n_both = int(round(recall * n_gt))
    n_sleap_only = max(int(n_sleap) - n_both, 0)
    n_dannce_only = max(int(n_gt) - n_both, 0)
    return n_both, n_sleap_only, n_dannce_only


per_rat = {"R1": [], "R2": [], "R3": []}
n_dropped = 0
for r in rows:
    if "error" in r:
        continue
    d = session_date(r["session"])
    if not (LO <= d <= HI):
        continue
    c_corr = counts_from_row(r, "corrected_xyz")
    c_raw = counts_from_row(r, "raw_xyz")
    if c_corr is None or c_raw is None:
        continue
    total = c_corr[0] + c_corr[1] + c_corr[2]
    if total < MIN_TOTAL_MATCHES:
        n_dropped += 1
        continue
    per_rat[r["rat"]].append((d, r, c_raw, c_corr))

RAW_COLORS = {"both": "#a6dba0", "sleap_only": "#a6cee3", "dannce_only": "#fdbf6f"}
CORR_COLORS = {"both": "#1b7837", "sleap_only": "#1f78b4", "dannce_only": "#ff7f00"}

LINE_COLORS = {
    "raw_xyz": "#7f7f7f",
    "corrected_xyz": "#2ca02c",
}

out_dir = REPO / "corrector" / "figures"
out_dir.mkdir(exist_ok=True)

print(f"Dropped {n_dropped} sessions with <{MIN_TOTAL_MATCHES} matches.")
for rat in per_rat:
    print(f"  {rat}: {len(per_rat[rat])} sessions kept")

SESSION_JITTER = 0.22
PAIR_OFFSET = 0.12
BAR_WIDTH = 0.18

for rat, items in per_rat.items():
    if not items:
        continue
    items.sort(key=lambda t: (t[0], t[1]["session"]))

    by_date = defaultdict(list)
    for d, r, c_raw, c_corr in items:
        by_date[d].append((r, c_raw, c_corr))

    centers = []
    rows_ordered = []
    raw_counts = []
    corr_counts = []
    for d in sorted(by_date.keys()):
        group = by_date[d]
        n = len(group)
        if n == 1:
            offsets = [0.0]
        elif n == 2:
            offsets = [-SESSION_JITTER, +SESSION_JITTER]
        else:
            half = (n - 1) / 2.0
            offsets = [(i - half) * SESSION_JITTER * 2 / max(n - 1, 1)
                       for i in range(n)]
        for (r, c_raw, c_corr), off in zip(group, offsets):
            centers.append(mdates.date2num(d) + off)
            rows_ordered.append(r)
            raw_counts.append(c_raw)
            corr_counts.append(c_corr)

    centers = np.array(centers)
    xs_raw = centers - PAIR_OFFSET
    xs_corr = centers + PAIR_OFFSET

    raw_both = np.array([c[0] for c in raw_counts])
    raw_slop = np.array([c[1] for c in raw_counts])
    raw_dnop = np.array([c[2] for c in raw_counts])
    corr_both = np.array([c[0] for c in corr_counts])
    corr_slop = np.array([c[1] for c in corr_counts])
    corr_dnop = np.array([c[2] for c in corr_counts])

    fig, (ax_counts, ax_f1, ax_align) = plt.subplots(
        3, 1, figsize=(max(11, 0.55 * len(items)), 11), sharex=True
    )

    ax_counts.bar(xs_raw, raw_both, width=BAR_WIDTH, color=RAW_COLORS["both"])
    ax_counts.bar(xs_raw, raw_slop, width=BAR_WIDTH, bottom=raw_both,
                  color=RAW_COLORS["sleap_only"])
    ax_counts.bar(xs_raw, raw_dnop, width=BAR_WIDTH, bottom=raw_both + raw_slop,
                  color=RAW_COLORS["dannce_only"])

    ax_counts.bar(xs_corr, corr_both, width=BAR_WIDTH, color=CORR_COLORS["both"])
    ax_counts.bar(xs_corr, corr_slop, width=BAR_WIDTH, bottom=corr_both,
                  color=CORR_COLORS["sleap_only"])
    ax_counts.bar(xs_corr, corr_dnop, width=BAR_WIDTH,
                  bottom=corr_both + corr_slop,
                  color=CORR_COLORS["dannce_only"])

    legend_handles = [
        mpatches.Patch(color=RAW_COLORS["both"], label="raw — both"),
        mpatches.Patch(color=RAW_COLORS["sleap_only"], label="raw — SLEAP-only"),
        mpatches.Patch(color=RAW_COLORS["dannce_only"], label="raw — DANNCE-only"),
        mpatches.Patch(color=CORR_COLORS["both"], label="corrected — both"),
        mpatches.Patch(color=CORR_COLORS["sleap_only"],
                       label="corrected — SLEAP-only"),
        mpatches.Patch(color=CORR_COLORS["dannce_only"],
                       label="corrected — DANNCE-only"),
    ]
    ax_counts.set_ylabel("# template matches (300 ms)")
    ax_counts.set_title(
        f"{rat} — TEMPLATE_1 — offline match overlap, raw_xyz (L) vs corrected_xyz (R)  "
        f"[{LO} .. {HI}, total>={MIN_TOTAL_MATCHES}]"
    )
    ax_counts.legend(handles=legend_handles, fontsize=8, loc="upper right",
                     ncol=2)
    ax_counts.grid(axis="y", alpha=0.25)

    for label, color in LINE_COLORS.items():
        ys = [r.get(f"{label}_f1_300") for r in rows_ordered]
        ax_f1.plot(centers, ys, "-o", color=color, label=label, lw=1.2, ms=4)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1@300 ms")
    ax_f1.set_title(f"{rat} — TEMPLATE_1 — F1: raw_xyz vs corrected_xyz")
    ax_f1.legend(fontsize=9, loc="lower right")
    ax_f1.grid(alpha=0.25)

    kp_align = [r.get("kp_mse_align") for r in rows_ordered]
    ax_align.plot(centers, kp_align, "-o", color="#d62728",
                  label="kp_mse_align (mm²)", lw=1.2, ms=4)
    ax_align.set_ylabel("kp_mse_align (mm²)", color="#d62728")
    ax_align.tick_params(axis="y", labelcolor="#d62728")
    ax_align.grid(alpha=0.25)
    ax_align.set_title(
        f"{rat} — TEMPLATE_1 — SLEAP–DANNCE keypoint MSE after alignment"
    )
    ax_align.set_xlabel("Session date")

    ax_align.xaxis_date()
    ax_align.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_align.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")

    fig.tight_layout()
    out_path = out_dir / f"velacc_oct25_dec10_template1_{rat}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")
