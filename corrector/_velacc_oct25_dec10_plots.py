"""Per-rat plots for sessions from 2025-10-25 through 2025-12-10, using
template_2 for each rat. Three bars per session in the top panel:

  - raw_xyz                  (stored SLEAP template_2)
  - corrected_xyz (stored)   (velacc-corrected SLEAP, stored template_2)
  - corrected_xyz (DANNCE)   (velacc-corrected SLEAP, DANNCE-defined template)

GT for all three variants: DANNCE keypoints matched against the
DANNCE-defined template.

Filters:
  - Date window: 2025-10-25 .. 2025-12-10
  - Drop sessions with fewer than 10 total template matches
    (both + SLEAP-only + DANNCE-only on corrected_xyz_stored @ 300 ms).
  - Same-day sessions are spread side-by-side via small day offsets.
"""
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")

df = pd.read_csv(REPO / "corrector/results/velacc_template2_three_variants.csv",
                 parse_dates=["date"])

LO = date(2025, 10, 25)
HI = date(2025, 12, 10)
MIN_TOTAL_MATCHES = 10


def counts_for(row, prefix):
    n_gt = int(row["n_gt_xyz"])
    n_sleap = int(row[f"{prefix}_n_sleap"])
    recall = float(row[f"{prefix}_recall_300"])
    n_both = int(round(recall * n_gt))
    n_sleap_only = max(n_sleap - n_both, 0)
    n_dannce_only = max(n_gt - n_both, 0)
    return n_both, n_sleap_only, n_dannce_only


per_rat = {"R1": [], "R2": [], "R3": []}
n_dropped = 0
for _, row in df.iterrows():
    d = row["date"].date()
    if not (LO <= d <= HI):
        continue
    c_raw = counts_for(row, "raw_xyz")
    c_corr = counts_for(row, "corrected_xyz_stored")
    c_dann = counts_for(row, "corrected_xyz_dannce")
    total = c_corr[0] + c_corr[1] + c_corr[2]
    if total < MIN_TOTAL_MATCHES:
        n_dropped += 1
        continue
    per_rat[row["rat"]].append((d, row, c_raw, c_corr, c_dann))

RAW_COLORS = {"both": "#c7e9c0", "sleap_only": "#c6dbef", "dannce_only": "#fee5d9"}
CORR_COLORS = {"both": "#74c476", "sleap_only": "#6baed6", "dannce_only": "#fc9272"}
DANN_COLORS = {"both": "#005a32", "sleap_only": "#08306b", "dannce_only": "#a50f15"}

LINE_COLORS = {
    "raw_xyz": "#7f7f7f",
    "corrected_xyz (stored tmpl)": "#2ca02c",
    "corrected_xyz (DANNCE tmpl)": "#1f77b4",
}

out_dir = REPO / "corrector" / "figures"
out_dir.mkdir(exist_ok=True)

print(f"Dropped {n_dropped} sessions with <{MIN_TOTAL_MATCHES} matches.")
for rat in per_rat:
    print(f"  {rat}: {len(per_rat[rat])} sessions kept")

SESSION_JITTER = 0.30
TRIO_OFFSET = 0.12
BAR_WIDTH = 0.11

for rat, items in per_rat.items():
    if not items:
        continue
    items.sort(key=lambda t: (t[0], t[1]["session"]))

    by_date = defaultdict(list)
    for d, r, c_raw, c_corr, c_dann in items:
        by_date[d].append((r, c_raw, c_corr, c_dann))

    centers = []
    rows_ordered = []
    raw_counts, corr_counts, dann_counts = [], [], []
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
        for (r, c_raw, c_corr, c_dann), off in zip(group, offsets):
            centers.append(mdates.date2num(d) + off)
            rows_ordered.append(r)
            raw_counts.append(c_raw)
            corr_counts.append(c_corr)
            dann_counts.append(c_dann)

    centers = np.array(centers)
    xs_raw = centers - TRIO_OFFSET
    xs_corr = centers
    xs_dann = centers + TRIO_OFFSET

    def to_arrays(triples):
        b = np.array([t[0] for t in triples])
        s = np.array([t[1] for t in triples])
        d = np.array([t[2] for t in triples])
        return b, s, d

    raw_b, raw_s, raw_d = to_arrays(raw_counts)
    corr_b, corr_s, corr_d = to_arrays(corr_counts)
    dann_b, dann_s, dann_d = to_arrays(dann_counts)

    fig, (ax_counts, ax_f1, ax_align) = plt.subplots(
        3, 1, figsize=(max(11, 0.55 * len(items)), 11), sharex=True
    )

    for x, b, s, d, palette in [
        (xs_raw, raw_b, raw_s, raw_d, RAW_COLORS),
        (xs_corr, corr_b, corr_s, corr_d, CORR_COLORS),
        (xs_dann, dann_b, dann_s, dann_d, DANN_COLORS),
    ]:
        ax_counts.bar(x, b, width=BAR_WIDTH, color=palette["both"])
        ax_counts.bar(x, s, width=BAR_WIDTH, bottom=b,
                      color=palette["sleap_only"])
        ax_counts.bar(x, d, width=BAR_WIDTH, bottom=b + s,
                      color=palette["dannce_only"])

    legend_handles = [
        mpatches.Patch(color=RAW_COLORS["both"], label="raw — both"),
        mpatches.Patch(color=RAW_COLORS["sleap_only"], label="raw — SLEAP-only"),
        mpatches.Patch(color=RAW_COLORS["dannce_only"], label="raw — DANNCE-only"),
        mpatches.Patch(color=CORR_COLORS["both"], label="corr (stored) — both"),
        mpatches.Patch(color=CORR_COLORS["sleap_only"],
                       label="corr (stored) — SLEAP-only"),
        mpatches.Patch(color=CORR_COLORS["dannce_only"],
                       label="corr (stored) — DANNCE-only"),
        mpatches.Patch(color=DANN_COLORS["both"], label="corr (DANNCE) — both"),
        mpatches.Patch(color=DANN_COLORS["sleap_only"],
                       label="corr (DANNCE) — SLEAP-only"),
        mpatches.Patch(color=DANN_COLORS["dannce_only"],
                       label="corr (DANNCE) — DANNCE-only"),
    ]
    ax_counts.set_ylabel("# template matches (300 ms)")
    ax_counts.set_title(
        f"{rat} — TEMPLATE_2 — offline match overlap "
        f"(GT: DANNCE vs DANNCE-defined template_2)  "
        f"raw | corr-stored | corr-DANNCE  "
        f"[{LO} .. {HI}, total>={MIN_TOTAL_MATCHES}]"
    )
    ax_counts.legend(handles=legend_handles, fontsize=7, loc="upper right",
                     ncol=3)
    ax_counts.grid(axis="y", alpha=0.25)

    raw_f1 = [float(r["raw_xyz_f1_300"]) for r in rows_ordered]
    corr_f1 = [float(r["corrected_xyz_stored_f1_300"]) for r in rows_ordered]
    dann_f1 = [float(r["corrected_xyz_dannce_f1_300"]) for r in rows_ordered]
    ax_f1.plot(centers, raw_f1, "-o",
               color=LINE_COLORS["raw_xyz"], label="raw_xyz", lw=1.2, ms=4)
    ax_f1.plot(centers, corr_f1, "-o",
               color=LINE_COLORS["corrected_xyz (stored tmpl)"],
               label="corrected_xyz (stored tmpl)", lw=1.2, ms=4)
    ax_f1.plot(centers, dann_f1, "-o",
               color=LINE_COLORS["corrected_xyz (DANNCE tmpl)"],
               label="corrected_xyz (DANNCE tmpl)", lw=1.2, ms=4)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1@300 ms")
    ax_f1.set_title(
        f"{rat} — TEMPLATE_2 — F1: raw / corrected (stored) / corrected (DANNCE)"
    )
    ax_f1.legend(fontsize=9, loc="lower right")
    ax_f1.grid(alpha=0.25)

    kp_align = [float(r["kp_mse_align"]) for r in rows_ordered]
    ax_align.plot(centers, kp_align, "-o", color="#d62728",
                  label="kp_mse_align (mm²)", lw=1.2, ms=4)
    ax_align.set_ylabel("kp_mse_align (mm²)", color="#d62728")
    ax_align.tick_params(axis="y", labelcolor="#d62728")
    ax_align.grid(alpha=0.25)
    ax_align.set_title(
        f"{rat} — TEMPLATE_2 — SLEAP–DANNCE keypoint MSE after alignment"
    )
    ax_align.set_xlabel("Session date")

    ax_align.xaxis_date()
    ax_align.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_align.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")

    fig.tight_layout()
    out_path = out_dir / f"velacc_oct25_dec10_template2_{rat}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")
