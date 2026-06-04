"""Per-rat plots comparing the STORED template_1 against the REBUILT
template_1 (DANNCE-defined, mean of matched windows). Mirrors the layout of
_velacc_oct25_dec10_template1_plots.py but reads from
corrector/results/velacc_template1_rebuild.csv (which contains BOTH
stored and rebuilt variants per session in a single row).

For each rat, three stacked panels:
  1) Stacked bar counts: rebuilt (left bar) vs stored (right bar) per session
     (for the corrected_xyz variant).
  2) F1@300 ms line: corrected_xyz_rebuilt vs corrected_xyz_stored, with
     raw_xyz_rebuilt / raw_xyz_stored as faded reference.
  3) kp_mse_align (SLEAP-DANNCE post-Procrustes mm²) line — same series as
     the existing template_1 plot.

Filters: 2025-10-25 .. 2025-12-10, drop sessions with <10 total matches on
either variant of corrected_xyz. Same-day sessions side-by-side.

Outputs (NEW filenames; nothing existing is overwritten):
  corrector/figures/velacc_template1_rebuild_{R1,R2,R3}.png
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
CSV_PATH = REPO / "corrector/results/velacc_template1_rebuild.csv"
OUT_DIR = REPO / "corrector/figures"
OUT_DIR.mkdir(exist_ok=True)

LO = date(2025, 10, 25)
HI = date(2025, 12, 10)
MIN_TOTAL_MATCHES = 10


def session_date(sess):
    parts = sess.split("_")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def counts_from_row(row, variant_label):
    """For a corrected_xyz_<variant> column-set, return
    (both, sleap_only, dannce_only) at 300 ms."""
    n_gt = row.get(f"n_gt_xyz_{variant_label}")
    n_sleap = row.get(f"corrected_xyz_{variant_label}_n_sleap")
    recall = row.get(f"corrected_xyz_{variant_label}_recall_300")
    if any(pd.isna(v) or v is None for v in (n_gt, n_sleap, recall)):
        return None
    n_both = int(round(recall * n_gt))
    n_sleap_only = max(int(n_sleap) - n_both, 0)
    n_dannce_only = max(int(n_gt) - n_both, 0)
    return n_both, n_sleap_only, n_dannce_only


df = pd.read_csv(CSV_PATH)
df["date"] = df["session"].apply(session_date)
df = df[(df["date"] >= LO) & (df["date"] <= HI)].copy()

per_rat = defaultdict(list)
n_dropped = 0
for _, r in df.iterrows():
    c_reb = counts_from_row(r, "rebuilt")
    c_sto = counts_from_row(r, "stored")
    if c_reb is None or c_sto is None:
        n_dropped += 1
        continue
    if (sum(c_reb) < MIN_TOTAL_MATCHES) and (sum(c_sto) < MIN_TOTAL_MATCHES):
        n_dropped += 1
        continue
    per_rat[r["rat"]].append((r["date"], r, c_reb, c_sto))

print(f"Dropped {n_dropped} sessions with <{MIN_TOTAL_MATCHES} matches "
      f"on both variants.")
for rat in ["R1", "R2", "R3"]:
    print(f"  {rat}: {len(per_rat[rat])} sessions kept")

REB_COLORS = {"both": "#1b7837", "sleap_only": "#1f78b4", "dannce_only": "#ff7f00"}
STO_COLORS = {"both": "#a6dba0", "sleap_only": "#a6cee3", "dannce_only": "#fdbf6f"}
LINE_COLORS = {
    "corrected_xyz_rebuilt": "#1b7837",
    "corrected_xyz_stored":  "#7f7f7f",
    "raw_xyz_rebuilt":       "#a6dba0",
    "raw_xyz_stored":        "#cccccc",
}

SESSION_JITTER = 0.22
PAIR_OFFSET = 0.12
BAR_WIDTH = 0.18


for rat in ["R1", "R2", "R3"]:
    items = per_rat[rat]
    if not items:
        continue
    items.sort(key=lambda t: (t[0], t[1]["session"]))

    by_date = defaultdict(list)
    for d, r, c_reb, c_sto in items:
        by_date[d].append((r, c_reb, c_sto))

    centers, rows_ord, reb_counts, sto_counts = [], [], [], []
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
        for (r, c_reb, c_sto), off in zip(group, offsets):
            centers.append(mdates.date2num(d) + off)
            rows_ord.append(r)
            reb_counts.append(c_reb)
            sto_counts.append(c_sto)

    centers = np.array(centers)
    xs_reb = centers - PAIR_OFFSET
    xs_sto = centers + PAIR_OFFSET

    reb_both = np.array([c[0] for c in reb_counts])
    reb_slop = np.array([c[1] for c in reb_counts])
    reb_dnop = np.array([c[2] for c in reb_counts])
    sto_both = np.array([c[0] for c in sto_counts])
    sto_slop = np.array([c[1] for c in sto_counts])
    sto_dnop = np.array([c[2] for c in sto_counts])

    fig, (ax_counts, ax_f1, ax_align) = plt.subplots(
        3, 1, figsize=(max(11, 0.55 * len(items)), 11), sharex=True
    )

    ax_counts.bar(xs_reb, reb_both, width=BAR_WIDTH, color=REB_COLORS["both"])
    ax_counts.bar(xs_reb, reb_slop, width=BAR_WIDTH, bottom=reb_both,
                  color=REB_COLORS["sleap_only"])
    ax_counts.bar(xs_reb, reb_dnop, width=BAR_WIDTH, bottom=reb_both + reb_slop,
                  color=REB_COLORS["dannce_only"])

    ax_counts.bar(xs_sto, sto_both, width=BAR_WIDTH, color=STO_COLORS["both"])
    ax_counts.bar(xs_sto, sto_slop, width=BAR_WIDTH, bottom=sto_both,
                  color=STO_COLORS["sleap_only"])
    ax_counts.bar(xs_sto, sto_dnop, width=BAR_WIDTH, bottom=sto_both + sto_slop,
                  color=STO_COLORS["dannce_only"])

    legend_handles = [
        mpatches.Patch(color=REB_COLORS["both"], label="rebuilt — both"),
        mpatches.Patch(color=REB_COLORS["sleap_only"], label="rebuilt — SLEAP-only"),
        mpatches.Patch(color=REB_COLORS["dannce_only"], label="rebuilt — DANNCE-only"),
        mpatches.Patch(color=STO_COLORS["both"], label="stored — both"),
        mpatches.Patch(color=STO_COLORS["sleap_only"], label="stored — SLEAP-only"),
        mpatches.Patch(color=STO_COLORS["dannce_only"], label="stored — DANNCE-only"),
    ]
    ax_counts.set_ylabel("# template matches (300 ms)")
    ax_counts.set_title(
        f"{rat} — TEMPLATE_1 REBUILT vs STORED — corrected_xyz match overlap  "
        f"[{LO} .. {HI}, total>={MIN_TOTAL_MATCHES} on either variant]"
    )
    ax_counts.legend(handles=legend_handles, fontsize=8, loc="upper right",
                     ncol=2)
    ax_counts.grid(axis="y", alpha=0.25)

    for label, color in LINE_COLORS.items():
        ys = [r.get(f"{label}_f1_300") for r in rows_ord]
        ls = "-o" if "corrected" in label else "--o"
        lw = 1.4 if "corrected" in label else 0.9
        ax_f1.plot(centers, ys, ls, color=color, label=label, lw=lw, ms=4,
                    alpha=1.0 if "corrected" in label else 0.7)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_ylabel("F1@300 ms")
    ax_f1.set_title(
        f"{rat} — F1: corrected_xyz rebuilt (green) vs stored (gray)  "
        "+ raw_xyz dashed reference"
    )
    ax_f1.legend(fontsize=8, loc="lower right", ncol=2)
    ax_f1.grid(alpha=0.25)

    kp_align = [r.get("kp_mse_align") for r in rows_ord]
    ax_align.plot(centers, kp_align, "-o", color="#d62728",
                  label="kp_mse_align (mm²)", lw=1.2, ms=4)
    ax_align.set_ylabel("kp_mse_align (mm²)", color="#d62728")
    ax_align.tick_params(axis="y", labelcolor="#d62728")
    ax_align.grid(alpha=0.25)
    ax_align.set_title(
        f"{rat} — SLEAP–DANNCE keypoint MSE after Procrustes alignment"
    )
    ax_align.set_xlabel("Session date")

    ax_align.xaxis_date()
    ax_align.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_align.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")

    fig.tight_layout()
    out_path = OUT_DIR / f"velacc_template1_rebuild_{rat}.png"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


# Per-rat summary table
print("\nPer-rat summary (means across kept sessions):")
keep_pairs = []
for rat in ["R1", "R2", "R3"]:
    keep_pairs.extend([(rat, r["session"]) for _, r, _, _ in per_rat[rat]])
keep_set = set(keep_pairs)
df_kept = df[df.apply(lambda r: (r["rat"], r["session"]) in keep_set, axis=1)].copy()
print(df_kept.groupby("rat").agg(
    n=("session", "count"),
    gt_rebuilt=("n_gt_xyz_rebuilt", "mean"),
    gt_stored=("n_gt_xyz_stored", "mean"),
    raw_reb=("raw_xyz_rebuilt_f1_300", "mean"),
    raw_sto=("raw_xyz_stored_f1_300", "mean"),
    corr_reb=("corrected_xyz_rebuilt_f1_300", "mean"),
    corr_sto=("corrected_xyz_stored_f1_300", "mean"),
).round(3).to_string())
