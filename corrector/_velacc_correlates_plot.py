"""Plot velacc F1gO vs covariates (date, Procrustes residual, mean SLEAP confidence,
fraction low-confidence) per rat, plus per-rat F1gO timeline.

Reads corrector/results/velacc_session_correlates.csv.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("corrector/results/velacc_session_correlates.csv",
                 parse_dates=["date"])
out_dir = Path("corrector/figures")
out_dir.mkdir(exist_ok=True)

rats = ["R1", "R2", "R3"]
colors = {"R1": "#1f77b4", "R2": "#2ca02c", "R3": "#d62728"}

# Figure 1: F1gO timeline per rat (chronological)
fig, ax = plt.subplots(figsize=(11, 5))
for rat in rats:
    sub = df[df.rat == rat].sort_values("date")
    ax.plot(sub.date, sub.f1go, "-o", color=colors[rat], label=f"{rat} corrected",
            lw=1.5, ms=6)
    ax.plot(sub.date, sub.f1go_raw, "--^", color=colors[rat], alpha=0.45,
            label=f"{rat} raw", lw=1, ms=4)
ax.axhline(0.7, color="gray", lw=0.6, ls=":")
ax.set_ylabel("F1@300ms (Group-O)")
ax.set_xlabel("Session date")
ax.set_ylim(0, 1.05)
ax.set_title("velacc model — per-session F1gO over time")
ax.legend(ncol=3, fontsize=8)
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(out_dir / "velacc_timeline.png", dpi=130)
plt.close(fig)

# Figure 2: 2x2 scatter of F1gO vs covariates (color = rat)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
specs = [
    ("proc_resid",      "Procrustes residual (mm)"),
    ("kp_mse_align",    "kp_mse (Procrustes-aligned, pre-correction)"),
    ("mean_conf",       "mean SLEAP 2D confidence"),
    ("frac_low_conf",   "fraction of SLEAP detections with conf<0.5"),
]
for ax, (col, xlabel) in zip(axes.flat, specs):
    for rat in rats:
        sub = df[df.rat == rat]
        ax.scatter(sub[col], sub.f1go, s=55, color=colors[rat], label=rat,
                   edgecolor="k", lw=0.5)
    ax.axhline(0.7, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1gO (corrected)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    # Annotate session labels for the bottom 2 sessions per rat
    for rat in rats:
        sub = df[df.rat == rat].sort_values("f1go").head(2)
        for _, r in sub.iterrows():
            ax.annotate(r.session[-7:], (r[col], r.f1go), fontsize=7,
                        xytext=(4, 2), textcoords="offset points",
                        color=colors[rat])
axes[0, 0].legend(fontsize=9, loc="lower left")
fig.suptitle("velacc — F1gO vs session covariates (low F1 sessions labeled)",
             fontsize=11)
fig.tight_layout()
fig.savefig(out_dir / "velacc_correlates.png", dpi=130)
plt.close(fig)

# Figure 3: change in F1gO from raw to corrected, per rat, sorted
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, rat in zip(axes, rats):
    sub = df[df.rat == rat].sort_values("f1go_raw")
    n = len(sub)
    x = np.arange(n)
    ax.bar(x - 0.18, sub.f1go_raw, width=0.36, color="lightgray",
           edgecolor="k", lw=0.4, label="raw")
    ax.bar(x + 0.18, sub.f1go, width=0.36, color=colors[rat],
           edgecolor="k", lw=0.4, label="velacc corrected")
    ax.set_xticks(x)
    ax.set_xticklabels([s[5:] for s in sub.session], rotation=70, fontsize=7)
    ax.set_title(f"{rat} (n={n})")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
axes[0].set_ylabel("F1@300ms (Group-O)")
fig.suptitle("velacc — raw vs corrected F1gO, per session", fontsize=11)
fig.tight_layout()
fig.savefig(out_dir / "velacc_raw_vs_corrected_bars.png", dpi=130)
plt.close(fig)

print("Wrote:")
print(f"  {out_dir/'velacc_timeline.png'}")
print(f"  {out_dir/'velacc_correlates.png'}")
print(f"  {out_dir/'velacc_raw_vs_corrected_bars.png'}")
