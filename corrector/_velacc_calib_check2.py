"""Split the NEW-calibration era into (a) 2026_02_06 only vs (b) after."""
import os
from pathlib import Path
import pandas as pd

df = pd.read_csv("corrector/results/velacc_session_correlates_calib.csv",
                 parse_dates=["date"])

def era(r):
    if r.calibration == "2025_12_08":
        return "OLD (pre-2026_02_06)"
    if str(r.date)[:10] == "2026-02-06":
        return "NEW, day-of (2026_02_06)"
    return "NEW, after (>=2026_02_07)"

df["era"] = df.apply(era, axis=1)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

print(df[["rat", "session", "date", "era", "f1go_raw", "f1go",
          "proc_resid", "kp_mse_align"]].sort_values(["era", "rat", "date"]).to_string(index=False))

print("\n=== Means by era (pooled across rats) ===")
g = df.groupby("era").agg(
    n=("session", "count"),
    f1go_raw=("f1go_raw", "mean"),
    f1go=("f1go", "mean"),
    proc_resid=("proc_resid", "mean"),
    kp_mse_align=("kp_mse_align", "mean"),
)
print(g.to_string())

print("\n=== Per rat ===")
g2 = df.groupby(["rat", "era"]).agg(
    n=("session", "count"),
    f1go_raw=("f1go_raw", "mean"),
    f1go=("f1go", "mean"),
    proc_resid=("proc_resid", "mean"),
    kp_mse_align=("kp_mse_align", "mean"),
)
print(g2.to_string())
