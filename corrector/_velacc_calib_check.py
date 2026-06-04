import os
from pathlib import Path
import pandas as pd

df = pd.read_csv("corrector/results/velacc_session_correlates.csv",
                 parse_dates=["date"])
DATA = Path("/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data")

cals = []
for _, r in df.iterrows():
    cal_dir = DATA / r.rat / r.session / "sleap" / "calibration"
    subs = sorted(os.listdir(cal_dir)) if cal_dir.exists() else []
    cals.append(subs[0] if subs else "<missing>")
df["calibration"] = cals
df["cal_short"] = df.calibration.map(
    {"2025_12_08": "OLD (2025_12_08)", "2026_02_06": "NEW (2026_02_06)"}
).fillna(df.calibration)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

print(df[["rat", "session", "date", "calibration", "f1go_raw", "f1go",
          "proc_resid", "kp_mse_align"]].to_string(index=False))

print("\n=== Summary by calibration era ===")
g = df.groupby("cal_short").agg(
    n=("session", "count"),
    f1go_raw_mean=("f1go_raw", "mean"),
    f1go_mean=("f1go", "mean"),
    proc_resid_mean=("proc_resid", "mean"),
    kp_align_mean=("kp_mse_align", "mean"),
)
print(g.to_string())

print("\n=== Per-rat ===")
g2 = df.groupby(["rat", "cal_short"]).agg(
    n=("session", "count"),
    f1go_raw=("f1go_raw", "mean"),
    f1go=("f1go", "mean"),
    proc_resid=("proc_resid", "mean"),
    kp_align=("kp_mse_align", "mean"),
)
print(g2.to_string())
df.to_csv("corrector/results/velacc_session_correlates_calib.csv", index=False)
print("\nWrote corrector/results/velacc_session_correlates_calib.csv")
