"""Per-session correlates of velacc F1gO performance.

For each session in R1R2R3_velacc_all.json compute:
  F1gO, kp_mse_corr/align, Procrustes residual, date,
  mean SLEAP 2D confidence, fraction low-confidence detections.
Saves a CSV and prints Spearman correlations.
"""
import json, sys
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

with open("corrector/results/R1R2R3_velacc_all.json") as f:
    d = json.load(f)
rows = d["per_session"]

DATA = Path("/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data")

records = []
for r in rows:
    rat, sess = r["rat"], r["session"]
    parts = sess.split("_")
    sess_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
    sleap_2d_path = DATA / rat / sess / "sleap" / "sleap_keys_2D.npy"
    mean_conf = float("nan"); frac_low = float("nan")
    if sleap_2d_path.exists():
        a = np.load(sleap_2d_path)
        conf = a[..., 2]
        mean_conf = float(np.nanmean(conf))
        frac_low = float(np.mean(conf < 0.5))
    records.append(dict(
        rat=rat, session=sess, date=sess_date,
        f1go=r.get("corrected_groupO_f1_300"),
        f1go_raw=r.get("raw_groupO_f1_300"),
        f1xyz=r.get("corrected_xyz_f1_300"),
        kp_mse_corr=r.get("kp_mse_corr"),
        kp_mse_align=r.get("kp_mse_align"),
        proc_resid=r.get("procrustes_residual"),
        mean_conf=mean_conf, frac_low_conf=frac_low,
    ))

df = pd.DataFrame(records).sort_values(["rat", "date"]).reset_index(drop=True)
out_csv = "corrector/results/velacc_session_correlates.csv"
df.to_csv(out_csv, index=False)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)
print(df.to_string())
print(f"\nWrote {out_csv}", flush=True)

print("\n=== Spearman rank correlations of F1gO with covariates (per rat) ===")
for rat in ["R1", "R2", "R3"]:
    sub = df[df.rat == rat]
    print(f"\n{rat} (n={len(sub)}):")
    for col in ["proc_resid", "kp_mse_align", "kp_mse_corr", "mean_conf", "frac_low_conf"]:
        x = sub[col].values; y = sub["f1go"].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            continue
        rho, pval = spearmanr(x[m], y[m])
        print(f"  F1gO vs {col:<15}  rho={rho:+.2f}  p={pval:.3f}")

print("\n=== Pooled across rats ===")
for col in ["proc_resid", "kp_mse_align", "kp_mse_corr", "mean_conf", "frac_low_conf"]:
    x = df[col].values; y = df["f1go"].values
    m = np.isfinite(x) & np.isfinite(y)
    rho, pval = spearmanr(x[m], y[m])
    print(f"  F1gO vs {col:<15}  rho={rho:+.2f}  p={pval:.3f}")

print("\nDone.", flush=True)
