"""Re-run the template_1 analysis using the REBUILT template
(<rat>_template_1_rebuild.npz) instead of the stored one.

For each session in 2025-10-25..2025-12-10 (same window as
_velacc_oct25_dec10_template1_plots.py), compute three F1 variants:

  1) raw_xyz_rebuilt       -- raw SLEAP, rebuilt template_1
  2) corrected_xyz_rebuilt -- velacc-corrected SLEAP, rebuilt template_1
  3) raw_xyz_stored        -- raw SLEAP, stored template_1   (for comparison)
  4) corrected_xyz_stored  -- velacc-corrected SLEAP, stored template_1

Ground truth: DANNCE matched against the SAME template (rebuilt for variants
1/2, stored for variants 3/4). This keeps each comparison internally
consistent — rebuilt-vs-rebuilt and stored-vs-stored.

Writes:
  corrector/results/velacc_template1_rebuild.csv
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from data_io import load_aligned_data, load_template  # noqa: E402
from exp_utils import (compute_alignment_multi_tol,  # noqa: E402
                        estimate_temporal_offset, run_template_matching)
from skeleton import normalize_skeleton_batch  # noqa: E402

from corrector.data_world import SLEAP_HZ, load_paired_world  # noqa: E402
from corrector.evaluate_world import project_to_template_pcs  # noqa: E402
from corrector.world_alignment import calibration_indices, fit_procrustes  # noqa: E402
from corrector.models import build_model  # noqa: E402
from corrector.evaluate_all import correct_world, DEFAULT_BOUNDS, WIN, TOLS  # noqa: E402

LO = date(2025, 10, 25)
HI = date(2025, 12, 10)


def session_date(sess):
    p = sess.split("_")
    return date(int(p[0]), int(p[1]), int(p[2]))


def main():
    with open(REPO / "corrector/results/R1R2R3_velacc_all.json") as f:
        eval_data = json.load(f)
    eval_rows = {(r["rat"], r["session"]): r for r in eval_data["per_session"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = str(REPO / "corrector/checkpoints/R1R2R3_velacc.pt")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"],
                        hidden=ck.get("hidden", 128),
                        n_hidden_layers=ck.get("n_hidden_layers", 2))
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()

    stored_tmpl = {rat: dict(load_template(rat, f"{rat}_template_1.npz"))
                   for rat in ["R1", "R2", "R3"]}
    rebuilt_tmpl = {rat: dict(load_template(rat, f"{rat}_template_1_rebuild.npz"))
                    for rat in ["R1", "R2", "R3"]}

    sessions = []
    for (rat, sess), r in eval_rows.items():
        if "error" in r:
            continue
        d = session_date(sess)
        if d < LO or d > HI:
            continue
        sessions.append((rat, sess))
    sessions.sort()
    print(f"Will process {len(sessions)} sessions", flush=True)

    rows_out = []
    for i, (rat, sess) in enumerate(sessions):
        try:
            sl, dn = load_paired_world(rat, sess)
        except Exception as e:
            print(f"  [{i+1}/{len(sessions)}] {rat}/{sess}: load err {e}",
                  flush=True)
            continue
        if len(sl) < 1000:
            continue

        idx = calibration_indices(len(sl), 5.0, SLEAP_HZ, 1000, seed=0)
        if len(idx) < 100:
            continue
        tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
        if tx["residual"] > 60.0:
            continue

        sl_aligned_dn = tx["apply"](sl).astype(np.float32)
        sl_corr_dn = correct_world(model, sl_aligned_dn, device,
                                    ctx=1, vel_acc=True)
        eval_start = int(5 * 60 * SLEAP_HZ)
        if eval_start >= len(sl) - 1000:
            eval_start = 0
        sl_corr_sleap = tx["apply_inverse"](sl_corr_dn).astype(np.float32)
        dn_w = dn.astype(np.float32)

        def for_template(arr):
            out = arr.copy(); out[:, :, 2] = -out[:, :, 2]; return out

        sl_raw_t = for_template(sl[eval_start:])
        sl_c_t = for_template(sl_corr_sleap[eval_start:])
        dn_t = for_template(tx["apply_inverse"](dn_w[eval_start:]))

        aligned = load_aligned_data(rat, sess)
        st_full = np.array(aligned["sleap_times_ms"]).ravel()
        st = st_full[eval_start: eval_start + len(sl_raw_t)]

        row = dict(rat=rat, session=sess, date=str(session_date(sess)),
                   procrustes_residual=float(tx["residual"]))

        a_eval = sl_aligned_dn[eval_start:]
        d_eval = dn_w[eval_start:]
        row["kp_mse_align"] = float(((a_eval - d_eval) ** 2).sum(axis=2).mean())

        for tmpl_label, tmpl in [("rebuilt", rebuilt_tmpl[rat]),
                                  ("stored", stored_tmpl[rat])]:
            pcu = tmpl["pcs_to_use"].ravel().astype(int)
            feat_stds = tmpl["feature_stds"]
            bnds_scalar = DEFAULT_BOUNDS[rat]
            bnds = np.tile(feat_stds[pcu] * bnds_scalar, (WIN, 1))
            template_pc = tmpl["template"][:, pcu]

            pc_raw = project_to_template_pcs(sl_raw_t, tmpl, pcu)
            pc_corr = project_to_template_pcs(sl_c_t, tmpl, pcu)
            pc_d = project_to_template_pcs(dn_t, tmpl, pcu)

            gt = run_template_matching(pc_d, template_pc, bnds,
                                        max_outside=3, refractory_frames=WIN)
            row[f"n_gt_xyz_{tmpl_label}"] = int(len(gt))
            if not gt:
                continue

            for variant_label, variant_pcs in [("raw_xyz", pc_raw),
                                                ("corrected_xyz", pc_corr)]:
                sl_m = run_template_matching(variant_pcs, template_pc, bnds,
                                              max_outside=3,
                                              refractory_frames=WIN)
                if len(sl_m) >= 2 and len(gt) >= 2:
                    off = estimate_temporal_offset(sl_m, gt, st, st)
                else:
                    off = 0.0
                al = compute_alignment_multi_tol(sl_m, gt, TOLS, st, st, off)
                r300 = al["tol_300ms"]
                tag = f"{variant_label}_{tmpl_label}"
                row[f"{tag}_n_sleap"] = int(len(sl_m))
                row[f"{tag}_recall_300"] = float(r300["recall"])
                row[f"{tag}_precision_300"] = float(r300["precision"])
                row[f"{tag}_f1_300"] = float(r300["f1"])
                row[f"{tag}_offset_ms"] = float(off)

        rows_out.append(row)
        if (i + 1) % 20 == 0 or i < 3:
            n_gt_r = row.get("n_gt_xyz_rebuilt", -1)
            n_gt_s = row.get("n_gt_xyz_stored", -1)
            f1_r = row.get("corrected_xyz_rebuilt_f1_300", float("nan"))
            f1_s = row.get("corrected_xyz_stored_f1_300", float("nan"))
            print(f"  [{i+1}/{len(sessions)}] {rat}/{sess}  "
                  f"gt(rebuilt={n_gt_r} stored={n_gt_s})  "
                  f"corr_f1(rebuilt={f1_r:.3f} stored={f1_s:.3f})",
                  flush=True)

    df = pd.DataFrame(rows_out)
    out_csv = REPO / "corrector/results/velacc_template1_rebuild.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df)} rows)")

    print("\nPer-rat means (n excludes sessions with no GT):")
    print(df.groupby("rat").agg(
        n=("session", "count"),
        gt_mean_rebuilt=("n_gt_xyz_rebuilt", "mean"),
        gt_mean_stored=("n_gt_xyz_stored", "mean"),
        raw_f1_rebuilt=("raw_xyz_rebuilt_f1_300", "mean"),
        raw_f1_stored=("raw_xyz_stored_f1_300", "mean"),
        corr_f1_rebuilt=("corrected_xyz_rebuilt_f1_300", "mean"),
        corr_f1_stored=("corrected_xyz_stored_f1_300", "mean"),
    ).to_string())


if __name__ == "__main__":
    main()
