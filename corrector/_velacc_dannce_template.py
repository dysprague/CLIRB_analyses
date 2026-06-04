"""For each session in 2025-10-25..2025-12-10, compute three F1 variants
using *_template_2.npz* (NOT template_1):

  1) raw_xyz                  -- raw SLEAP, stored template_2 (SLEAP-defined)
  2) corrected_xyz_stored     -- velacc-corrected SLEAP, stored template_2
  3) corrected_xyz_dannce     -- velacc-corrected SLEAP, DANNCE-defined template
                                 (rebuilt at template_2's origin_idx using
                                  DANNCE keypoints — same PC space + bounds)

Ground truth: DANNCE keypoints matched against the DANNCE-defined template
(same template that's used in variant 3). This GT set is used for ALL three
SLEAP variants.

Writes:
  corrector/results/velacc_template2_three_variants.csv
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from data_io import (load_aligned_data, load_template, load_sleap_dannce_keys,
                     template_path)
from exp_utils import (compute_alignment_multi_tol, run_template_matching,
                       estimate_temporal_offset)
from skeleton import normalize_skeleton_batch

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.evaluate_world import project_to_template_pcs
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model
from corrector.evaluate_all import correct_world, DEFAULT_BOUNDS, WIN, TOLS

LO = date(2025, 10, 25)
HI = date(2025, 12, 10)

TEMPLATE_FILE = {rat: f"{rat}_template_2.npz" for rat in ["R1", "R2", "R3"]}


def session_date(sess):
    p = sess.split("_")
    return date(int(p[0]), int(p[1]), int(p[2]))


def build_dannce_template(stored_tmpl):
    """DANNCE template at origin window, projected through the *_template_2*
    PC weights and feature_means."""
    temp_origin_str = str(stored_tmpl["temp_origin_file"])
    temp_rat, temp_sess = temp_origin_str.strip().split("/")
    temp_idx = int(stored_tmpl["temp_origin_idx"])
    template_length = int(stored_tmpl["template"].shape[0])

    keys = load_sleap_dannce_keys(temp_rat, temp_sess)
    aligned = load_aligned_data(temp_rat, temp_sess)
    aligned_idx = aligned["dannce_idx_for_sleap_cams"].astype(int).ravel()[1:]

    dn = keys["dannce_keys_3D"]
    if dn.ndim == 4:
        dn = dn.squeeze(axis=1).transpose(0, 2, 1)
    else:
        dn = np.transpose(dn, [0, 2, 1])
    dn = dn[aligned_idx, :, :]
    dn = median_filter(dn, size=(11, 1, 1))

    start = max(0, temp_idx)
    end = temp_idx + template_length
    window = dn[start:end]
    if window.shape[0] < template_length:
        pad = template_length - window.shape[0]
        window = np.concatenate(
            [window[:1].repeat(pad, axis=0), window], axis=0)

    rot, _, _ = normalize_skeleton_batch(window.astype(np.float64))
    flat = rot.reshape(template_length, -1)
    fm = stored_tmpl["feature_means"]
    pw = stored_tmpl["pc_weights"]
    pcu = stored_tmpl["pcs_to_use"].ravel().astype(int)
    pcs_full = (flat - fm) @ pw.T
    return pcs_full[:, pcu].astype(np.float64)


def main():
    # Use the existing eval JSON only to enumerate the session list.
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

    # Load template_2 + build DANNCE-defined template per rat.
    stored = {}
    dannce_pc = {}
    for rat in ["R1", "R2", "R3"]:
        t = dict(load_template(rat, TEMPLATE_FILE[rat]))
        stored[rat] = t
        dannce_pc[rat] = build_dannce_template(t)
        pcu = t["pcs_to_use"].ravel().astype(int)
        diff = np.linalg.norm(dannce_pc[rat] - t["template"][:, pcu])
        print(f"  {rat} template_2: origin={t['temp_origin_file']}  "
              f"idx={int(t['temp_origin_idx'])}  pcs={pcu.tolist()}  "
              f"DANNCE-vs-stored L2={diff:.3f}", flush=True)

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

        tmpl = stored[rat]
        pcu = tmpl["pcs_to_use"].ravel().astype(int)
        feat_stds = tmpl["feature_stds"]
        bnds_scalar = DEFAULT_BOUNDS[rat]
        bnds = np.tile(feat_stds[pcu] * bnds_scalar, (WIN, 1))
        tmpl_stored = tmpl["template"][:, pcu]
        tmpl_dannce = dannce_pc[rat]

        pc_raw = project_to_template_pcs(sl_raw_t, tmpl, pcu)
        pc_corr = project_to_template_pcs(sl_c_t, tmpl, pcu)
        pc_d = project_to_template_pcs(dn_t, tmpl, pcu)

        aligned = load_aligned_data(rat, sess)
        st_full = np.array(aligned["sleap_times_ms"]).ravel()
        st = st_full[eval_start: eval_start + len(pc_d)]

        gt = run_template_matching(pc_d, tmpl_dannce, bnds,
                                    max_outside=3, refractory_frames=WIN)
        if not gt:
            continue

        def score(sl_pcs, tmpl_pc, label_out):
            sl_m = run_template_matching(sl_pcs, tmpl_pc, bnds,
                                          max_outside=3, refractory_frames=WIN)
            if len(sl_m) >= 2 and len(gt) >= 2:
                off = estimate_temporal_offset(sl_m, gt, st, st)
            else:
                off = 0.0
            al = compute_alignment_multi_tol(sl_m, gt, TOLS, st, st, off)
            r300 = al["tol_300ms"]
            return {
                f"{label_out}_n_sleap": int(len(sl_m)),
                f"{label_out}_recall_300": float(r300["recall"]),
                f"{label_out}_precision_300": float(r300["precision"]),
                f"{label_out}_f1_300": float(r300["f1"]),
                f"{label_out}_offset_ms": float(off),
            }

        row = dict(rat=rat, session=sess, date=str(session_date(sess)),
                   n_gt_xyz=int(len(gt)),
                   procrustes_residual=float(tx["residual"]))
        row.update(score(pc_raw, tmpl_stored, "raw_xyz"))
        row.update(score(pc_corr, tmpl_stored, "corrected_xyz_stored"))
        row.update(score(pc_corr, tmpl_dannce, "corrected_xyz_dannce"))

        # kp_mse_align for the alignment plot
        a_eval = sl_aligned_dn[eval_start:]
        d_eval = dn_w[eval_start:]
        row["kp_mse_align"] = float(((a_eval - d_eval) ** 2).sum(axis=2).mean())
        rows_out.append(row)

        if (i + 1) % 20 == 0 or i < 3:
            print(f"  [{i+1}/{len(sessions)}] {rat}/{sess}  gt={len(gt)}  "
                  f"raw={row['raw_xyz_f1_300']:.3f}  "
                  f"corr_stored={row['corrected_xyz_stored_f1_300']:.3f}  "
                  f"corr_dannce={row['corrected_xyz_dannce_f1_300']:.3f}",
                  flush=True)

    df = pd.DataFrame(rows_out)
    out_csv = REPO / "corrector/results/velacc_template2_three_variants.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df)} rows)")
    print(df.groupby("rat").agg(
        n=("session", "count"),
        gt_mean=("n_gt_xyz", "mean"),
        raw_f1=("raw_xyz_f1_300", "mean"),
        corr_stored_f1=("corrected_xyz_stored_f1_300", "mean"),
        corr_dannce_f1=("corrected_xyz_dannce_f1_300", "mean"),
    ).to_string())


if __name__ == "__main__":
    main()
