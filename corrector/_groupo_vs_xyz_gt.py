"""For each session in the Oct 1 – Feb 5 range, compute the GT match indices
in both the stored xyz-PCA space (current online matcher) and the per-session
Group-O pooled-PCA space (using the winning groupO sweep parameters from the
existing eval JSON). Then compute how well the two ground-truth sets overlap
under the 300 ms tolerance comparison.

Writes corrector/results/velacc_groupo_vs_xyz_gt.csv with one row per session:
  rat, session, date,
  n_gt_xyz, n_gt_groupo,
  n_both, n_xyz_only, n_groupo_only,
  recall_xyz_in_groupo, recall_groupo_in_xyz, f1
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

from data_io import load_aligned_data, load_template
from exp_utils import (compute_alignment, compute_pairwise_distances,
                       run_template_matching)
from skeleton import normalize_skeleton_batch

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.evaluate_world import RAT_TEMPLATE
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model
from corrector.evaluate_all import correct_world

WIN = 30
DEFAULT_BOUNDS = {"R1": 1.5, "R2": 1.0, "R3": 1.0}
LO = date(2025, 10, 1)
HI = date(2026, 2, 5)


def session_date(sess):
    p = sess.split("_")
    return date(int(p[0]), int(p[1]), int(p[2]))


def main():
    # Load eval JSON for per-session winning groupO sweep params
    with open(REPO / "corrector/results/R1R2R3_velacc_all.json") as f:
        eval_data = json.load(f)
    eval_rows = {(r["rat"], r["session"]): r for r in eval_data["per_session"]}

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = str(REPO / "corrector/checkpoints/R1R2R3_velacc.pt")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"],
                        hidden=ck.get("hidden", 128),
                        n_hidden_layers=ck.get("n_hidden_layers", 2))
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()

    # Templates
    tmpl_cache = {rat: dict(load_template(rat, RAT_TEMPLATE[rat]))
                  for rat in ["R1", "R2", "R3"]}

    # Sessions in range
    sessions = []
    for (rat, sess), r in eval_rows.items():
        if "error" in r:
            continue
        if session_date(sess) < LO or session_date(sess) > HI:
            continue
        sessions.append((rat, sess))
    sessions.sort()
    print(f"Will process {len(sessions)} sessions", flush=True)

    rows_out = []
    for i, (rat, sess) in enumerate(sessions):
        ev = eval_rows[(rat, sess)]
        # Skip if groupO didn't produce results
        n_pcs = ev.get("corrected_groupO_n_pcs")
        bnd_scalar = ev.get("corrected_groupO_bounds_scalar")
        if n_pcs is None or bnd_scalar is None:
            print(f"  [{i+1}/{len(sessions)}] {rat}/{sess}: SKIP (no groupO sweep)",
                  flush=True)
            continue

        try:
            sl, dn = load_paired_world(rat, sess)
        except Exception as e:
            print(f"  [{i+1}/{len(sessions)}] {rat}/{sess}: load error {e}",
                  flush=True)
            continue
        if len(sl) < 1000:
            continue

        tmpl_data = tmpl_cache[rat]
        idx = calibration_indices(len(sl), 5.0, SLEAP_HZ, 1000, seed=0)
        if len(idx) < 100:
            continue
        tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)

        # Run velacc corrector
        sl_aligned_dn = tx["apply"](sl).astype(np.float32)
        sl_corr_dn = correct_world(model, sl_aligned_dn, device,
                                    ctx=1, vel_acc=True)

        eval_start = int(5 * 60 * SLEAP_HZ)
        if eval_start >= len(sl) - 1000:
            eval_start = 0
        sl_c_sleap = tx["apply_inverse"](sl_corr_dn).astype(np.float32)
        sl_c_ev = sl_c_sleap[eval_start:]
        dn_w = dn.astype(np.float32)
        dn_ev = tx["apply_inverse"](dn_w[eval_start:]).astype(np.float32)

        aligned = load_aligned_data(rat, sess)
        st_full = np.array(aligned["sleap_times_ms"]).ravel()
        st = st_full[eval_start: eval_start + len(dn_ev)]

        # ----- xyz GT matches -----
        def for_template(arr):
            out = arr.copy(); out[:, :, 2] = -out[:, :, 2]; return out

        dn_t = for_template(dn_ev)
        pcu = tmpl_data["pcs_to_use"].ravel().astype(int)
        feature_stds = tmpl_data["feature_stds"]
        template_pc = tmpl_data["template"][:, pcu]
        bounds_xyz = np.tile(feature_stds[pcu] * DEFAULT_BOUNDS[rat], (WIN, 1))

        # Project DANNCE into xyz template space
        from corrector.evaluate_world import project_to_template_pcs
        pc_dn_xyz = project_to_template_pcs(dn_t, tmpl_data, pcu)
        gt_xyz = run_template_matching(pc_dn_xyz, template_pc, bounds_xyz,
                                        max_outside=3, refractory_frames=WIN)
        if not gt_xyz:
            continue

        # ----- groupO GT matches -----
        sl_z = for_template(sl_c_ev)
        dn_z = for_template(dn_ev)
        sl_rot, _, _ = normalize_skeleton_batch(sl_z.astype(np.float64))
        dn_rot, _, _ = normalize_skeleton_batch(dn_z.astype(np.float64))
        sl_pw = compute_pairwise_distances(sl_rot)
        dn_pw = compute_pairwise_distances(dn_rot)
        pooled = np.vstack([sl_pw, dn_pw])
        pw_mean = pooled.mean(axis=0)
        _, _, Vt = np.linalg.svd(pooled - pw_mean, full_matrices=False)
        comps_full = Vt

        # Anchor the template in groupO using the best xyz-matched DANNCE window
        sl_orig_pc = ((sl_rot.reshape(len(sl_rot), -1)
                       - tmpl_data["feature_means"])
                      @ tmpl_data["pc_weights"].T)[:, pcu]
        dn_orig_pc = ((dn_rot.reshape(len(dn_rot), -1)
                       - tmpl_data["feature_means"])
                      @ tmpl_data["pc_weights"].T)[:, pcu]
        gt_anchor = run_template_matching(dn_orig_pc, template_pc, bounds_xyz,
                                           max_outside=3, refractory_frames=WIN)
        if not gt_anchor:
            continue
        best_err, best_f = np.inf, None
        for f in gt_anchor:
            chunk = dn_orig_pc[f - WIN + 1: f + 1]
            if len(chunk) < WIN:
                continue
            e = np.mean((chunk - template_pc) ** 2)
            if e < best_err:
                best_err, best_f = e, f
        if best_f is None:
            continue
        win_pw = dn_pw[best_f - WIN + 1: best_f + 1]
        win_pc_full = (win_pw - pw_mean) @ comps_full.T

        # Build groupO template + bounds using winning sweep params
        comps = comps_full[:n_pcs]
        sl_pc = (sl_pw - pw_mean) @ comps.T
        dn_pc = (dn_pw - pw_mean) @ comps.T
        tmpl_o = win_pc_full[:, :n_pcs]
        feat_stds = np.std(np.vstack([sl_pc, dn_pc]), axis=0)
        bnds = np.tile(feat_stds * bnd_scalar, (WIN, 1))
        gt_groupo = run_template_matching(dn_pc, tmpl_o, bnds,
                                            max_outside=3,
                                            refractory_frames=WIN)
        if not gt_groupo:
            continue

        # ----- compare xyz GT vs groupO GT under 300 ms tol -----
        # We're comparing two sets of DANNCE-side match indices to each other,
        # so "sleap_times" and "dannce_times" are both st.
        # treat xyz as "ground truth"; check what fraction of groupO matches
        # have an xyz match within 300 ms, and vice versa.
        align = compute_alignment(gt_groupo, gt_xyz, tolerance_ms=300.0,
                                   sleap_times_ms=st, dannce_times_ms=st,
                                   offset_ms=0.0)
        rec_xyz_in_go = align["recall"]      # fraction of xyz GT also in groupO GT
        prec_xyz_in_go = align["precision"]  # fraction of groupO GT also in xyz
        f1 = align["f1"]
        n_both = align["n_both"]

        out = dict(
            rat=rat, session=sess, date=str(session_date(sess)),
            n_gt_xyz=len(gt_xyz), n_gt_groupo=len(gt_groupo),
            n_both=int(n_both),
            n_xyz_only=int(align["n_dannce_only"]),
            n_groupo_only=int(align["n_sleap_only"]),
            recall_xyz_in_groupo=float(rec_xyz_in_go),
            precision_xyz_in_groupo=float(prec_xyz_in_go),
            f1=float(f1),
            groupo_n_pcs=int(n_pcs),
            groupo_bounds_scalar=float(bnd_scalar),
        )
        rows_out.append(out)
        if (i + 1) % 10 == 0 or i < 3:
            print(f"  [{i+1}/{len(sessions)}] {rat}/{sess}: "
                  f"xyz n={out['n_gt_xyz']}  groupO n={out['n_gt_groupo']}  "
                  f"both={out['n_both']}  F1={out['f1']:.3f}", flush=True)

    df = pd.DataFrame(rows_out)
    out_csv = REPO / "corrector/results/velacc_groupo_vs_xyz_gt.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(f"\n=== Means per rat ===")
    print(df.groupby("rat").agg(
        n_sessions=("session", "count"),
        n_gt_xyz_mean=("n_gt_xyz", "mean"),
        n_gt_groupo_mean=("n_gt_groupo", "mean"),
        f1_mean=("f1", "mean"),
        recall_xyz_in_groupo=("recall_xyz_in_groupo", "mean"),
        precision_xyz_in_groupo=("precision_xyz_in_groupo", "mean"),
    ).to_string())


if __name__ == "__main__":
    main()
