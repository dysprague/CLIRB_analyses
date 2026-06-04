"""Evaluate the v2 rebuilt templates against the stored and Phase-H rebuild.

For each clean-window session (2025-10-25..2025-12-10) that passes the same
Procrustes-residual gate Phase H used, evaluate three template variants per
rat per template-family (template_1 and template_2):

  stored   -- the production .npz from /olveczky_lab/.../templates/
  phaseh   -- <rat>_template_1_rebuild.npz (Phase H, template_1 only — for
              template_2 we fall back to 'stored')
  v2       -- results/template_rebuild_v2/<rat>_template_{1,2}.npz, with
              per-timepoint sigma-with-floor bounds applied where available

For each variant, run two SLEAP detectors:
  raw_xyz       -- raw SLEAP through the variant's PCA
  corrected_xyz -- velacc-corrected SLEAP through the variant's PCA

Ground truth per variant = DANNCE matched against the SAME variant
(internally consistent, the convention from handoff §2.6).

F1 at 300ms tolerance, with temporal-offset estimation.

Writes:
  results/template_rebuild_v2/eval_results.csv
  results/template_rebuild_v2/eval_summary.txt
"""
from __future__ import annotations

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
from exp_utils import (compute_alignment_multi_tol, estimate_temporal_offset,
                        run_template_matching)
from skeleton import normalize_skeleton_batch

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.evaluate_world import project_to_template_pcs
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model
from corrector.evaluate_all import correct_world, DEFAULT_BOUNDS, WIN, TOLS

V2_DIR = REPO / "results" / "template_rebuild_v2"
OUT_CSV = V2_DIR / "eval_results.csv"
OUT_SUMMARY = V2_DIR / "eval_summary.txt"
TEMPLATE_DIR = Path("/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data")

LO = date(2025, 10, 25)
HI = date(2025, 12, 10)
TEMPLATE_FAMILIES = ["template_1", "template_2"]


def session_date(sess):
    p = sess.split("_")
    return date(int(p[0]), int(p[1]), int(p[2]))


def load_variants(rat, family):
    """Return dict variant_label -> template dict for this rat & family."""
    out = {}
    out["stored"] = dict(load_template(rat, f"{rat}_{family}.npz"))
    if family == "template_1":
        ph = TEMPLATE_DIR / rat / "templates" / f"{rat}_template_1_rebuild.npz"
        if ph.exists():
            out["phaseh"] = dict(np.load(ph, allow_pickle=True))
    v2_path = V2_DIR / f"{rat}_{family}.npz"
    if v2_path.exists():
        out["v2"] = dict(np.load(v2_path, allow_pickle=True))
    return out


def bounds_for_variant(tmpl, rat):
    """Build the (WIN, n_pcs) bounds array used for matching.

    - v2 templates carry per-timepoint pc_template_bounds; use that sliced to pcs_to_use.
    - Stored / Phase H use flat DEFAULT_BOUNDS[rat] * feature_stds[pcs_to_use].
    """
    pcu = tmpl["pcs_to_use"].ravel().astype(int)
    if "pc_template_bounds" in tmpl:
        b = np.asarray(tmpl["pc_template_bounds"])
        return b[:, pcu]
    feat_stds = tmpl["feature_stds"]
    scalar = DEFAULT_BOUNDS[rat]
    return np.tile(feat_stds[pcu] * scalar, (WIN, 1))


def for_template(arr):
    out = arr.copy()
    out[:, :, 2] = -out[:, :, 2]
    return out


def main():
    # Session list from the existing velacc per-session JSON. This already
    # encodes the "passed corrector eval" gate, identical to what
    # _velacc_template1_rebuild_eval used.
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
    print(f"Loaded corrector on {device}", flush=True)

    rats = ["R1", "R2", "R3"]
    # Pre-load all template variants per rat per family.
    variants_cache = {(rat, fam): load_variants(rat, fam)
                      for rat in rats for fam in TEMPLATE_FAMILIES}
    for (rat, fam), v in variants_cache.items():
        print(f"  {rat}/{fam}: variants present = {list(v.keys())}", flush=True)

    sessions = []
    for (rat, sess), r in eval_rows.items():
        if "error" in r:
            continue
        d = session_date(sess)
        if d < LO or d > HI:
            continue
        sessions.append((rat, sess))
    sessions.sort()
    print(f"\nWill process {len(sessions)} sessions\n", flush=True)

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

        sl_raw_t = for_template(sl[eval_start:])
        sl_c_t = for_template(sl_corr_sleap[eval_start:])
        dn_t = for_template(tx["apply_inverse"](dn_w[eval_start:]))

        aligned = load_aligned_data(rat, sess)
        st_full = np.array(aligned["sleap_times_ms"]).ravel()
        st = st_full[eval_start: eval_start + len(sl_raw_t)]

        for family in TEMPLATE_FAMILIES:
            variants = variants_cache[(rat, family)]
            for v_label, tmpl in variants.items():
                pcu = tmpl["pcs_to_use"].ravel().astype(int)
                template_pc = tmpl["template"][:, pcu]
                bnds = bounds_for_variant(tmpl, rat)

                pc_raw = project_to_template_pcs(sl_raw_t, tmpl, pcu)
                pc_corr = project_to_template_pcs(sl_c_t, tmpl, pcu)
                pc_d = project_to_template_pcs(dn_t, tmpl, pcu)

                gt = run_template_matching(pc_d, template_pc, bnds,
                                            max_outside=3,
                                            refractory_frames=WIN)

                row_base = dict(
                    rat=rat, session=sess, date=str(session_date(sess)),
                    family=family, variant=v_label,
                    procrustes_residual=float(tx["residual"]),
                    n_gt=int(len(gt)),
                )

                if not gt:
                    rows_out.append(row_base)
                    continue

                for det_label, det_pcs in [("raw_xyz", pc_raw),
                                            ("corrected_xyz", pc_corr)]:
                    sl_m = run_template_matching(det_pcs, template_pc, bnds,
                                                  max_outside=3,
                                                  refractory_frames=WIN)
                    if len(sl_m) >= 2 and len(gt) >= 2:
                        off = estimate_temporal_offset(sl_m, gt, st, st)
                    else:
                        off = 0.0
                    al = compute_alignment_multi_tol(sl_m, gt, TOLS, st, st, off)
                    r300 = al["tol_300ms"]
                    tag = det_label
                    row_base[f"{tag}_n_sleap"] = int(len(sl_m))
                    row_base[f"{tag}_recall_300"] = float(r300["recall"])
                    row_base[f"{tag}_precision_300"] = float(r300["precision"])
                    row_base[f"{tag}_f1_300"] = float(r300["f1"])
                    row_base[f"{tag}_offset_ms"] = float(off)

                rows_out.append(row_base)

        if (i + 1) % 10 == 0 or i < 3:
            t1_v2 = next((r for r in rows_out[-6:]
                          if r["family"] == "template_1" and r["variant"] == "v2"),
                         None)
            t1_st = next((r for r in rows_out[-6:]
                          if r["family"] == "template_1" and r["variant"] == "stored"),
                         None)
            if t1_v2 and t1_st:
                f1v2 = t1_v2.get("corrected_xyz_f1_300", float("nan"))
                f1st = t1_st.get("corrected_xyz_f1_300", float("nan"))
                print(f"  [{i+1}/{len(sessions)}] {rat}/{sess}  "
                      f"t1 corr_f1  v2={f1v2:.3f}  stored={f1st:.3f}",
                      flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows)", flush=True)

    # ----- Summary -----
    lines = []
    lines.append(f"Eval window: {LO} .. {HI}")
    lines.append(f"Total session-rows: {len(df)}")
    lines.append("")
    for family in TEMPLATE_FAMILIES:
        lines.append(f"=== {family} ===")
        sub = df[df["family"] == family]
        agg = sub.groupby(["rat", "variant"]).agg(
            n=("session", "count"),
            gt_mean=("n_gt", "mean"),
            raw_f1=("raw_xyz_f1_300", "mean"),
            corr_f1=("corrected_xyz_f1_300", "mean"),
            raw_recall=("raw_xyz_recall_300", "mean"),
            corr_recall=("corrected_xyz_recall_300", "mean"),
            raw_precision=("raw_xyz_precision_300", "mean"),
            corr_precision=("corrected_xyz_precision_300", "mean"),
        ).round(3)
        lines.append(agg.to_string())
        lines.append("")
    out = "\n".join(lines)
    OUT_SUMMARY.write_text(out)
    print(out)
    print(f"\nWrote {OUT_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
