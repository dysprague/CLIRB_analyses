"""
F1 evaluation: corrector + v4 Group O (pairwise distances + pooled SLEAP+DANNCE PCA).

For each test session we run template matching on three SLEAP variants:

  raw              : original SLEAP (matches the v4 Group O baseline exactly)
  procrustes       : Procrustes-aligned SLEAP, no MLP
  corrected        : Procrustes + MLP corrector outputs (mapped back to SLEAP space)

For each variant we sweep:
  n_pcs        ∈ {2, 3}
  bounds_scalar ∈ {1.0, 1.25, 1.5}
  max_outside  ∈ {1, 2, 3}

For each (variant, sweep setting) we score against DANNCE-detected matches in
the same pairwise-distance pooled-PCA space, with the standard temporal-offset
correction.

Saves a JSON of all rows + per-rat per-variant best F1 to
corrector/results/<ckpt_stem>_f1_combined.json.

Usage:
    python -m corrector.evaluate_f1_combined \
        --ckpt corrector/checkpoints/R1R2R3_world_temporal_mlp.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from data_io import (get_sessions, load_aligned_data, load_sleap_dannce_keys,
                      load_template)
from exp_utils import (compute_alignment_multi_tol, compute_pairwise_distances,
                       estimate_temporal_offset, run_template_matching)
from skeleton import normalize_skeleton_batch

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model

RESULTS_DIR = _THIS.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WIN = 30
TOLS = [100, 300, 500]
N_PCS_SWEEP = [2, 3]
BOUNDS_SWEEP = [1.0, 1.25, 1.5]
MAX_OUT_SWEEP = [1, 2, 3]


def correct_world(model, x, device, batch=8192, ctx: int = 1):
    out = np.empty_like(x, dtype=np.float32)
    if ctx <= 1:
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xt = torch.from_numpy(x[i:i + batch].astype(np.float32)).to(device)
                out[i:i + batch] = model(xt).cpu().numpy()
        return out
    T = len(x); pad = ctx - 1
    x_padded = np.concatenate([np.repeat(x[:1], pad, axis=0), x], axis=0)
    win_starts = np.arange(T)
    with torch.no_grad():
        for i in range(0, T, batch):
            idx = win_starts[i:i + batch]
            windows = np.stack([x_padded[s:s + ctx] for s in idx], axis=0)
            xt = torch.from_numpy(windows.astype(np.float32)).to(device)
            out[i:i + batch] = model(xt).cpu().numpy()
    return out


def find_best_dannce_window(dn_pw_pcs, tmpl_pw_pcs, feat_stds, bounds_scalar):
    """Surrogate for Group O's per-session-best-DANNCE-window template fallback.
    Pick the DANNCE GT match whose pairwise-distance window has the lowest MSE
    to the template — used when no GT is found and we need a session-anchored
    template.
    """
    best_frame, best_err = None, np.inf
    for f in range(WIN, len(dn_pw_pcs)):
        chunk = dn_pw_pcs[f - WIN + 1:f + 1]
        err = np.mean((chunk - tmpl_pw_pcs) ** 2)
        if err < best_err:
            best_err = err
            best_frame = f
    return best_frame


def evaluate_session(model, model_ctx, rat, session, device, max_residual=60.0,
                     calibration_minutes=5.0, calibration_n_sample=1000):
    """Run all 3 variants × full sweep, return one list of rows per session.

    Each row dict has rat, session, variant, n_pcs, bounds_scalar, max_outside,
    and the standard sleap/recall/precision/F1 fields.
    """
    try:
        sl, dn = load_paired_world(rat, session)
    except Exception as e:
        return [{"rat": rat, "session": session, "error": f"load: {e}"}]
    if len(sl) < 1000:
        return [{"rat": rat, "session": session, "error": "too few frames"}]

    idx = calibration_indices(len(sl), calibration_minutes, SLEAP_HZ,
                               calibration_n_sample, seed=0)
    if len(idx) < 100:
        return [{"rat": rat, "session": session, "error": "no calibration window"}]
    tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
    if tx["residual"] > max_residual:
        return [{"rat": rat, "session": session,
                 "error": f"residual {tx['residual']:.1f} > {max_residual}"}]

    sl_aligned_dn = tx["apply"](sl).astype(np.float32)
    sl_corrected_dn = correct_world(model, sl_aligned_dn, device, ctx=model_ctx)
    # Map back to SLEAP world space for downstream pipeline (matches qc/render flow)
    sl_aligned_sleap = tx["apply_inverse"](sl_aligned_dn).astype(np.float32)
    sl_corrected_sleap = tx["apply_inverse"](sl_corrected_dn).astype(np.float32)

    # The v4 pipeline operates on z-FLIPPED SLEAP (per exp_utils.load_session_data).
    # Do the same for all three variants so matching is comparable.
    sl_for_v4 = sl.copy()
    sl_for_v4[:, :, 2] = -sl_for_v4[:, :, 2]
    sl_aligned_for_v4 = sl_aligned_sleap.copy()
    sl_aligned_for_v4[:, :, 2] = -sl_aligned_for_v4[:, :, 2]
    sl_corrected_for_v4 = sl_corrected_sleap.copy()
    sl_corrected_for_v4[:, :, 2] = -sl_corrected_for_v4[:, :, 2]

    # DANNCE: same as v4 — already in DANNCE world space, post-resample
    dn_for_v4 = dn.astype(np.float64)

    # Time axis
    aligned = load_aligned_data(rat, session)
    st = np.array(aligned["sleap_times_ms"]).ravel()

    # Original template (used to find the per-session "best DANNCE window" anchor)
    from corrector.evaluate_world import RAT_TEMPLATE
    tmpl_data = dict(load_template(rat, RAT_TEMPLATE[rat]))
    pcu = tmpl_data["pcs_to_use"].ravel().astype(int)
    pw_orig = tmpl_data["pc_weights"]
    fm_orig = tmpl_data["feature_means"]
    template_pcs = tmpl_data["template"][:, pcu]                 # (WIN, n_pcs)
    feature_stds_orig = tmpl_data["feature_stds"]
    bounds_anchor = float(tmpl_data["bounds"])

    rows = []

    # For each SLEAP variant, fit a fresh pooled PCA on (variant + DANNCE) pairwise distances.
    variants = {
        "raw":         sl_for_v4.astype(np.float64),
        "procrustes":  sl_aligned_for_v4.astype(np.float64),
        "corrected":   sl_corrected_for_v4.astype(np.float64),
    }

    for variant_name, sl_arr in variants.items():
        sl_rot, _, _ = normalize_skeleton_batch(sl_arr)
        dn_rot, _, _ = normalize_skeleton_batch(dn_for_v4)
        sl_pw = compute_pairwise_distances(sl_rot)
        dn_pw = compute_pairwise_distances(dn_rot)

        # Pooled PCA (this is the v4 Group O step)
        pooled = np.vstack([sl_pw, dn_pw])
        pw_mean = pooled.mean(axis=0)
        centered = pooled - pw_mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        comps_full = Vt

        # Find a per-session anchor template: project DANNCE through the original
        # PC space (using egocentric-normalized coords — sl_rot, dn_rot), run
        # template matching to get GT, pick the best DANNCE match window, and
        # use those frames in pairwise-distance space as the template.
        sl_flat = sl_rot.reshape(len(sl_rot), -1)
        dn_flat = dn_rot.reshape(len(dn_rot), -1)
        sl_orig_pc = ((sl_flat - fm_orig) @ pw_orig.T)[:, pcu]
        dn_orig_pc = ((dn_flat - fm_orig) @ pw_orig.T)[:, pcu]
        bounds_orig = np.tile(feature_stds_orig[pcu] * bounds_anchor,
                              (WIN, 1))
        gt_anchor = run_template_matching(dn_orig_pc, template_pcs, bounds_orig,
                                          max_outside=3,
                                          refractory_frames=WIN)
        if not gt_anchor:
            continue
        # Pick the best-scoring DANNCE GT window
        best_err = np.inf; best_f = None
        for f in gt_anchor:
            chunk = dn_orig_pc[f - WIN + 1:f + 1]
            if len(chunk) < WIN: continue
            err = np.mean((chunk - template_pcs) ** 2)
            if err < best_err:
                best_err = err; best_f = f
        if best_f is None: continue

        # That window in PAIRWISE-DISTANCE space, projected to the new pooled PCA
        win_pw = dn_pw[best_f - WIN + 1:best_f + 1]
        win_pc_full = (win_pw - pw_mean) @ comps_full.T

        # Sweep
        for n_pcs in N_PCS_SWEEP:
            comps = comps_full[:n_pcs]
            sl_pc = (sl_pw - pw_mean) @ comps.T
            dn_pc = (dn_pw - pw_mean) @ comps.T
            tmpl = win_pc_full[:, :n_pcs]
            feat_stds = np.std(np.vstack([sl_pc, dn_pc]), axis=0)

            for scalar in BOUNDS_SWEEP:
                bounds = np.tile(feat_stds * scalar, (WIN, 1))
                # GT in pairwise space — same as v4 Group O
                gt_m = run_template_matching(dn_pc, tmpl, bounds,
                                              max_outside=3,
                                              refractory_frames=WIN)
                if not gt_m:
                    continue
                # Estimate offset from the SLEAP variant's own initial matches
                sl_init = run_template_matching(sl_pc, tmpl, bounds,
                                                max_outside=3,
                                                refractory_frames=WIN)
                if len(sl_init) >= 2 and len(gt_m) >= 2:
                    offset_ms = estimate_temporal_offset(sl_init, gt_m, st, st)
                else:
                    offset_ms = 0.0

                for mo in MAX_OUT_SWEEP:
                    sl_m = run_template_matching(sl_pc, tmpl, bounds,
                                                 max_outside=mo,
                                                 refractory_frames=WIN)
                    al = compute_alignment_multi_tol(sl_m, gt_m, TOLS, st, st,
                                                      offset_ms)
                    row = {
                        "rat": rat, "session": session,
                        "variant": variant_name,
                        "n_pcs": n_pcs, "bounds_scalar": scalar,
                        "max_outside": mo,
                        "n_gt": len(gt_m), "n_sleap": len(sl_m),
                        "offset_ms": float(offset_ms),
                    }
                    for tol in TOLS:
                        r = al[f"tol_{tol}ms"]
                        row[f"recall_{tol}"] = float(r["recall"])
                        row[f"precision_{tol}"] = float(r["precision"])
                        row[f"f1_{tol}"] = float(r["f1"])
                    rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_residual", type=float, default=60.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_kwargs = dict(hidden=ck.get("hidden", 128),
                        n_hidden_layers=ck.get("n_hidden_layers", 2))
    if ck["model_name"] == "temporal_mlp":
        model_kwargs["ctx"] = ck.get("ctx", 5)
    model = build_model(ck["model_name"], **model_kwargs)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    model_ctx = ck.get("ctx", 1)
    print(f"loaded {args.ckpt}: {ck['model_name']} ctx={model_ctx}", flush=True)

    rows = []
    for rat, sessions in ck["splits"]["test"].items():
        for s in sessions:
            t0 = time.time()
            r = evaluate_session(model, model_ctx, rat, s, device,
                                  max_residual=args.max_residual)
            ok = sum(1 for x in r if "error" not in x)
            print(f"  {rat}/{s}: {ok} rows ({time.time() - t0:.1f}s)", flush=True)
            rows.extend(r)

    valid = [r for r in rows if "error" not in r]
    print(f"\nValid rows: {len(valid)} / {len(rows)}", flush=True)

    # Aggregate: best per-variant per-rat over the sweep
    import collections
    groups = collections.defaultdict(list)
    for r in valid:
        key = (r["rat"], r["variant"], r["n_pcs"], r["bounds_scalar"], r["max_outside"])
        groups[key].append(r)

    # Mean F1 per setting
    settings = []
    for (rat, var, n_pcs, scalar, mo), rs in groups.items():
        if len(rs) < 2: continue
        settings.append({
            "rat": rat, "variant": var,
            "n_pcs": n_pcs, "bounds_scalar": scalar, "max_outside": mo,
            "n_sessions": len(rs),
            "f1_300": float(np.mean([r["f1_300"] for r in rs])),
            "recall_300": float(np.mean([r["recall_300"] for r in rs])),
            "precision_300": float(np.mean([r["precision_300"] for r in rs])),
        })

    print(f"\n{'rat':<5} {'variant':<11} {'n_pcs':>5} {'bounds':>7} {'mo':>3} "
          f"{'f1':>6} {'recall':>7} {'prec':>6}  n")
    print("-" * 80)
    # Best per (rat, variant)
    best = {}
    for s in settings:
        k = (s["rat"], s["variant"])
        if k not in best or s["f1_300"] > best[k]["f1_300"]:
            best[k] = s
    for (rat, var) in sorted(best):
        s = best[(rat, var)]
        print(f"{rat:<5} {var:<11} {s['n_pcs']:>5} {s['bounds_scalar']:>7.2f} "
              f"{s['max_outside']:>3} {s['f1_300']:>6.3f} {s['recall_300']:>7.3f} "
              f"{s['precision_300']:>6.3f}  {s['n_sessions']}")

    # Re-key best_per_variant from tuple to "rat::variant" string for JSON
    best_serializable = {f"{k[0]}::{k[1]}": v for k, v in best.items()}
    out = RESULTS_DIR / f"{Path(args.ckpt).stem}_f1_combined.json"
    out.write_text(json.dumps({"rows": rows, "best_per_variant": best_serializable,
                                "all_settings": settings}, indent=2))
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
