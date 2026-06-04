"""
F1 re-evaluation for a world-space corrector against the existing v3/v4 baseline.

For each test session we run template matching on three streams in the rat's
own template PC space:

  1. raw_sleap        : original SLEAP (no Procrustes, no corrector) — equivalent
                        to v3/v4 baseline 'A_xyz uniform max_outside=3'.
  2. procrustes_only  : Procrustes-aligned SLEAP, no corrector.
  3. corrected        : Procrustes + corrector outputs (mapped back to SLEAP space).

We score each against the DANNCE-detected matches (GT) at 100/300/500 ms
tolerance with the standard temporal-offset correction. The intent is to put
the corrector head-to-head with v3/v4 winners on the same metric.

Usage:
    python -m corrector.evaluate_f1 \
        --ckpt corrector/checkpoints/R2R3_world_mlp.pt \
        --extra_rats R1
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
from exp_utils import (compute_alignment_multi_tol, estimate_temporal_offset,
                        run_template_matching)
from skeleton import normalize_skeleton_batch

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.evaluate_world import RAT_TEMPLATE, project_to_template_pcs
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model

RESULTS_DIR = _THIS.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WIN = 30
TOLS = [100, 300, 500]
DEFAULT_BOUNDS = {"R1": 1.5, "R2": 1.0, "R3": 1.0}


def correct_world(model, x, device, batch=8192, ctx=1):
    """Apply a (possibly temporal) corrector to world-space SLEAP.

    For ctx > 1 we feed sliding windows; the first ctx-1 frames of the output
    repeat the model output at the first valid frame (zero residual).
    """
    out = np.empty_like(x, dtype=np.float32)
    if ctx <= 1:
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xt = torch.from_numpy(x[i:i + batch].astype(np.float32)).to(device)
                out[i:i + batch] = model(xt).cpu().numpy()
        return out
    # Windowed inference: accumulate windows in batches
    T = len(x)
    pad = ctx - 1
    # Pad with first frame (causal warmup)
    x_padded = np.concatenate([np.repeat(x[:1], pad, axis=0), x], axis=0)
    win_starts = np.arange(T)
    with torch.no_grad():
        for i in range(0, T, batch):
            idx = win_starts[i:i + batch]
            windows = np.stack([x_padded[s:s + ctx] for s in idx], axis=0)  # (B, ctx, 23, 3)
            xt = torch.from_numpy(windows.astype(np.float32)).to(device)
            out[i:i + batch] = model(xt).cpu().numpy()
    return out


def fit_and_evaluate_session(model, model_ctx, rat, session, device,
                              calibration_minutes=5.0,
                              calibration_n_sample=1000,
                              max_residual=60.0):
    """Returns dict with F1/recall/precision for raw/procrustes/corrected, or
    a dict with 'error' if the session can't be processed."""
    try:
        sl, dn = load_paired_world(rat, session)
    except Exception as e:
        return {"rat": rat, "session": session, "error": f"load: {e}"}
    if len(sl) < 1000:
        return {"rat": rat, "session": session, "error": "too few frames"}

    idx = calibration_indices(len(sl), calibration_minutes, SLEAP_HZ,
                               calibration_n_sample, seed=0)
    if len(idx) < 100:
        return {"rat": rat, "session": session, "error": "no calibration window"}
    tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
    procrustes_residual = float(tx["residual"])
    if procrustes_residual > max_residual:
        return {"rat": rat, "session": session,
                "error": f"residual {procrustes_residual:.1f} > {max_residual}"}

    sl_aligned_dannce_space = tx["apply"](sl).astype(np.float32)
    sl_corrected_dannce_space = correct_world(model, sl_aligned_dannce_space,
                                               device, ctx=model_ctx)
    # Bring everything back into SLEAP world space for projection through the
    # rat's template (which was fit in SLEAP egocentric space).
    sl_aligned_in_sleap = tx["apply_inverse"](sl_aligned_dannce_space).astype(np.float32)
    sl_corrected_in_sleap = tx["apply_inverse"](sl_corrected_dannce_space).astype(np.float32)
    dn_in_sleap = tx["apply_inverse"](dn).astype(np.float32)

    # Match to template — but the template was fit on z-FLIPPED SLEAP per
    # exp_utils.load_session_data. So flip z back here for compatibility.
    sl_for_template = sl.copy(); sl_for_template[:, :, 2] = -sl_for_template[:, :, 2]
    sl_aligned_for_template = sl_aligned_in_sleap.copy()
    sl_aligned_for_template[:, :, 2] = -sl_aligned_for_template[:, :, 2]
    sl_corrected_for_template = sl_corrected_in_sleap.copy()
    sl_corrected_for_template[:, :, 2] = -sl_corrected_for_template[:, :, 2]
    dn_for_template = dn_in_sleap.copy()
    dn_for_template[:, :, 2] = -dn_for_template[:, :, 2]

    # Project all four streams to the rat's template PC space
    tmpl = dict(load_template(rat, RAT_TEMPLATE[rat]))
    pcs_to_use = tmpl["pcs_to_use"].ravel().astype(int)
    feature_stds = tmpl["feature_stds"]
    template = tmpl["template"][:, pcs_to_use]
    bounds_scalar = DEFAULT_BOUNDS[rat]

    raw_pc = project_to_template_pcs(sl_for_template, tmpl, pcs_to_use)
    align_pc = project_to_template_pcs(sl_aligned_for_template, tmpl, pcs_to_use)
    corr_pc = project_to_template_pcs(sl_corrected_for_template, tmpl, pcs_to_use)
    dn_pc = project_to_template_pcs(dn_for_template, tmpl, pcs_to_use)

    # Time axis (SLEAP frame times)
    aligned = load_aligned_data(rat, session)
    st = np.array(aligned["sleap_times_ms"]).ravel()

    # Run template matching
    xyz_stds = feature_stds[pcs_to_use]
    bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
    gt_m = run_template_matching(dn_pc, template, bounds, max_outside=3,
                                 refractory_frames=WIN)
    if not gt_m:
        return {"rat": rat, "session": session, "error": "no DANNCE GT matches"}

    out = {"rat": rat, "session": session,
           "n_frames": int(len(dn_pc)),
           "procrustes_residual": procrustes_residual,
           "n_gt": len(gt_m)}
    for label, pcs in [("raw", raw_pc), ("procrustes", align_pc),
                        ("corrected", corr_pc)]:
        sl_m = run_template_matching(pcs, template, bounds, max_outside=3,
                                      refractory_frames=WIN)
        if len(sl_m) >= 2 and len(gt_m) >= 2:
            offset_ms = estimate_temporal_offset(sl_m, gt_m, st, st)
        else:
            offset_ms = 0.0
        al = compute_alignment_multi_tol(sl_m, gt_m, TOLS, st, st, offset_ms)
        out[f"{label}_n_sleap"] = len(sl_m)
        out[f"{label}_offset_ms"] = float(offset_ms)
        for tol in TOLS:
            r = al[f"tol_{tol}ms"]
            out[f"{label}_recall_{tol}"] = float(r["recall"])
            out[f"{label}_precision_{tol}"] = float(r["precision"])
            out[f"{label}_f1_{tol}"] = float(r["f1"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--extra_rats", nargs="*", default=[])
    ap.add_argument("--max_residual", type=float, default=60.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_kwargs = {}
    if ck["model_name"] == "temporal_mlp":
        model_kwargs["ctx"] = ck.get("ctx", 5)
    model_kwargs["hidden"] = ck.get("hidden", 128)
    model_kwargs["n_hidden_layers"] = ck.get("n_hidden_layers", 2)
    model = build_model(ck["model_name"], **model_kwargs)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    model_ctx = ck.get("ctx", 1)
    print(f"loaded {args.ckpt}: model={ck['model_name']} ctx={model_ctx}")

    rows = []
    held_out = ck["splits"]["test"]
    for rat, sessions in held_out.items():
        for s in sessions:
            print(f"  test {rat}/{s} ...", flush=True)
            r = fit_and_evaluate_session(model, model_ctx, rat, s, device,
                                          max_residual=args.max_residual)
            r["split"] = "test"
            rows.append(r)

    for rat in args.extra_rats:
        sessions = sorted(get_sessions(rat=rat)["session"].tolist())
        for s in sessions:
            print(f"  extra {rat}/{s} ...", flush=True)
            r = fit_and_evaluate_session(model, model_ctx, rat, s, device,
                                          max_residual=args.max_residual)
            r["split"] = "extra"
            rows.append(r)

    valid = [r for r in rows if "error" not in r]

    print(f"\nValid: {len(valid)} / {len(rows)}", flush=True)
    print(f"\n{'rat':<5} {'n':>3} {'raw F1':>8} {'proc F1':>9} {'corr F1':>9} "
          f"{'raw R':>8} {'proc R':>8} {'corr R':>8} "
          f"{'raw P':>8} {'proc P':>8} {'corr P':>8}")
    print("-" * 100)
    by_rat = {}
    for rat in sorted(set(r["rat"] for r in valid)):
        sub = [r for r in valid if r["rat"] == rat]
        agg = {}
        for label in ("raw", "procrustes", "corrected"):
            for met in ("recall", "precision", "f1"):
                agg[f"{label}_{met}_300"] = float(np.mean(
                    [r[f"{label}_{met}_300"] for r in sub]))
        by_rat[rat] = agg | {"n": len(sub)}
        print(f"{rat:<5} {len(sub):>3} "
              f"{agg['raw_f1_300']:>8.3f} {agg['procrustes_f1_300']:>9.3f} "
              f"{agg['corrected_f1_300']:>9.3f} "
              f"{agg['raw_recall_300']:>8.3f} {agg['procrustes_recall_300']:>8.3f} "
              f"{agg['corrected_recall_300']:>8.3f} "
              f"{agg['raw_precision_300']:>8.3f} {agg['procrustes_precision_300']:>8.3f} "
              f"{agg['corrected_precision_300']:>8.3f}")

    out = RESULTS_DIR / f"{Path(args.ckpt).stem}_f1.json"
    out.write_text(json.dumps({"rows": rows, "by_rat": by_rat}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
