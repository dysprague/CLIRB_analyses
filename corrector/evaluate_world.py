"""
Evaluate a world-space corrector.

For each test session in the checkpoint's splits we:
  1. Load raw (sleap_world, dannce_world).
  2. Fit Procrustes on the first 5 min calibration epoch.
  3. Apply Procrustes-aligned-SLEAP through the corrector.
  4. Compare baseline (Procrustes-only) and corrected against DANNCE.

Reports per (rat, session):
  - keypoint MSE (mm^2 across xyz, mean over frames and 23 keypoints)
  - per-keypoint MSE
  - bone-length consistency (per-edge std of lengths)
  - per-PC MSE in the rat's own template PC space

Aggregated and saved to corrector/results/<tag>_world_<model>_eval.json.

Usage:
    python -m corrector.evaluate_world --ckpt corrector/checkpoints/R2R3_world_mlp.pt
    python -m corrector.evaluate_world --ckpt ... --extra_rats R1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from config import EDGES, NODES
from data_io import load_template
from skeleton import normalize_skeleton_batch, project_to_pcs

from corrector.data_world import (SLEAP_HZ, WorldPairedDataset, load_paired_world,
                                   session_split_multi)
from corrector.world_alignment import calibration_indices, fit_procrustes
from corrector.models import build_model

RESULTS_DIR = _THIS.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RAT_TEMPLATE = {
    "R1": "R1_template_1_rebuild.npz",
    "R2": "R2_template_1_rebuild.npz",
    "R3": "R3_template_1_rebuild.npz",
}


def correct_world(model, x: np.ndarray, device, batch=8192,
                  ctx: int = 1) -> np.ndarray:
    """Apply a (possibly temporal) corrector. ctx=1 for single-frame models,
    ctx>1 for temporal models — sliding causal window with first-frame padding."""
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


def per_keypoint_mse(a: np.ndarray, b: np.ndarray):
    """a, b: (T, 23, 3)  → returns per_kp (23,) and overall (scalar) MSE."""
    err = (a - b) ** 2
    per_kp = err.sum(axis=2).mean(axis=0)            # (23,)
    return per_kp, float(per_kp.mean())


def bone_length_consistency(pts: np.ndarray, edges=EDGES):
    """Returns (mean_length_per_edge, std_per_edge, cv_per_edge)."""
    n_edges = len(edges)
    lengths = np.zeros((len(pts), n_edges), dtype=np.float32)
    for ei, (i, j) in enumerate(edges):
        lengths[:, ei] = np.linalg.norm(pts[:, i, :] - pts[:, j, :], axis=1)
    return lengths.mean(axis=0), lengths.std(axis=0), \
           lengths.std(axis=0) / (lengths.mean(axis=0) + 1e-8)


def project_to_template_pcs(world_pts: np.ndarray, template: dict,
                             pcs_to_use=None):
    """Egocentrically normalize, then project to the rat's PC space."""
    eg, _, _ = normalize_skeleton_batch(world_pts.astype(np.float64))
    flat = eg.reshape(len(eg), -1).astype(np.float64)
    fm, pw = template["feature_means"], template["pc_weights"]
    pcs = (flat - fm) @ pw.T
    if pcs_to_use is None:
        pcs_to_use = template["pcs_to_use"].ravel().astype(int)
    return pcs[:, pcs_to_use]


def evaluate_session(model, rat, session, device, max_residual=60.0,
                     calibration_minutes=5.0, calibration_n_sample=1000,
                     ctx: int = 1):
    """Returns dict with metrics for one session, or None if unusable."""
    try:
        sl, dn = load_paired_world(rat, session)
    except Exception as e:
        return {"error": f"load: {e}"}

    if len(sl) < 1000:
        return {"error": "too few frames"}
    idx = calibration_indices(len(sl), calibration_minutes, SLEAP_HZ,
                               calibration_n_sample, seed=0)
    if len(idx) < 100:
        return {"error": "too few calibration frames"}
    tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
    procrustes_residual = float(tx["residual"])
    if procrustes_residual > max_residual:
        return {"error": f"residual {procrustes_residual:.1f} > {max_residual}"}

    sl_aligned = tx["apply"](sl).astype(np.float32)
    sl_corrected = correct_world(model, sl_aligned, device, ctx=ctx)

    # Skip the calibration window when computing metrics so we don't
    # over-credit ourselves on the data the alignment was fit to
    eval_start = int(calibration_minutes * 60 * SLEAP_HZ)
    if eval_start >= len(sl) - 1000:
        eval_start = 0

    a_eval = sl_aligned[eval_start:]
    c_eval = sl_corrected[eval_start:]
    d_eval = dn[eval_start:].astype(np.float32)

    per_kp_align, mse_align = per_keypoint_mse(a_eval, d_eval)
    per_kp_corr, mse_corr = per_keypoint_mse(c_eval, d_eval)

    # Bone length: report each system's own internal bone-length variance
    bl_sleap_aligned = bone_length_consistency(a_eval)
    bl_sleap_corr = bone_length_consistency(c_eval)
    bl_dannce = bone_length_consistency(d_eval)

    # PC-space MSE in the rat's own PC space
    tmpl = dict(load_template(rat, RAT_TEMPLATE[rat]))
    pcs_to_use = tmpl["pcs_to_use"].ravel().astype(int)
    sl_pc_align = project_to_template_pcs(a_eval, tmpl, pcs_to_use)
    sl_pc_corr = project_to_template_pcs(c_eval, tmpl, pcs_to_use)
    dn_pc = project_to_template_pcs(d_eval, tmpl, pcs_to_use)

    pc_mse_align = ((sl_pc_align - dn_pc) ** 2).mean(axis=0)
    pc_mse_corr = ((sl_pc_corr - dn_pc) ** 2).mean(axis=0)

    return {
        "rat": rat, "session": session,
        "n_frames": int(len(d_eval)),
        "procrustes_residual": procrustes_residual,
        "keypoint_mse_aligned": float(mse_align),
        "keypoint_mse_corrected": float(mse_corr),
        "per_keypoint_mse_aligned": per_kp_align.tolist(),
        "per_keypoint_mse_corrected": per_kp_corr.tolist(),
        "bone_cv_sleap_aligned": bl_sleap_aligned[2].tolist(),
        "bone_cv_sleap_corrected": bl_sleap_corr[2].tolist(),
        "bone_cv_dannce": bl_dannce[2].tolist(),
        "pc_mse_aligned": pc_mse_align.tolist(),
        "pc_mse_corrected": pc_mse_corr.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--extra_rats", nargs="*", default=[],
                    help="rats to also evaluate on (all sessions, not just splits)")
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
    eval_ctx = ck.get("ctx", 1)

    rows = []
    held_out = ck["splits"]["test"]
    for rat, sessions in held_out.items():
        for s in sessions:
            print(f"  test {rat}/{s} ...", flush=True)
            r = evaluate_session(model, rat, s, device,
                                 max_residual=args.max_residual,
                                 ctx=eval_ctx)
            r["split"] = "test"
            rows.append(r)

    # All sessions of extra_rats (for cross-rat generalization)
    from data_io import get_sessions
    for rat in args.extra_rats:
        sessions = sorted(get_sessions(rat=rat)["session"].tolist())
        for s in sessions:
            print(f"  extra {rat}/{s} ...", flush=True)
            r = evaluate_session(model, rat, s, device,
                                 max_residual=args.max_residual,
                                 ctx=eval_ctx)
            r["split"] = "extra"
            rows.append(r)

    # Aggregate
    valid = [r for r in rows if "error" not in r]
    print(f"\nValid sessions: {len(valid)} / {len(rows)}", flush=True)

    def agg(group_rows, label):
        if not group_rows:
            print(f"  {label}: no sessions"); return None
        ma = np.mean([r["keypoint_mse_aligned"] for r in group_rows])
        mc = np.mean([r["keypoint_mse_corrected"] for r in group_rows])
        n_pcs = len(group_rows[0]["pc_mse_aligned"])
        pa = np.mean([r["pc_mse_aligned"] for r in group_rows], axis=0)
        pc = np.mean([r["pc_mse_corrected"] for r in group_rows], axis=0)
        print(f"  {label}: keypoint MSE  aligned={ma:.2f} -> corrected={mc:.2f}  "
              f"(n={len(group_rows)})", flush=True)
        for j in range(min(n_pcs, 4)):
            print(f"    PC{j+1} MSE  aligned={pa[j]:7.3f} -> corrected={pc[j]:7.3f}",
                  flush=True)
        return {"keypoint_mse_aligned": float(ma),
                "keypoint_mse_corrected": float(mc),
                "pc_mse_aligned": pa.tolist(), "pc_mse_corrected": pc.tolist(),
                "n": len(group_rows)}

    print("\nPer-rat aggregate:", flush=True)
    by_rat = {}
    for rat in sorted(set(r["rat"] for r in valid)):
        sub = [r for r in valid if r["rat"] == rat]
        by_rat[rat] = agg(sub, rat)

    out = RESULTS_DIR / f"{Path(args.ckpt).stem}_eval.json"
    out.write_text(json.dumps({"rows": rows, "by_rat": by_rat}, indent=2))
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
