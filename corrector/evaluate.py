"""
Evaluate a trained corrector on the held-out test sessions.

Reports:
  - Keypoint-space MSE (mean over frames, all 23 keypoints)
  - Per-keypoint MSE (so we see which keypoints improve most)
  - Per-PC bias fraction before vs after correction (uses each rat's template
    PC weights from the existing pipeline). Lower bias fraction => more of the
    cross-system error is noise rather than structured offset.
  - Inference time per frame on CPU and GPU.
  - Template-matching F1 on test sessions (uses the v3 'A_xyz uniform' setting:
    runs the existing run_template_matching with original feature stds and the
    rat's current template). Reported separately for original SLEAP and
    corrected SLEAP.

Saves a JSON report to corrector/results/<rat>_<model>_eval.json.
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

from data_io import load_template
from exp_utils import (compute_alignment_multi_tol, estimate_temporal_offset,
                       load_session_data, run_template_matching, smooth_keypoints)
from skeleton import normalize_skeleton_batch, project_to_pcs
from corrector.data import SL_SMOOTH, DN_SMOOTH
from corrector.models import build_model

RESULTS_DIR = _THIS.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WIN = 30
TOLS = [100, 300, 500]

RAT_TEMPLATE = {
    "R1": "R1_template_1.npz",   # the SLEAP-derived one — also used by R1_secondary
    "R2": "R2_template_1.npz",
    "R3": "R3_template_1.npz",
}
DEFAULT_BOUNDS = {"R1": 1.5, "R2": 1.0, "R3": 1.0}


def load_corrector(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"],
                        hidden=ck.get("hidden", 128),
                        n_hidden_layers=ck.get("n_hidden_layers", 2))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def correct_session(model, sleap_egocentric: np.ndarray, device, batch=8192):
    """Apply the model to (T, 23, 3) numpy. Returns numpy float32 array."""
    out = np.empty_like(sleap_egocentric, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(sleap_egocentric), batch):
            x = torch.from_numpy(sleap_egocentric[i:i + batch]).to(device)
            out[i:i + batch] = model(x).cpu().numpy()
    return out


def per_pc_bias_fraction(sleap_pc: np.ndarray, dannce_pc: np.ndarray):
    """|mean(sleap-dannce)| / rmse(sleap-dannce)  per PC.

    > 0.5 means cross-system error is dominantly a fixed offset (systematic).
    """
    err = sleap_pc - dannce_pc
    mu = err.mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    return np.where(rmse > 0, np.abs(mu) / rmse, 0.0)


def keypoint_mse(sleap_eg: np.ndarray, dannce_eg: np.ndarray):
    """Mean squared error in egocentric keypoint coords. Returns (per_kp, overall)."""
    err = sleap_eg - dannce_eg                 # (T, 23, 3)
    per_kp = (err ** 2).sum(axis=2).mean(axis=0)   # (23,)
    overall = per_kp.mean()
    return per_kp, float(overall)


def template_match_f1(sleap_pc, dannce_pc, template, feature_stds, pcs_to_use,
                      bounds_scalar, st_ms):
    """Run the standard run_template_matching at uniform bounds (max_outside=3),
    same setting the v3 baseline uses, return SLEAP/DANNCE F1@300ms."""
    xyz_stds = feature_stds[pcs_to_use]
    tmpl = template[:, pcs_to_use]
    bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
    sl_m = run_template_matching(sleap_pc, tmpl, bounds, max_outside=3,
                                 refractory_frames=WIN)
    dn_m = run_template_matching(dannce_pc, tmpl, bounds, max_outside=3,
                                 refractory_frames=WIN)
    if not dn_m:
        return None
    if len(sl_m) >= 2 and len(dn_m) >= 2 and st_ms is not None:
        offset_ms = estimate_temporal_offset(sl_m, dn_m, st_ms, st_ms)
    else:
        offset_ms = 0.0
    al_sl = compute_alignment_multi_tol(sl_m, dn_m, TOLS, st_ms, st_ms, offset_ms)
    return {
        "n_sleap": len(sl_m), "n_dannce": len(dn_m),
        "offset_ms": float(offset_ms),
        "f1_300": al_sl["tol_300ms"]["f1"],
        "recall_300": al_sl["tol_300ms"]["recall"],
        "precision_300": al_sl["tol_300ms"]["precision"],
    }


def benchmark_inference(model, device, n_frames=10000, batch=512):
    """Per-frame inference time on the given device."""
    model.eval()
    x = torch.randn(batch, 23, 3, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_frames // batch):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    per_frame_us = elapsed / (n_frames // batch * batch) * 1e6
    return per_frame_us


def evaluate_one(rat: str, model_name: str = "mlp"):
    ckpt = _THIS.parent / "checkpoints" / f"{rat}_{model_name}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_corrector(ckpt, device)
    test_sessions = ck["splits"]["test"]
    print(f"{rat} {model_name}: {len(test_sessions)} test sessions")

    # Template / PCA setup
    tmpl = dict(load_template(rat, RAT_TEMPLATE[rat]))
    pcs_to_use = tmpl["pcs_to_use"].ravel().astype(int)
    fm, pw = tmpl["feature_means"], tmpl["pc_weights"]
    feature_stds = tmpl["feature_stds"]
    template_arr = tmpl["template"]
    bounds_scalar = DEFAULT_BOUNDS[rat]

    per_kp_baseline, per_kp_corrected = [], []
    pc_bias_baseline, pc_bias_corrected = [], []
    f1_baseline, f1_corrected = [], []
    overall_baseline, overall_corrected = [], []

    for s in test_sessions:
        try:
            sleap_3d, dannce_3d, aligned = load_session_data(rat, s)
        except Exception as e:
            print(f"  skip {s}: {e}"); continue
        sl = smooth_keypoints(sleap_3d, *SL_SMOOTH)
        dn = smooth_keypoints(dannce_3d, *DN_SMOOTH)
        sl_eg, _, _ = normalize_skeleton_batch(sl)
        dn_eg, _, _ = normalize_skeleton_batch(dn)
        sl_eg = sl_eg.astype(np.float32)
        dn_eg = dn_eg.astype(np.float32)

        # Apply corrector
        sl_eg_c = correct_session(model, sl_eg, device)

        # Keypoint-space MSE
        per_kp_b, ov_b = keypoint_mse(sl_eg, dn_eg)
        per_kp_c, ov_c = keypoint_mse(sl_eg_c, dn_eg)
        per_kp_baseline.append(per_kp_b); per_kp_corrected.append(per_kp_c)
        overall_baseline.append(ov_b); overall_corrected.append(ov_c)

        # Project both to the rat's PC space
        flat_b = sl_eg.reshape(len(sl_eg), -1)
        flat_c = sl_eg_c.reshape(len(sl_eg_c), -1)
        flat_d = dn_eg.reshape(len(dn_eg), -1)
        sl_pc_b = ((flat_b - fm) @ pw.T)[:, pcs_to_use]
        sl_pc_c = ((flat_c - fm) @ pw.T)[:, pcs_to_use]
        dn_pc   = ((flat_d - fm) @ pw.T)[:, pcs_to_use]

        pc_bias_baseline.append(per_pc_bias_fraction(sl_pc_b, dn_pc))
        pc_bias_corrected.append(per_pc_bias_fraction(sl_pc_c, dn_pc))

        # Template matching
        st_ms = np.array(aligned["sleap_times_ms"]).ravel() if aligned else None
        r_b = template_match_f1(sl_pc_b, dn_pc, template_arr, feature_stds,
                                pcs_to_use, bounds_scalar, st_ms)
        r_c = template_match_f1(sl_pc_c, dn_pc, template_arr, feature_stds,
                                pcs_to_use, bounds_scalar, st_ms)
        if r_b is not None: f1_baseline.append(r_b)
        if r_c is not None: f1_corrected.append(r_c)

    per_kp_b = np.mean(per_kp_baseline, axis=0) if per_kp_baseline else None
    per_kp_c = np.mean(per_kp_corrected, axis=0) if per_kp_corrected else None
    pc_b = np.mean(pc_bias_baseline, axis=0) if pc_bias_baseline else None
    pc_c = np.mean(pc_bias_corrected, axis=0) if pc_bias_corrected else None

    f1_mean_b = float(np.mean([r["f1_300"] for r in f1_baseline])) if f1_baseline else None
    f1_mean_c = float(np.mean([r["f1_300"] for r in f1_corrected])) if f1_corrected else None
    r_mean_b = float(np.mean([r["recall_300"] for r in f1_baseline])) if f1_baseline else None
    r_mean_c = float(np.mean([r["recall_300"] for r in f1_corrected])) if f1_corrected else None
    p_mean_b = float(np.mean([r["precision_300"] for r in f1_baseline])) if f1_baseline else None
    p_mean_c = float(np.mean([r["precision_300"] for r in f1_corrected])) if f1_corrected else None

    # Inference time
    cpu_us = benchmark_inference(model.to("cpu"), torch.device("cpu"))
    gpu_us = benchmark_inference(model.to(device), device) if device.type == "cuda" else None

    report = {
        "rat": rat, "model_name": model_name,
        "ckpt": str(ckpt),
        "n_test_sessions_used": len(per_kp_baseline),
        "keypoint_mse_baseline": float(np.mean(overall_baseline)) if overall_baseline else None,
        "keypoint_mse_corrected": float(np.mean(overall_corrected)) if overall_corrected else None,
        "per_keypoint_mse_baseline": per_kp_b.tolist() if per_kp_b is not None else None,
        "per_keypoint_mse_corrected": per_kp_c.tolist() if per_kp_c is not None else None,
        "per_pc_bias_fraction_baseline": pc_b.tolist() if pc_b is not None else None,
        "per_pc_bias_fraction_corrected": pc_c.tolist() if pc_c is not None else None,
        "f1_300_baseline": f1_mean_b, "f1_300_corrected": f1_mean_c,
        "recall_300_baseline": r_mean_b, "recall_300_corrected": r_mean_c,
        "precision_300_baseline": p_mean_b, "precision_300_corrected": p_mean_c,
        "inference_us_per_frame_cpu": cpu_us,
        "inference_us_per_frame_gpu": gpu_us,
    }
    out = RESULTS_DIR / f"{rat}_{model_name}_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved report to {out}")
    print(f"  keypoint MSE: baseline={report['keypoint_mse_baseline']:.2f} "
          f"-> corrected={report['keypoint_mse_corrected']:.2f}")
    if pc_b is not None and pc_c is not None:
        print(f"  PC bias frac (mean over PCs): baseline={pc_b.mean():.3f} -> {pc_c.mean():.3f}")
    if f1_mean_b is not None and f1_mean_c is not None:
        print(f"  F1@300 ms:    baseline={f1_mean_b:.3f} -> corrected={f1_mean_c:.3f}")
    print(f"  inference: CPU={cpu_us:.1f} us/frame   GPU={gpu_us:.1f} us/frame"
          if gpu_us is not None else f"  inference: CPU={cpu_us:.1f} us/frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", required=True, choices=["R1", "R2", "R3"])
    ap.add_argument("--model", default="mlp", choices=["linear", "mlp"])
    args = ap.parse_args()
    evaluate_one(args.rat, args.model)


if __name__ == "__main__":
    main()
