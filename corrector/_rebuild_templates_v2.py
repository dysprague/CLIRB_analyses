"""Rebuild template_1 and template_2 from DANNCE keypoints, v2.

Generalizes corrector/_rebuild_template1.py:
  * Pools every session in the clean window 2025-10-25..2025-12-10 from the
    canonical good-sessions CSV (Procrustes residual <= 60 mm gate, same as
    Phase H), instead of a hand-picked first-K list.
  * Operates on BOTH template_1 and template_2 with the same recipe. The
    anchor is always the *stored* template (SLEAP-side matches), so this is
    Phase H's recipe applied to a wider pool, twice.
  * Estimates per-timepoint, per-PC bounds from the within-motif std across
    instances, with a floor so reliable-by-coincidence frames don't go to
    zero: bounds[t, pc] = max(k * std_within(t, pc), floor * feature_stds[pc])
  * Writes outputs under CLIRB_analyses/results/template_rebuild_v2/ so the
    shared data drive is not touched.

Outputs:
  results/template_rebuild_v2/<rat>_template_{1,2}.npz
  results/template_rebuild_v2/rebuild_meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from data_io import get_sessions, load_template
from exp_utils import run_template_matching
from skeleton import normalize_skeleton_batch

from corrector.data_world import SLEAP_HZ, load_paired_world
from corrector.evaluate_all import DEFAULT_BOUNDS, WIN
from corrector.world_alignment import calibration_indices, fit_procrustes

OUT_DIR = REPO / "results" / "template_rebuild_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean window from Phase G/H. Inclusive on both ends.
WINDOW_START = "2025_10_25"
WINDOW_END = "2025_12_10"
MAX_RESIDUAL_MM = 60.0
MIN_FRAMES = 1000

# Per-timepoint bound knobs.
#
# Calibration policy: scale per-timepoint sigma bounds so that their MEAN
# across (t, pcs_to_use) matches the stored flat bound width
# (bounds_scalar * feature_stds[pcu]). This makes v2 the same average
# corridor width as production, with per-timepoint shaping doing the work.
# A per-PC floor still prevents pathological tightening.
CALIBRATE_TO_STORED_MEAN = True
SIGMA_K_FALLBACK = 2.0   # used only if CALIBRATE_TO_STORED_MEAN is False
FLOOR_FRAC = 0.5         # floor each bound at floor_frac * (bounds_scalar * feature_stds[pc])

RATS = ["R1", "R2", "R3"]
TEMPLATE_FILES = {
    "template_1": lambda rat: f"{rat}_template_1.npz",
    "template_2": lambda rat: f"{rat}_template_2.npz",
}


def for_template_z_flip(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[:, :, 2] = -out[:, :, 2]
    return out


def project_sleap_to_template_pcs(sleap_xyz_world, tmpl):
    pcu = tmpl["pcs_to_use"].ravel().astype(int)
    flipped = for_template_z_flip(sleap_xyz_world.astype(np.float64))
    rot, _, _ = normalize_skeleton_batch(flipped)
    flat = rot.reshape(len(rot), -1)
    pcs_full = (flat - tmpl["feature_means"]) @ tmpl["pc_weights"].T
    return pcs_full[:, pcu]


def project_dannce_to_template_pcs(dannce_xyz_world, tmpl):
    pcu = tmpl["pcs_to_use"].ravel().astype(int)
    rot, _, _ = normalize_skeleton_batch(dannce_xyz_world.astype(np.float64))
    flat = rot.reshape(len(rot), -1)
    pcs_full = (flat - tmpl["feature_means"]) @ tmpl["pc_weights"].T
    return pcs_full[:, pcu]


def sessions_in_window(rat):
    df = get_sessions(rat=rat)
    in_win = df[(df["session"] >= WINDOW_START) & (df["session"] <= WINDOW_END)]
    return sorted(in_win["session"].unique().tolist())


def rebuild_one_template(rat, template_file):
    tmpl = dict(load_template(rat, template_file))
    pcu = tmpl["pcs_to_use"].ravel().astype(int)
    template_pc = tmpl["template"][:, pcu]
    feature_stds = tmpl["feature_stds"]
    bounds_scalar = float(tmpl.get("bounds", DEFAULT_BOUNDS[rat]))
    flat_bounds_for_matching = np.tile(feature_stds[pcu] * bounds_scalar, (WIN, 1))

    sessions = sessions_in_window(rat)
    print(f"  pool: {len(sessions)} sessions in [{WINDOW_START}, {WINDOW_END}]",
          flush=True)

    all_windows: list[np.ndarray] = []  # each (WIN, n_pcs)
    per_session_log = []
    sessions_used = []

    for sess in sessions:
        try:
            sl, dn = load_paired_world(rat, sess)
        except Exception as e:
            per_session_log.append({"session": sess, "skipped": f"load: {e}"})
            continue
        if len(sl) < MIN_FRAMES:
            per_session_log.append({"session": sess, "skipped": "few_frames"})
            continue

        idx = calibration_indices(len(sl), 5.0, SLEAP_HZ, 1000, seed=0)
        if len(idx) < 100:
            per_session_log.append({"session": sess, "skipped": "no_cal_window"})
            continue
        tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
        if tx["residual"] > MAX_RESIDUAL_MM:
            per_session_log.append({
                "session": sess,
                "skipped": f"residual_{tx['residual']:.1f}",
            })
            continue

        sl_pc = project_sleap_to_template_pcs(sl, tmpl)
        sl_matches = run_template_matching(
            sl_pc, template_pc, flat_bounds_for_matching,
            max_outside=3, refractory_frames=WIN,
        )

        dn_pc = project_dannce_to_template_pcs(dn, tmpl)
        kept = 0
        for f in sl_matches:
            start = f - WIN + 1
            if start < 0 or f + 1 > len(dn_pc):
                continue
            window = dn_pc[start:f + 1]
            if window.shape[0] != WIN:
                continue
            all_windows.append(window)
            kept += 1

        sessions_used.append(sess)
        per_session_log.append({
            "session": sess,
            "n_sleap_matches": int(len(sl_matches)),
            "n_dannce_windows_kept": int(kept),
            "procrustes_residual_mm": float(tx["residual"]),
        })

    if not all_windows:
        raise RuntimeError(f"{rat}/{template_file}: no usable windows accumulated")

    stack = np.stack(all_windows, axis=0)               # (N, WIN, n_pcs)
    rebuilt_pc = stack.mean(axis=0)                     # (WIN, n_pcs)
    within_std_pc = stack.std(axis=0, ddof=1)           # (WIN, n_pcs)

    # Stored flat bound width per PC, for calibration & floor.
    stored_width_pc = bounds_scalar * feature_stds[pcu]  # (n_pcs,)

    if CALIBRATE_TO_STORED_MEAN:
        # Per-PC SIGMA_K so the time-mean of (sigma_k * within_std) equals
        # stored_width_pc. This preserves average corridor width while
        # letting per-frame std reshape it (tight on reliable frames, loose
        # on variable ones).
        mean_within_pc = within_std_pc.mean(axis=0)
        mean_within_pc = np.maximum(mean_within_pc, 1e-6)
        sigma_k_pc = stored_width_pc / mean_within_pc   # (n_pcs,)
    else:
        sigma_k_pc = np.full(stored_width_pc.shape, SIGMA_K_FALLBACK)

    floor_pc = FLOOR_FRAC * stored_width_pc             # (n_pcs,)
    bounds_pc = np.maximum(sigma_k_pc[None, :] * within_std_pc,
                           floor_pc[None, :])           # (WIN, n_pcs)

    # The output template uses the original full-rank shape. Fill the
    # non-pcs_to_use columns from the stored template so downstream code that
    # iterates the full PCA basis still finds something sensible there.
    new_template = tmpl["template"].copy().astype(np.float64)
    new_template[:, pcu] = rebuilt_pc

    # Same idea for the bounds array. Off-axis PCs use a flat fallback
    # consistent with the scalar bounds field.
    pc_template_bounds = np.tile(feature_stds * bounds_scalar, (WIN, 1)).astype(np.float64)
    pc_template_bounds[:, pcu] = bounds_pc

    print(f"  rebuilt: N={len(stack)} windows across {len(sessions_used)} sessions",
          flush=True)
    print(f"  per-PC mean within-motif std (PCs used): "
          f"{within_std_pc.mean(axis=0).tolist()}", flush=True)
    print(f"  per-PC sigma_k (PCs used): {sigma_k_pc.tolist()}", flush=True)
    print(f"  per-PC floor (PCs used): {floor_pc.tolist()}", flush=True)
    print(f"  per-PC mean bound (PCs used): "
          f"{bounds_pc.mean(axis=0).tolist()} "
          f"vs stored flat {stored_width_pc.tolist()}", flush=True)

    return {
        "new_template": new_template,
        "pc_template_bounds": pc_template_bounds,
        "within_std_pc": within_std_pc,
        "sigma_k_pc": sigma_k_pc,
        "floor_pc": floor_pc,
        "stored_width_pc": stored_width_pc,
        "n_windows": int(len(stack)),
        "sessions_used": sessions_used,
        "per_session_log": per_session_log,
        "tmpl_in": tmpl,
        "pcu": pcu.tolist(),
        "bounds_scalar": bounds_scalar,
    }


def save_template(rat, label, result):
    tmpl = result["tmpl_in"]
    out_path = OUT_DIR / f"{rat}_{label}.npz"
    np.savez(
        out_path,
        # Existing fields the rest of the codebase reads.
        template=result["new_template"],
        pc_weights=tmpl["pc_weights"],
        feature_means=tmpl["feature_means"],
        feature_stds=tmpl["feature_stds"],
        bounds=tmpl["bounds"],
        pcs_to_use=tmpl["pcs_to_use"],
        # Optional fields carried through if present in the stored template.
        temp_origin_file=tmpl.get("temp_origin_file", np.array("", dtype="U1")),
        temp_origin_idx=tmpl.get("temp_origin_idx", np.array(-1, dtype=np.int64)),
        # New fields the v2 matcher can use directly.
        pc_template_bounds=result["pc_template_bounds"],
        within_motif_std=result["within_std_pc"],
        sigma_k_pc=result["sigma_k_pc"],
        floor_pc=result["floor_pc"],
        stored_width_pc=result["stored_width_pc"],
        n_windows=np.array(result["n_windows"], dtype=np.int64),
        floor_frac=np.array(FLOOR_FRAC, dtype=np.float64),
        calibrated_to_stored_mean=np.array(CALIBRATE_TO_STORED_MEAN, dtype=bool),
    )
    print(f"  WROTE {out_path}", flush=True)
    return str(out_path)


def main():
    meta = {
        "window": [WINDOW_START, WINDOW_END],
        "max_residual_mm": MAX_RESIDUAL_MM,
        "calibrated_to_stored_mean": CALIBRATE_TO_STORED_MEAN,
        "floor_frac": FLOOR_FRAC,
        "templates": {},
    }

    for rat in RATS:
        for label, fn_for in TEMPLATE_FILES.items():
            template_file = fn_for(rat)
            print(f"\n=== {rat} / {template_file} ===", flush=True)
            try:
                result = rebuild_one_template(rat, template_file)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                meta["templates"][f"{rat}_{label}"] = {"error": str(e)}
                continue

            out_path = save_template(rat, label, result)
            meta["templates"][f"{rat}_{label}"] = {
                "source_template": template_file,
                "out_path": out_path,
                "n_windows": result["n_windows"],
                "sessions_used": result["sessions_used"],
                "pcs_to_use": result["pcu"],
                "bounds_scalar_used_for_matching": result["bounds_scalar"],
                "per_session_log": result["per_session_log"],
            }

    out_meta = OUT_DIR / "rebuild_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_meta}", flush=True)


if __name__ == "__main__":
    main()
