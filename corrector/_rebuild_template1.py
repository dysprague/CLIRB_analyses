"""Rebuild *_template_1.npz* from DANNCE keypoints.

Strategy (no stored origin_idx exists for template_1):
  1. Use the *stored* SLEAP-defined template_1 to find template-match end-frames
     on raw SLEAP across a handful of early pre-cutover sessions per rat.
  2. For each match end-frame f, grab the 30-frame DANNCE window [f-29, f]
     on the SLEAP-aligned timeline (load_paired_world already resamples).
  3. Normalize each DANNCE window via skeleton.normalize_skeleton_batch and
     project through the stored pc_weights/feature_means; slice to pcs_to_use.
  4. Average across windows → rebuilt template (30, len(pcs_to_use)).
  5. Save as <rat>_template_1_rebuild.npz alongside the original. Keep
     pc_weights / feature_means / feature_stds / bounds / pcs_to_use unchanged.

DANNCE is already in the template's z-convention (z positive), so no z-flip
is applied to DANNCE windows. SLEAP must be z-flipped before matching against
the stored template — same convention used everywhere else in this codebase.

Outputs:
  /home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data/<rat>/templates/
      <rat>_template_1_rebuild.npz
  corrector/results/template1_rebuild_meta.json   (records sessions, match counts)
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from data_io import load_aligned_data, load_template, template_path  # noqa: E402
from exp_utils import run_template_matching  # noqa: E402
from skeleton import normalize_skeleton_batch  # noqa: E402

from corrector.data_world import SLEAP_HZ, load_paired_world  # noqa: E402
from corrector.evaluate_all import DEFAULT_BOUNDS, WIN  # noqa: E402
from corrector.world_alignment import calibration_indices, fit_procrustes  # noqa: E402

TEMPLATE_DIR = Path("/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data")

# Candidate early sessions per rat (post-2025-10-25 calibration spin-up,
# Procrustes residual < 60). Walk in order; stop once we have enough matches.
CANDIDATE_SESSIONS = {
    "R1": [
        "2025_10_27_2", "2025_10_28_1", "2025_10_30_1", "2025_10_31_3",
        "2025_11_01_1", "2025_11_03_1", "2025_11_04_1", "2025_11_05_1",
        "2025_11_06_1", "2025_11_07_1",
    ],
    "R2": [
        "2025_10_28_1", "2025_10_28_2", "2025_10_29_1", "2025_10_29_2",
        "2025_10_30_1", "2025_10_30_2", "2025_10_31_1", "2025_10_31_2",
        "2025_11_01_1",
    ],
    "R3": [
        "2025_10_28_1", "2025_10_28_2", "2025_10_31_2", "2025_11_01_1",
        "2025_11_02_1", "2025_11_03_1", "2025_11_04_1", "2025_11_05_1",
        "2025_11_06_1",
    ],
}

# Target ≥ this many SLEAP matches per rat before stopping.
TARGET_MATCHES = 60
# Max sessions to scan per rat (safety cap).
MAX_SESSIONS = 8


def for_template_z_flip(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[:, :, 2] = -out[:, :, 2]
    return out


def project_sleap_to_template_pcs(sleap_xyz_world, tmpl):
    """SLEAP-world → z-flip → egocentric → flatten → (x-fm)@pw.T → slice pcu.
    Matches the matching pipeline."""
    pcu = tmpl["pcs_to_use"].ravel().astype(int)
    flipped = for_template_z_flip(sleap_xyz_world.astype(np.float64))
    rot, _, _ = normalize_skeleton_batch(flipped)
    flat = rot.reshape(len(rot), -1)
    pcs_full = (flat - tmpl["feature_means"]) @ tmpl["pc_weights"].T
    return pcs_full[:, pcu]


def project_dannce_to_template_pcs(dannce_xyz_world, tmpl):
    """DANNCE-world → egocentric (no z-flip; already in template-z convention)
    → flatten → (x-fm)@pw.T → slice pcu."""
    pcu = tmpl["pcs_to_use"].ravel().astype(int)
    rot, _, _ = normalize_skeleton_batch(dannce_xyz_world.astype(np.float64))
    flat = rot.reshape(len(rot), -1)
    pcs_full = (flat - tmpl["feature_means"]) @ tmpl["pc_weights"].T
    return pcs_full[:, pcu]


def rebuild_for_rat(rat, tmpl):
    pcu = tmpl["pcs_to_use"].ravel().astype(int)
    template_pc = tmpl["template"][:, pcu]
    feature_stds = tmpl["feature_stds"]
    bounds_scalar = DEFAULT_BOUNDS[rat]
    bounds = np.tile(feature_stds[pcu] * bounds_scalar, (WIN, 1))

    accumulated_dannce_pc_windows = []  # each: (WIN, len(pcu))
    sessions_used = []
    per_session_log = []

    for sess in CANDIDATE_SESSIONS[rat][:MAX_SESSIONS]:
        try:
            sl, dn = load_paired_world(rat, sess)
        except Exception as e:
            per_session_log.append({"rat": rat, "session": sess,
                                     "error": f"load: {e}"})
            continue
        if len(sl) < 1000:
            per_session_log.append({"rat": rat, "session": sess,
                                     "error": "too few frames"})
            continue

        # Sanity-check Procrustes — same threshold as evaluate_all.
        idx = calibration_indices(len(sl), 5.0, SLEAP_HZ, 1000, seed=0)
        if len(idx) < 100:
            per_session_log.append({"rat": rat, "session": sess,
                                     "error": "no cal window"})
            continue
        tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
        if tx["residual"] > 60.0:
            per_session_log.append({"rat": rat, "session": sess,
                                     "error": f"residual {tx['residual']:.1f}"})
            continue

        # Match on raw SLEAP (z-flipped, normalized, projected).
        sl_pc = project_sleap_to_template_pcs(sl, tmpl)
        sl_matches = run_template_matching(sl_pc, template_pc, bounds,
                                            max_outside=3, refractory_frames=WIN)

        # For each SLEAP match end-frame, grab the corresponding DANNCE window
        # and project to template PC space.
        dn_pc_for_template = project_dannce_to_template_pcs(dn, tmpl)

        good_windows = 0
        for f in sl_matches:
            start = f - WIN + 1
            if start < 0 or f + 1 > len(dn_pc_for_template):
                continue
            window = dn_pc_for_template[start:f + 1]  # (WIN, len(pcu))
            if window.shape[0] != WIN:
                continue
            accumulated_dannce_pc_windows.append(window)
            good_windows += 1

        sessions_used.append(sess)
        per_session_log.append({
            "rat": rat, "session": sess,
            "n_sleap_matches": int(len(sl_matches)),
            "n_dannce_windows_kept": int(good_windows),
            "procrustes_residual": float(tx["residual"]),
        })
        print(f"  {rat}/{sess}: SLEAP_matches={len(sl_matches)}  "
              f"DANNCE_windows_kept={good_windows}  "
              f"residual={tx['residual']:.1f}", flush=True)

        if len(accumulated_dannce_pc_windows) >= TARGET_MATCHES:
            break

    if not accumulated_dannce_pc_windows:
        raise RuntimeError(f"{rat}: no usable DANNCE windows accumulated")

    stack = np.stack(accumulated_dannce_pc_windows, axis=0)  # (N, WIN, n_pcs)
    rebuilt_template = stack.mean(axis=0)                     # (WIN, n_pcs)
    print(f"  {rat}: rebuilt template from N={len(stack)} windows across "
          f"{len(sessions_used)} sessions", flush=True)

    return rebuilt_template, sessions_used, per_session_log, len(stack)


def main():
    meta = {}
    for rat in ["R1", "R2", "R3"]:
        tmpl = dict(load_template(rat, f"{rat}_template_1.npz"))
        pcu = tmpl["pcs_to_use"].ravel().astype(int)
        print(f"\n=== {rat} ===")
        print(f"  stored template shape: {tmpl['template'].shape}  "
              f"pcs_to_use={pcu.tolist()}  bounds={float(tmpl['bounds']):.3f}")

        rebuilt_pc, sessions_used, log, n_windows = rebuild_for_rat(rat, tmpl)

        # Build the new template: keep all other fields the same, swap only the
        # `template` field. Match shape exactly: stored template_1 is (30, 2).
        new_template = tmpl["template"].copy().astype(np.float64)
        new_template[:, pcu] = rebuilt_pc

        out_path = TEMPLATE_DIR / rat / "templates" / f"{rat}_template_1_rebuild.npz"
        if out_path.exists():
            raise FileExistsError(f"Refusing to overwrite {out_path}")
        np.savez(out_path,
                 template=new_template,
                 pc_weights=tmpl["pc_weights"],
                 feature_means=tmpl["feature_means"],
                 feature_stds=tmpl["feature_stds"],
                 bounds=tmpl["bounds"],
                 pcs_to_use=tmpl["pcs_to_use"])
        print(f"  WROTE {out_path}")

        # Compare to stored
        stored_pc = tmpl["template"][:, pcu]
        diff = np.linalg.norm(rebuilt_pc - stored_pc)
        print(f"  L2(rebuilt - stored) on pcs_to_use: {diff:.3f}")
        meta[rat] = {
            "out_path": str(out_path),
            "sessions_used": sessions_used,
            "n_windows": int(n_windows),
            "l2_vs_stored": float(diff),
            "per_session_log": log,
            "stored_template_pcu": stored_pc.tolist(),
            "rebuilt_template_pcu": rebuilt_pc.tolist(),
        }

    out_meta = REPO / "corrector/results/template1_rebuild_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_meta}")


if __name__ == "__main__":
    main()
