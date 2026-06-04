"""
World-space SLEAP → DANNCE alignment, two flavors:

  fit_procrustes(sl, dn)   → 7-DoF: rotation (3) + isotropic scale (1) + translation (3)
                              already implemented in qc_utils.find_sleap_dannce_alignment.
                              Returns dict with apply() callable mapping SLEAP→DANNCE.

  fit_affine(sl, dn)       → 12-DoF: 3x3 linear (which subsumes rotation,
                              anisotropic scale and shear) + translation (3).
                              Solved in closed form via least squares.

Both helpers expect the same input shapes: arrays of (N, 23, 3) for SLEAP and
(N, 23, 3) for DANNCE on matched frames. They flatten to (N*23, 3) for the fit
and return a dict with at minimum:
    apply(pts)        : (..., 3) → (..., 3) mapping SLEAP world coords to DANNCE world
    apply_inverse(pts): (..., 3) → (..., 3) mapping DANNCE world back to SLEAP world
    residual          : float, mean Euclidean error after alignment
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
from qc_utils import _try_alignment


# ---------------------------------------------------------------------------
# Procrustes (rigid + isotropic scale)  — wrap qc_utils._try_alignment
# ---------------------------------------------------------------------------

def fit_procrustes(sl: np.ndarray, dn: np.ndarray, try_z_flip: bool = True) -> dict:
    """7-DoF rigid + isotropic-scale fit. sl, dn shape (N, 23, 3)."""
    sl_flat = sl.reshape(-1, 3)
    dn_flat = dn.reshape(-1, 3)

    candidates = []
    R, s, t, res = _try_alignment(sl_flat, dn_flat)
    candidates.append(("nominal", R, s, t, res, False))
    if try_z_flip:
        sl_f = sl_flat.copy(); sl_f[:, 2] = -sl_f[:, 2]
        Rf, sf, tf, resf = _try_alignment(sl_f, dn_flat)
        candidates.append(("zflip", Rf, sf, tf, resf, True))
    name, R, s, t, res, z_flipped = min(candidates, key=lambda c: c[4])

    def apply(pts: np.ndarray) -> np.ndarray:
        shape = pts.shape
        flat = pts.reshape(-1, 3)
        if z_flipped:
            flat = flat.copy(); flat[:, 2] = -flat[:, 2]
        out = (flat @ (s * R).T) + t
        return out.reshape(shape)

    R_inv = R.T
    s_inv = 1.0 / s
    t_inv = -s_inv * (R_inv @ t)

    def apply_inverse(pts: np.ndarray) -> np.ndarray:
        shape = pts.shape
        flat = pts.reshape(-1, 3)
        out = (flat @ (s_inv * R_inv).T) + t_inv
        if z_flipped:
            out = out.copy(); out[:, 2] = -out[:, 2]
        return out.reshape(shape)

    return {
        "kind": "procrustes",
        "R": R, "s": float(s), "t": t, "z_flipped": bool(z_flipped),
        "residual": float(res),
        "apply": apply, "apply_inverse": apply_inverse,
    }


# ---------------------------------------------------------------------------
# Affine (12 DoF: 3x3 linear + 3-vec translation)
# ---------------------------------------------------------------------------

def fit_affine(sl: np.ndarray, dn: np.ndarray) -> dict:
    """12-DoF affine fit.

    Solves:  dn ≈ sl @ A.T + b   for A:(3,3) and b:(3,) via least squares.
    Equivalently, augmenting sl with a 1-column gives [sl|1] @ M.T = dn with M:(3,4).
    """
    sl_flat = sl.reshape(-1, 3)
    dn_flat = dn.reshape(-1, 3)
    N = sl_flat.shape[0]
    X = np.hstack([sl_flat, np.ones((N, 1))])           # (N, 4)
    M, _residuals, _rank, _sv = np.linalg.lstsq(X, dn_flat, rcond=None)  # (4, 3)
    A = M[:3, :].T                                       # (3, 3) linear
    b = M[3, :]                                          # (3,)   translation
    pred = sl_flat @ A.T + b
    residual = float(np.mean(np.linalg.norm(pred - dn_flat, axis=1)))

    A_inv = np.linalg.inv(A)
    b_inv = -A_inv @ b

    def apply(pts: np.ndarray) -> np.ndarray:
        shape = pts.shape
        flat = pts.reshape(-1, 3)
        return (flat @ A.T + b).reshape(shape)

    def apply_inverse(pts: np.ndarray) -> np.ndarray:
        shape = pts.shape
        flat = pts.reshape(-1, 3)
        return (flat @ A_inv.T + b_inv).reshape(shape)

    return {
        "kind": "affine",
        "A": A, "b": b, "residual": residual,
        "apply": apply, "apply_inverse": apply_inverse,
    }


# ---------------------------------------------------------------------------
# Calibration epoch helper
# ---------------------------------------------------------------------------

def calibration_indices(n_total: int, calibration_minutes: float = 5.0,
                        sleap_hz: float = 20.0,
                        n_sample: int = 1000, seed: int = 42):
    """Return ~n_sample indices drawn from the first calibration_minutes of frames.

    If calibration window is shorter than n_sample it returns all frames within
    the window.
    """
    cap = int(min(n_total, calibration_minutes * 60 * sleap_hz))
    if cap <= 0:
        return np.array([], dtype=int)
    if cap <= n_sample:
        return np.arange(cap, dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(cap, size=n_sample, replace=False)
