"""DANNCE referee: an independent, spatially unbiased view of the behavior.

DANNCE runs on 6 cameras, SLEAP on 3, so DANNCE is the stronger reconstruction.
Both are projected through the SAME pc_weights / feature_means (the space the
online rule operates in), but the REFERENCE TEMPLATE for measuring behavior
quality is built from a SPATIALLY STRATIFIED sample of DANNCE windows.

This is the key methodological point. multi_session's analyse_dual_session
rebuilds both templates from SLEAP-selected matches, which imports exactly the
spatial bias under test. Nothing here is selected using SLEAP match outcome.
"""
import numpy as np

import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from clirb import events as ev  # noqa: E402

W = ev.TEMPLATE_LEN
NC = ev.N_COMPONENTS


def dannce_aligned(raw):
    """(T_sleap, 23, 3) DANNCE keypoints on the SLEAP frame grid, or None."""
    if not raw.get('has_dual', False):
        return None
    d = raw['dannce_keys_3D']
    if d.ndim == 4:
        d = d.squeeze(axis=1).transpose(0, 2, 1)
    else:
        d = np.transpose(d, [0, 2, 1])
    idx = np.asarray(raw['dannce_idx_for_sleap_cams'], dtype=int).ravel()[1:]
    if idx.size == 0:
        return None
    idx = np.clip(idx, 0, len(d) - 1)
    return d[idx]


def project_both(raw, template):
    """(sleap_pcs, dannce_pcs) in the template's PC space, on the SLEAP grid.
    SLEAP is z-flipped; DANNCE is NOT."""
    s_pcs, _ = ev.project_session_to_pcs(raw['triang_3d'], template, flip_z=True)
    d3 = dannce_aligned(raw)
    if d3 is None:
        return s_pcs, None
    n = min(len(s_pcs), len(d3))
    d_pcs, _ = ev.project_session_to_pcs(d3[:n], template, flip_z=False)
    return s_pcs[:n], d_pcs


def windows_at(pcs, ends, w=W, ncomp=NC):
    """(n, w, ncomp) stack of windows ending at `ends` (inclusive)."""
    ends = np.asarray(ends, dtype=int)
    ok = (ends >= w - 1) & (ends < len(pcs))
    e = ends[ok]
    idx = (e - (w - 1))[:, None] + np.arange(w)[None, :]
    return pcs[idx, :ncomp], ok


def build_reference(pcs_list, ends_list, sector_list, n_per_sector=40,
                    n_iter=2, rng=None, w=W, ncomp=NC):
    """Spatially stratified reference template.

    pcs_list/ends_list/sector_list: per-session DANNCE PCs, candidate window
    end-frames, and the arena sector of each candidate.

    Draws up to n_per_sector windows PER SECTOR (pooled across sessions), then
    runs a 2-step align->average refinement. Returns (template (w,ncomp),
    bounds (w,ncomp) = max_dev).
    """
    rng = rng or np.random.default_rng(0)
    wins, secs = [], []
    for pcs, ends, sec in zip(pcs_list, ends_list, sector_list):
        ww, ok = windows_at(pcs, ends, w, ncomp)
        if len(ww):
            wins.append(ww); secs.append(np.asarray(sec)[ok])
    if not wins:
        return None, None
    X = np.concatenate(wins); S = np.concatenate(secs)
    keep = []
    for s in np.unique(S):
        ii = np.where(S == s)[0]
        keep.extend(rng.choice(ii, size=min(len(ii), n_per_sector), replace=False))
    keep = np.array(keep)
    Xs = X[keep]
    tmpl = Xs.mean(axis=0)
    # refine: re-select, per candidate, the best-aligned window near its end
    for _ in range(n_iter - 1):
        mse = ((Xs - tmpl[None]) ** 2).mean(axis=(1, 2))
        good = Xs[mse <= np.percentile(mse, 80)]
        if len(good) >= 10:
            tmpl = good.mean(axis=0)
    lo, hi = Xs.min(axis=0), Xs.max(axis=0)
    bounds = np.maximum(tmpl - lo, hi - tmpl)
    return tmpl, bounds


def align_distance(pcs, start, end, template, buf=ev.ALIGN_BUF_FR,
                   w=W, ncomp=NC):
    """min-MSE of any w-frame window ending in [start, end+buf] to `template`.
    Returns (best_end, mse)."""
    e_min = max(start, w - 1)
    e_max = min(end + buf, len(pcs) - 1)
    if e_max < e_min:
        return -1, np.nan
    ends = np.arange(e_min, e_max + 1)
    idx = (ends - (w - 1))[:, None] + np.arange(w)[None, :]
    win = pcs[idx, :ncomp]
    mse = ((win - template[None]) ** 2).mean(axis=(1, 2))
    k = int(np.argmin(mse))
    return int(ends[k]), float(mse[k])


def residual_snippets(s_pcs, d_pcs, ends, w=W, ncomp=NC):
    """(n, w, ncomp) SLEAP-minus-DANNCE residual windows ending at `ends`."""
    sw, ok = windows_at(s_pcs, ends, w, ncomp)
    dw, _ = windows_at(d_pcs, ends, w, ncomp)
    n = min(len(sw), len(dw))
    return sw[:n] - dw[:n], ok
