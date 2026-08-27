"""Rearing-manifold density plots on a BASELINE-fit embedding (attempts_umap pipeline).

Variant of rearing_manifold_umap.py. Same verbatim attempts_umap.py pipeline and
30-frame PC-window feature, but the 2D embedding is fit on the BASELINE natural
repertoire (non-overlapping tiles) rather than the attempt bank; the trained
matched rears are projected into that fixed baseline space for the density plot.

TWO EMBEDDINGS from the same 60-dim feature (30-frame windows of the 2 shared
template PCs, z-scored):
  * PC   : sklearn PCA(2)
  * UMAP : umap.UMAP(2)  (n_neighbors / min_dist from attempts_umap CONFIG)

WHAT FITS THE EMBEDDING
  The FULL set of NON-OVERLAPPING 30-frame windows tiled across the 3 baseline
  natural sessions (windows [0,W), [W,2W), ...). Each session is projected through
  the shared template PC space (attempts_umap project_session: rotate+flip ->
  causal median -> project). The z-scored tile windows fit PCA and UMAP; the
  z-score + embedding are then held fixed and every condition is .transform()ed in.

WHAT IS PLOTTED (projected into the fitted baseline embedding via .transform())
  * baseline (full natural repertoire): the baseline TILES themselves (the fit set).
  * early  (template_num==1 sessions) : only MATCHED rears.
  * late   (template_num==5 sessions) : only MATCHED rears.
  A rear is detected on PC0 (hysteresis 200/150, >0.5 s); its window is the
  30-frame best-aligned (min-MSE, PC space) window inside [rear_start,
  rear_end+0.75 s]. A rear is MATCHED iff that window passes the session's own
  template BASE band (attempts_umap's is_match).

For each embedding a figure mirrors rearing_manifold_plot.py: grey repertoire +
blue early->late ramp, 50/90 % KDE HDR contours, k-NN overlap in the legend.

USAGE  .../envs/clirb_analysis/bin/python rearing_manifold_umap_basefit.py --output_dir out/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # headless; must precede attempts_umap's pyplot import

CLIRB_DIR = "/home/yutaka-sprague/CLIRB"
sys.path.insert(0, CLIRB_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- attempts_umap.py machinery, imported VERBATIM ----
import attempts_umap as au  # noqa: E402
from attempts_umap import (  # noqa: E402
    project_session, find_matches, window_oob, build_template_bank,
    find_template_file, read_template_num, discover_sessions,
    DATA_ROOT, WIDEN_FACTOR, WIDEN_MATCH_METHOD, ACCEPTED_ERRORS,
    WIDEN_JITTER, COUNT_REFRACTORY,
)

RAT = "R5"
FPS = 20.0

# attempts_umap uses a MODULE-GLOBAL `rat` inside build_template_bank /
# discover_sessions (its on-disk copy is left set to another rat). Point it at
# ours so those verbatim functions build the right paths.
au.rat = RAT

# conditions
BASELINE_SESSIONS = ["2026_06_05_1", "2026_06_08_1", "2026_06_09_1"]
EARLY_TNUM, LATE_TNUM = 1, 5

# rear detector (single/multi_session hysteresis, on the shared PC0)
REAR_RISE, REAR_RESET, MIN_REAR_S = 200.0, 150.0, 0.5
ALIGN_BUFFER_S = 0.75

# UMAP params (attempts_umap CONFIG)
UMAP_NEIGHBORS = au.UMAP_NEIGHBORS
UMAP_MIN_DIST = au.UMAP_MIN_DIST

# ----- analysis constants (match rearing_manifold_plot.py / fig5_09) -----
GRID_N = 240
GRID_PAD = 0.14
PCTL = (0.3, 99.7)
BW = 0.30
N_DENS = 40000
K_NN, N_KNN, CAP_PCT = 5, 10000, 95
HDR_LEVELS = ((0.50, 1.8), (0.90, 1.1))
RGB_FULL = (0.80, 0.80, 0.82)
CLINE_FULL = (0.34, 0.34, 0.37)
RGB_REARS = (0.55, 0.78, 0.55)     # all-natural-rears bubble fill (green)
CLINE_REARS = (0.13, 0.55, 0.13)   # all-natural-rears contour (green)


# --------------------------------------------------------------------------- #
# load a session's shared-PC-space trajectory (attempts_umap pipeline verbatim)
# --------------------------------------------------------------------------- #
def session_pc(sess, fmean_ref, pcw_ref, pcs_ref):
    """SLEAP triang keys -> shared-PC trajectory (n, n_pcs), or None if missing.
    Uses attempts_umap.project_session (rotate+flip -> causal median -> project)
    and the same trailing-zero trim."""
    sdir = f"{DATA_ROOT}/{RAT}/{sess}/sleap"
    p = os.path.join(sdir, "triang_keys_3D.npy")
    if not os.path.isfile(p):
        return None
    keys = np.load(p)
    nz = np.any(keys != 0, axis=(1, 2))
    keys = keys[:(np.max(np.where(nz)[0]) + 1)] if nz.any() else keys[:0]
    if keys.shape[0] == 0:
        return None
    return project_session(keys, fmean_ref, pcw_ref, pcs_ref)


# --------------------------------------------------------------------------- #
# attempts across ALL sessions -> embedding-fit feature matrix (verbatim logic)
# --------------------------------------------------------------------------- #
def collect_all_attempts(sessions, bank, pcw_ref, fmean_ref, pcs_ref, W):
    """Reproduces attempts_umap.py MAIN step 2: union of widened-band matches
    across the bank per session, deduped within W//2, nearest-template assigned.
    Returns X (n_att, W*n_pcs) of flattened PC windows."""
    feats = []
    for sess in sessions:
        pc = session_pc(sess, fmean_ref, pcw_ref, pcs_ref)
        if pc is None:
            continue
        allf = []
        for t in bank:
            allf.extend(int(e) for e in find_matches(
                pc, t["tused"], t["band"] * WIDEN_FACTOR, W, WIDEN_MATCH_METHOD,
                ACCEPTED_ERRORS, WIDEN_JITTER, COUNT_REFRACTORY))
        if not allf:
            continue
        uniq = []
        for ef in sorted(set(allf)):
            if not uniq or ef - uniq[-1] > W // 2:
                uniq.append(ef)
        for ef in uniq:
            if ef - W + 1 < 0:
                continue
            win = pc[ef - W + 1:ef + 1]                      # (W, n_pcs)
            att = np.array([window_oob(win, t["tused"], t["band"] * WIDEN_FACTOR)
                            <= ACCEPTED_ERRORS for t in bank])
            if not att.any():
                continue                                     # attempts_umap keeps only attempts (with nearest fallback; here require an attempt)
            feats.append(win.reshape(-1))
    return np.stack(feats) if feats else np.zeros((0, W * len(pcs_ref)))


def baseline_tile_feats(sessions, pcw_ref, fmean_ref, pcs_ref, W):
    """Non-overlapping W-frame tiles over each baseline session -> (n_tiles,
    W*n_pcs). Windows end at frames W-1, 2W-1, ... (tiles [0,W), [W,2W), ...).
    This is the full natural repertoire the embedding is fit on."""
    feats, per = [], {}
    for sess in sessions:
        pc = session_pc(sess, fmean_ref, pcw_ref, pcs_ref)
        if pc is None:
            per[sess] = 0; continue
        n = pc.shape[0]
        n_tiles = n // W
        if n_tiles == 0:
            per[sess] = 0; continue
        ends = np.arange(1, n_tiles + 1) * W - 1            # last frame of each tile
        starts = ends - W + 1
        idx = starts[:, None] + np.arange(W)[None, :]
        win = pc[idx]                                       # (n_tiles, W, n_pcs)
        feats.append(win.reshape(win.shape[0], -1))
        per[sess] = n_tiles
    X = np.concatenate(feats, 0) if feats else np.zeros((0, W * len(pcs_ref)))
    return X, per


# --------------------------------------------------------------------------- #
# rears -> best-aligned W-window;  matched = passes session base band
# --------------------------------------------------------------------------- #
def detect_rears(pc0):
    rise, reset, min_above = REAR_RISE, REAR_RESET, MIN_REAR_S * FPS
    rears, state, start, n_above = [], "down", None, 0
    for i in range(len(pc0)):
        v = pc0[i]
        if state == "down":
            if v > rise:
                state, start, n_above = "up", i, 1
        else:
            if v > rise:
                n_above += 1
            if v < reset:
                rears.append((start, i, n_above))
                state, start, n_above = "down", None, 0
    if state == "up":
        rears.append((start, len(pc0) - 1, n_above))
    return [(s, e) for (s, e, na) in rears if na > min_above]


def best_align_end(rs, re, pc, tused, W, buf):
    """End-frame in [max(rs,W-1), re+buf] whose W-window has lowest MSE to the
    template PCs (== single_session _best_align_end, in the shared PC space)."""
    e_min = max(rs, W - 1)
    e_max = min(re + buf, pc.shape[0] - 1)
    if e_max < e_min:
        return None
    ends = np.arange(e_min, e_max + 1)
    starts = ends - (W - 1)
    idx = starts[:, None] + np.arange(W)[None, :]
    win = pc[idx]
    mse = ((win - tused[None]) ** 2).mean(axis=(1, 2))
    return int(ends[int(np.argmin(mse))])


def rear_windows(sess, bank_tpl, pcw_ref, fmean_ref, pcs_ref, W, matched_only):
    """Best-aligned W-frame PC windows for this session's rears.

    bank_tpl = (tused, band) for THIS session's own template (base band decides
    'matched'). If matched_only, keep only rears whose window passes the base
    band (attempts_umap is_match). Returns (feats (n,W*n_pcs), n_rears, n_kept)."""
    tused, band = bank_tpl
    pc = session_pc(sess, fmean_ref, pcw_ref, pcs_ref)
    if pc is None:
        return np.zeros((0, W * len(pcs_ref))), 0, 0
    rears = detect_rears(pc[:, 0])
    buf = int(round(ALIGN_BUFFER_S * FPS))
    feats = []
    for rs, re in rears:
        ef = best_align_end(rs, re, pc, tused, W, buf)
        if ef is None or ef - W + 1 < 0:
            continue
        win = pc[ef - W + 1:ef + 1]                          # (W, n_pcs)
        if matched_only and window_oob(win, tused, band) > ACCEPTED_ERRORS:
            continue                                         # not a base-band match
        feats.append(win.reshape(-1))
    X = np.stack(feats) if feats else np.zeros((0, W * len(pcs_ref)))
    return X, len(rears), X.shape[0]


def session_own_template(sess):
    """(tused, band) for a session's own template file, or None."""
    sdir = f"{DATA_ROOT}/{RAT}/{sess}/sleap"
    try:
        tpl_path = find_template_file(sdir)
    except FileNotFoundError:
        return None
    pcw, fmean, pcs, tused, band = au.load_template(tpl_path)
    return tused, band


# --------------------------------------------------------------------------- #
# density (KDE) + overlap (k-NN)  — copied from rearing_manifold_plot.py
# --------------------------------------------------------------------------- #
def kde_grid(scores, grid_xy, n_grid, dA, bw=BW):
    from scipy.stats import gaussian_kde
    Z = gaussian_kde(scores.T, bw_method=bw)(grid_xy).reshape(n_grid, n_grid)
    return Z / (Z.sum() * dA)


def hdr_level(Z, frac, dA):
    v = np.sort(Z.ravel())[::-1]
    c = np.cumsum(v) * dA
    return float(v[min(np.searchsorted(c / c[-1], frac), len(v) - 1)])


def _subsample(x, n, rng):
    return x if x.shape[0] <= n else x[rng.choice(x.shape[0], n, replace=False)]


def overlap_knn(policy, ref, rng, k=K_NN, cap_pct=CAP_PCT, n_sub=N_KNN, batch=2000):
    from scipy.spatial import cKDTree
    if policy.shape[0] == 0 or ref.shape[0] < k + 1:
        return float("nan")
    ref = _subsample(ref, n_sub, rng)
    policy = _subsample(policy, n_sub, rng)
    r = cKDTree(ref).query(ref, k=k + 1)[0][:, k]
    r = np.minimum(r, np.percentile(r, cap_pct))
    inside = np.zeros(policy.shape[0], dtype=bool)
    for s in range(0, policy.shape[0], batch):
        d = np.linalg.norm(policy[s:s + batch, None, :] - ref[None, :, :], axis=2)
        inside[s:s + batch] = (d <= r[None, :]).any(axis=1)
    return float(inside.mean())


# --------------------------------------------------------------------------- #
# figure (style matched to rearing_manifold_plot.py)
# --------------------------------------------------------------------------- #
def _weight(d, vmax, gamma, amax):
    return (amax * np.clip(d / vmax, 0.0, 1.0) ** gamma)[..., None]


def _composite(layers, vmax, gamma=0.55, amax=0.80):
    H, W = layers[0][0].shape
    out = np.ones((H, W, 3), np.float32)
    for d, rgb in layers:
        a = _weight(d, vmax, gamma, amax)
        out = out * (1.0 - a * (1.0 - np.asarray(rgb, np.float32)))
    return out


def _dim(rgb, f=0.35):
    return tuple(c + (1.0 - c) * f for c in rgb)


def blue_ramp(n):
    import matplotlib as mpl
    blues = mpl.colormaps["Blues"]
    fills = [blues(x)[:3] for x in np.linspace(0.42, 0.86, n)]
    lines = [blues(x)[:3] for x in np.linspace(0.66, 1.0, n)]
    return list(zip(fills, lines))


def make_figure(out_path, gx, gy, dens, labels, ov_full, ramp, embed_name, ev=None,
                extra=None):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
    from matplotlib.ticker import MaxNLocator

    try:
        mpl.style.use("seaborn-v0_8-deep")
    except OSError:
        pass
    mpl.rcParams.update({"figure.facecolor": "w", "savefig.dpi": 600,
                         "savefig.transparent": True, "font.size": 8,
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    dA = (gx[1] - gx[0]) * (gy[1] - gy[0])
    extent = [gx.min(), gx.max(), gy.min(), gy.max()]
    cgx = np.linspace(extent[0], extent[1], GRID_N)
    cgy = np.linspace(extent[2], extent[3], GRID_N)
    vmax = max(Z.max() for Z in dens.values())

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    comp = [(dens["full"], RGB_FULL)]
    if extra:
        comp.append((dens[extra["key"]], extra["fill"]))
    comp += [(dens[l], ramp[i][0]) for i, l in enumerate(labels)]
    ax.imshow(_composite(comp, vmax), extent=extent, origin="lower", aspect="auto",
              interpolation="bilinear", zorder=0)
    for frac, lw in HDR_LEVELS:
        cs = ax.contour(cgx, cgy, dens["full"], levels=[hdr_level(dens["full"], frac, dA)],
                        colors=[CLINE_FULL], linewidths=lw, zorder=3)
        t = ax.clabel(cs, inline=True, inline_spacing=1, fontsize=6,
                      fmt=lambda _l, f=frac: f"{f:.0%}")
        for tt in t:
            tt.set_zorder(6)
            tt.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground="white")])
    if extra:                                        # natural-rears bubble (green)
        for frac, lw in HDR_LEVELS:
            ax.contour(cgx, cgy, dens[extra["key"]],
                       levels=[hdr_level(dens[extra["key"]], frac, dA)],
                       colors=[extra["line"]], linewidths=lw, zorder=2)
    for i, lab in enumerate(labels):
        for frac, lw in HDR_LEVELS:
            ax.contour(cgx, cgy, dens[lab], levels=[hdr_level(dens[lab], frac, dA)],
                       colors=[ramp[i][1]], linewidths=lw, zorder=2)

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    if embed_name == "PC" and ev is not None:
        ax.set_xlabel(f"Attempt-PC 1 ({ev[0]*100:.0f}%)", fontsize=8, fontweight="bold", labelpad=1.5)
        ax.set_ylabel(f"Attempt-PC 2 ({ev[1]*100:.0f}%)", fontsize=8, fontweight="bold", labelpad=1.5)
    else:
        ax.set_xlabel(f"{embed_name}-1", fontsize=8, fontweight="bold", labelpad=1.5)
        ax.set_ylabel(f"{embed_name}-2", fontsize=8, fontweight="bold", labelpad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=8, length=2.5, width=0.8, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8); ax.spines[sp].set_position(("outward", 3))

    rows = [TextArea("Full natural\nrepertoire (tiles)",
                     textprops=dict(color=CLINE_FULL, size=8, linespacing=1.15))]
    if extra:
        rows.append(TextArea(extra["legend"],
                             textprops=dict(color=extra["line"], size=8, linespacing=1.15)))
    for i, lab in enumerate(labels):
        col = ramp[i][1]
        rows.append(TextArea(f"{lab} (matched rears)",
                             textprops=dict(color=col, size=8, linespacing=1.15)))
        rows.append(TextArea(f"{ov_full[lab]:.0%} in repertoire",
                             textprops=dict(color=_dim(col), size=6, linespacing=1.2)))
    ax.add_artist(AnchoredOffsetbox(loc="upper left",
                                    child=VPacker(children=rows, pad=0, sep=1.5, align="left"),
                                    frameon=False, borderpad=0.2, bbox_to_anchor=(1.02, 1.0),
                                    bbox_transform=ax.transAxes))
    fig.tight_layout(pad=0.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_path}.{ext}", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# density + figure for one embedding's scores
# --------------------------------------------------------------------------- #
def build_and_plot(scores, labels, out_path, embed_name, rng, ev=None, extra=None):
    """extra (optional): {"key","color","legend"} -> an additional KDE layer
    (contour+fill) drawn in its own color with a legend row and NO overlap %.
    Its scores must already be in `scores[extra["key"]]`."""
    grid_keys = ["full"] + ([extra["key"]] if extra else []) + labels
    allp = np.concatenate([scores[k] for k in grid_keys])
    lo = np.percentile(allp, PCTL[0], axis=0)
    hi = np.percentile(allp, PCTL[1], axis=0)
    pad = GRID_PAD * (hi - lo)
    lo, hi = lo - pad, hi + pad
    gx = np.linspace(lo[0], hi[0], GRID_N)
    gy = np.linspace(lo[1], hi[1], GRID_N)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.vstack([GX.ravel(), GY.ravel()])
    dA = (gx[1] - gx[0]) * (gy[1] - gy[0])
    cap = lambda s: _subsample(s, N_DENS, rng)
    dens = {k: kde_grid(cap(scores[k]), grid_xy, GRID_N, dA) for k in grid_keys}
    ov = {lab: overlap_knn(scores[lab], scores["full"], rng) for lab in labels}
    for lab in labels:
        print(f"[{embed_name} overlap] {lab}: repertoire={ov[lab]:.3f}", flush=True)
    make_figure(out_path, gx, gy, dens, labels, ov, blue_ramp(len(labels)), embed_name,
                ev, extra)
    return ov


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default="results/rearing_manifold_umap_basefit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. template bank + shared PC space over ALL sessions (verbatim) ----
    all_sessions = discover_sessions(RAT)
    bank, pcw_ref, fmean_ref, pcs_ref, W = build_template_bank(all_sessions)
    if not bank:
        raise SystemExit("no templates -> no bank")
    print(f"{RAT}: {len(all_sessions)} sessions; bank={len(bank)} templates; W={W} "
          f"n_pcs={len(pcs_ref)}", flush=True)

    # ---- 2. embedding-fit features = baseline non-overlapping W-tiles --------
    X_base, base_per = baseline_tile_feats(BASELINE_SESSIONS, pcw_ref, fmean_ref, pcs_ref, W)
    print(f"[embed-fit] baseline tiles: {X_base.shape[0]} (dim {X_base.shape[1]})  "
          f"per-session={base_per}", flush=True)
    if X_base.shape[0] < 3:
        raise SystemExit("too few baseline tiles to fit an embedding")

    # z-score exactly as attempts_umap (per-feature; guard zero-std) -- stats
    # from the fit set (baseline tiles), reused to project every condition.
    mu, sd = X_base.mean(0), X_base.std(0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    zscore = lambda M: (M - mu) / sd_safe
    Xz = zscore(X_base)

    # ---- 3. fit BOTH embeddings on the baseline tiles ----------------------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=args.seed).fit(Xz)
    ev = pca.explained_variance_ratio_
    print(f"[PC] fit on {Xz.shape[0]} baseline tiles; PC1={ev[0]*100:.1f}% PC2={ev[1]*100:.1f}%",
          flush=True)
    import umap
    reducer = umap.UMAP(n_neighbors=min(UMAP_NEIGHBORS, max(2, Xz.shape[0] - 1)),
                        min_dist=UMAP_MIN_DIST, n_components=2, random_state=args.seed)
    umap_emb_fit = reducer.fit_transform(Xz)
    print(f"[UMAP] fit on {Xz.shape[0]} baseline tiles", flush=True)

    # ---- 4. KDE-set windows: baseline TILES (full) + early/late matched rears -
    early_sessions = [s for s in all_sessions if read_template_num(
        f"{DATA_ROOT}/{RAT}/{s}/sleap") == EARLY_TNUM]
    late_sessions = [s for s in all_sessions if read_template_num(
        f"{DATA_ROOT}/{RAT}/{s}/sleap") == LATE_TNUM]
    print(f"early(template {EARLY_TNUM})={len(early_sessions)} "
          f"late(template {LATE_TNUM})={len(late_sessions)} baseline={len(BASELINE_SESSIONS)}",
          flush=True)

    # baseline natural sessions carry NO sleap-dir template; their rears are
    # aligned to the EARLY template (R5_template_1), as in rearing_manifold_umap.
    early_tpl = au.load_template(os.path.join(
        DATA_ROOT, RAT, "templates", f"{RAT}_template_{EARLY_TNUM}.npz"))
    early_align = (early_tpl[3], early_tpl[4])   # (tused, band)

    def condition_feats(sessions, matched_only, fallback_tpl=None):
        feats, nre, nkeep = [], 0, 0
        for s in sessions:
            tpl = session_own_template(s) or fallback_tpl
            if tpl is None:
                print(f"  {s}: no template file -> skipped", flush=True); continue
            X, nr, nk = rear_windows(s, tpl, pcw_ref, fmean_ref, pcs_ref, W, matched_only)
            feats.append(X); nre += nr; nkeep += nk
            print(f"  [{s}] rears={nr} kept={nk}"
                  f"{' (matched)' if matched_only else ' (all)'}", flush=True)
        X = np.concatenate(feats, 0) if feats else np.zeros((0, W * len(pcs_ref)))
        return X, nre, nkeep

    # grey repertoire = the baseline TILES (the embedding-fit set itself)
    Xfull = X_base
    # extra green bubble = ALL rears in the baseline natural sessions
    print("[rears] all natural rears (baseline sessions):", flush=True)
    Xrears, nrr, nrk = condition_feats(BASELINE_SESSIONS, matched_only=False,
                                       fallback_tpl=early_align)
    print("[early] matched rears:", flush=True)
    Xearly, ner, nek = condition_feats(early_sessions, matched_only=True)
    print("[late] matched rears:", flush=True)
    Xlate, nlr, nlk = condition_feats(late_sessions, matched_only=True)
    print(f"KDE-set windows: full(tiles)={Xfull.shape[0]} natural_rears={Xrears.shape[0]} "
          f"early={Xearly.shape[0]} late={Xlate.shape[0]}", flush=True)

    cond_X = {"full": Xfull, "natural_rears": Xrears, "early": Xearly, "late": Xlate}
    labels = [l for l in ("early", "late") if cond_X[l].shape[0] > 0]
    extra = ({"key": "natural_rears", "fill": RGB_REARS, "line": CLINE_REARS,
              "legend": "all natural rears"} if Xrears.shape[0] > 0 else None)

    # ---- 5. project KDE-set into BOTH fitted embeddings, plot ---------------
    # PC embedding
    pc_scores = {k: pca.transform(zscore(v)) for k, v in cond_X.items() if v.shape[0] > 0}
    ov_pc = build_and_plot(pc_scores, labels, os.path.join(args.output_dir, "rearing_manifold_PC"),
                           "PC", rng, ev=ev, extra=extra)
    # UMAP embedding (transform out-of-sample)
    um_scores = {k: reducer.transform(zscore(v)) for k, v in cond_X.items() if v.shape[0] > 0}
    ov_um = build_and_plot(um_scores, labels, os.path.join(args.output_dir, "rearing_manifold_UMAP"),
                           "UMAP", rng, extra=extra)

    json.dump({
        "rat": RAT,
        "n_sessions_total": len(all_sessions),
        "bank_size": len(bank), "W": int(W), "n_pcs": int(len(pcs_ref)),
        "embed_fit": "baseline non-overlapping W-frame tiles (3 natural sessions)",
        "n_tiles_fit": int(X_base.shape[0]),
        "baseline_tiles_per_session": {k: int(v) for k, v in base_per.items()},
        "baseline_sessions": BASELINE_SESSIONS,
        "early_sessions": early_sessions, "late_sessions": late_sessions,
        "kde_set": {"full_baseline_tiles": int(Xfull.shape[0]),
                    "all_natural_rears": int(Xrears.shape[0]),
                    "early_matched_rears": int(Xearly.shape[0]),
                    "late_matched_rears": int(Xlate.shape[0])},
        "rear_counts": {"baseline_rears": nrr, "baseline_rears_kept": nrk,
                        "early_rears": ner, "early_matched": nek,
                        "late_rears": nlr, "late_matched": nlk},
        "pc_variance": {"PC1": float(ev[0]), "PC2": float(ev[1])},
        "overlap_PC": ov_pc, "overlap_UMAP": ov_um,
        "method": "attempts_umap.py verbatim pipeline (30-frame windows of 2 template "
                  "PCs, z-scored). EMBEDDING (PCA(2) and UMAP(2)) FIT ON the baseline "
                  "non-overlapping W-frame tiles from the 3 natural sessions (the full "
                  "natural repertoire). KDE set projected in via .transform(): full = "
                  "the baseline tiles themselves; all_natural_rears = every rear in the "
                  "3 baseline sessions (green bubble, no overlap %); early/late = matched "
                  "rears (pass session base band) over template-1 / template-5 sessions. "
                  "KDE bw 0.30 240x240 (0.3-99.7 pct +14%); overlap = kNN precision (k=5) "
                  "vs the baseline-tile repertoire",
    }, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)

    import h5py
    with h5py.File(os.path.join(args.output_dir, "plot_data.h5"), "w") as f:
        for k, s in pc_scores.items():
            f.create_dataset(f"PC/{k}", data=s)
        for k, s in um_scores.items():
            f.create_dataset(f"UMAP/{k}", data=s)
    print(f"\nWrote PC + UMAP figures + summary.json + plot_data.h5 to {args.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()
