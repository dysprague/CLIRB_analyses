"""Rearing-kinematics pose-manifold plot from REAL R5 keypoint data.

CLIRB adaptation of `rearing_manifold_plot.py`. Instead of simulated FK body
sites + a 91-distance pose feature, each point here is a 50-frame WINDOW of
egocentric DANNCE keypoints (23 kp x 3 dims x 50 frames = 3450 features),
subsampled to the 20 Hz SLEAP grid. A 2-D PCA "pose-shape" space is fit on a
class-balanced pool of all conditions; the natural repertoire is the grey
reference manifold and the two training checkpoints are a blue early->late ramp,
with each checkpoint's k-NN overlap with the repertoire in the legend.

CONDITIONS (R5)
  full natural repertoire : 2026_06_05_1, 2026_06_08_1, 2026_06_09_1
  early training          : 2026_06_19_1
  late training           : 2026_07_27_1

HOW EACH CONDITION'S POINTS ARE BUILT
  * baseline (repertoire): every REAR in the 3 natural sessions (detected on
    SLEAP PC0 with the notebook hysteresis: rise 200 / reset 150, > 0.5 s). For
    each rear's full up-hold-down excursion, the 30-frame window best-aligned
    (min-MSE, 2-PC) to the EARLY template (R5_template_1) inside
    [rear_start, rear_end + 0.75 s] is found, then padded +/-10 -> 50 frames.
    These rear windows fit the PCA (pooled with the training conditions).
  * early / late training: the ONLINE template matches (behavior_log.csv rows
    with triggered==1) are the match set -- exactly what the rig fired on this
    session (104 early, 101 late). Each trigger's online frameNumber maps through
    frame_mapping.csv (processed_frame -> SLEAP triang row); the online trigger
    frame IS the match END (last frame of the 30-frame window), so padding 10
    frames before and after gives a 50-frame window ending at row+10.

POSE PIPELINE
  Matches are found ONLINE on SLEAP; here we take those match frames and build
  the manifold WINDOW from DANNCE keypoints subsampled to the closest SLEAP
  frames. DANNCE (N,3,23) -> (N,23,3) -> subsample to SLEAP frames via
  dannce_idx_for_sleap_cams[1:] -> normalize_skeleton_batch (egocentric: centre
  SpineM, rotate SpineF->SpineM to +X; NO z-flip for DANNCE) -> 5-frame CAUSAL
  median in rotated-keypoint space -> flatten (69/frame). No per-feature scaling.

Everything below the window feature (StandardScaler-free PCA(2) -> KDE HDR ->
k-NN overlap) mirrors rearing_manifold_plot.py's analysis constants exactly.

USAGE
  .../envs/clirb_analysis/bin/python rearing_manifold_real.py --output_dir out/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeleton import normalize_skeleton_batch, project_to_pcs  # noqa: E402

# --------------------------------------------------------------------------- #
# Data configuration
# --------------------------------------------------------------------------- #
DATA_ROOT = "/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data"
RAT = "R5"
BASELINE_SESSIONS = ["2026_06_05_1", "2026_06_08_1", "2026_06_09_1"]
# training conditions: (label, session). Matches come from each session's own
# behavior_log.csv (the online rule using that session's own template).
TRAIN_SESSIONS = [("early", "2026_06_19_1"), ("late", "2026_07_27_1")]

WIN = 50               # frames per point (2.5 s @ 20 Hz)
PAD = 10               # frames padded before/after the 30-frame window -> 50
TEMPLATE_LEN = 30      # online template / best-align window length (frames)
N_COMPONENTS = 2       # PCs kept for the manifold plot
FPS = 20
N_KP = 23

# Natural-repertoire rears are detected + best-aligned to the EARLY session's
# template (its pc_weights define the PC space; PC0 drives the rear detector).
ALIGN_TEMPLATE_FILE = "R5_template_1.npz"

# ----- rear detector (single_session/multi_session hysteresis on SLEAP PC0) ---
REAR_RISE = 200.0      # PC0 enter-rear threshold
REAR_RESET = 150.0     # PC0 must fall below this before a new rear counts
MIN_REAR_S = 0.5       # PC0 must stay above REAR_RISE for at least this long
ALIGN_BUFFER_S = 0.75  # best-align search span = [rear_start, rear_end + buffer]

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


# --------------------------------------------------------------------------- #
# Keypoint pipeline
# --------------------------------------------------------------------------- #
def _causal_median_5_kp(rotated):
    """5-frame CAUSAL median in rotated-keypoint space (matches the online rule /
    single_session cell 6). Do NOT median-filter raw keypoints."""
    pad = 4
    padded = np.concatenate([np.repeat(rotated[:1], pad, axis=0), rotated], axis=0)
    T = rotated.shape[0]
    widx = np.arange(T)[:, None] + np.arange(5)[None, :]
    return np.median(padded[widx], axis=1).astype(np.float32)


def _reshape_dannce(dannce):
    """Raw DANNCE (N,3,23) or (N,1,3,23) -> (N,23,3)."""
    dannce = np.asarray(dannce)
    if dannce.ndim == 4:
        return dannce.squeeze(axis=1).transpose(0, 2, 1)
    return np.transpose(dannce, [0, 2, 1])


def load_dannce_aligned(rat, session):
    """DANNCE keypoints subsampled to the SLEAP 20 Hz grid: (n_sleap, 23, 3).

    Prefers processed/sleap_dannce_keys.mat['dannce_keys_3D']; if that file is
    absent (session's alignment step not yet run) falls back to the raw DANNCE
    prediction save_data_AVG0.mat + this session's aligned_data.mat index.
    """
    sd = os.path.join(DATA_ROOT, rat, session)
    dual_p = os.path.join(sd, "processed", "sleap_dannce_keys.mat")
    aligned_p = os.path.join(sd, "processed", "aligned_data.mat")
    al = sio.loadmat(aligned_p, simplify_cells=True)
    aidx = np.asarray(al["dannce_idx_for_sleap_cams"], dtype=int).ravel()[1:]

    if os.path.isfile(dual_p):
        keys = sio.loadmat(dual_p, simplify_cells=True)
        dannce = _reshape_dannce(keys["dannce_keys_3D"])
        src = "sleap_dannce_keys.mat"
    else:
        # raw DANNCE prediction fallback (predict_results or predict_results_finetune)
        pred = None
        for sub in ("predict_results", "predict_results_finetune"):
            p = os.path.join(sd, "dannce", "DANNCE", sub, "save_data_AVG0.mat")
            if os.path.isfile(p):
                pred = _reshape_dannce(sio.loadmat(p, simplify_cells=True)["pred"])
                src = f"raw {sub}/save_data_AVG0.mat"
                break
        if pred is None:
            raise FileNotFoundError(
                f"{rat}/{session}: no sleap_dannce_keys.mat and no raw DANNCE prediction")
        dannce = pred

    if aidx.max() >= dannce.shape[0]:
        raise ValueError(f"{rat}/{session}: alignment idx max {aidx.max()} exceeds "
                         f"DANNCE frames {dannce.shape[0]}")
    aligned = dannce[aidx]                     # (n_sleap, 23, 3)
    print(f"[load] {rat}/{session}: {aligned.shape[0]} SLEAP-aligned DANNCE frames "
          f"({src})", flush=True)
    return aligned.astype(np.float32)


def rotated_keypoints(dannce_aligned):
    """(n,23,3) DANNCE -> egocentric + causal-median rotated kp (n,23,3)."""
    rot, _, _ = normalize_skeleton_batch(dannce_aligned)   # NO z-flip for DANNCE
    return _causal_median_5_kp(rot)


def sleap_pcs(rat, session, pc_weights, feature_means):
    """SLEAP triang keypoints -> PC scores via the given template weights.

    Online-equivalent: flip z (SLEAP only) -> normalize -> causal median in
    rotated-kp space -> project. Returns (n_sleap, n_pcs)."""
    sd = os.path.join(DATA_ROOT, rat, session)
    sleap = np.load(os.path.join(sd, "sleap", "triang_keys_3D.npy")).astype(np.float32)
    sleap[:, :, 2] = -sleap[:, :, 2]                       # z-flip is SLEAP-only
    rot, _, _ = normalize_skeleton_batch(sleap)
    rot = _causal_median_5_kp(rot)
    return project_to_pcs(rot, pc_weights, feature_means)


def window_feats(rot, ends):
    """rotated kp (n,23,3) + WIN-window END frames -> (len(ends), 3450).

    ends[k] is the LAST frame of the WIN-frame window [end-WIN+1, end].
    """
    ends = np.asarray(ends, dtype=int)
    starts = ends - WIN + 1
    idx = starts[:, None] + np.arange(WIN)[None, :]        # (n_ends, WIN)
    win = rot[idx]                                         # (n_ends, WIN, 23, 3)
    return win.reshape(win.shape[0], -1)                  # (n_ends, WIN*23*3=3450)


# --------------------------------------------------------------------------- #
# Baseline: natural-repertoire REARS, best-aligned to the early template
# --------------------------------------------------------------------------- #
def detect_rears(pc0):
    """PC0 hysteresis rear detector (single/multi_session): a rear opens when PC0
    rises above REAR_RISE and closes when it falls below REAR_RESET; kept only if
    PC0 stayed above REAR_RISE for > MIN_REAR_S. Returns [(start, end), ...] over
    the FULL up-hold-down excursion."""
    rise, reset = REAR_RISE, REAR_RESET
    min_above = MIN_REAR_S * FPS
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


def best_align_end(rs, re, pcs2, pc_template, buf):
    """End-frame in [max(rs, TEMPLATE_LEN-1), re+buf] whose TEMPLATE_LEN-frame
    window has the lowest MSE to pc_template (2-PC). None if no window fits.
    == single_session _best_align_end."""
    L = pc_template.shape[0]
    e_min = max(rs, L - 1)
    e_max = min(re + buf, pcs2.shape[0] - 1)
    if e_max < e_min:
        return None
    ends = np.arange(e_min, e_max + 1)
    starts = ends - (L - 1)
    idx = starts[:, None] + np.arange(L)[None, :]
    win = pcs2[idx, :pc_template.shape[1]]
    mse = ((win - pc_template[None]) ** 2).mean(axis=(1, 2))
    return int(ends[int(np.argmin(mse))])


def baseline_rear_window_feats(rat, session, rot, align_template):
    """Natural-repertoire rears -> best-aligned 30-frame window -> pad +/-PAD ->
    50-frame DANNCE windows. Returns (feats, n_rears, n_aligned, n_dropped_edge).

    Rears are detected on SLEAP PC0; the best-align search + PC template use the
    align template (early session's R5_template_1). The 50-frame window is built
    from the DANNCE-aligned rotated kp `rot`."""
    pcw = align_template["pc_weights"]
    fm = align_template["feature_means"]
    pct = align_template["template"][:, :N_COMPONENTS]
    pcs2 = sleap_pcs(rat, session, pcw, fm)[:, :N_COMPONENTS]
    rears = detect_rears(pcs2[:, 0])
    buf = int(round(ALIGN_BUFFER_S * FPS))
    n = rot.shape[0]
    win_ends, n_aligned = [], 0
    for rs, re in rears:
        m = best_align_end(rs, re, pcs2, pct, buf)         # 30-frame window END
        if m is None:
            continue                                       # rear too short to fit a window
        n_aligned += 1
        we = m + PAD                                       # 50-frame window END
        if we - WIN + 1 >= 0 and we <= n - 1:
            win_ends.append(we)
    n_drop = n_aligned - len(win_ends)
    feats = (window_feats(rot, np.array(win_ends, dtype=int)) if win_ends
             else np.zeros((0, WIN * N_KP * 3), np.float32))
    return feats, len(rears), n_aligned, n_drop


# --------------------------------------------------------------------------- #
# Online template matches -> training windows
# --------------------------------------------------------------------------- #
def online_match_rows(rat, session):
    """SLEAP triang-row index of every ONLINE template match (behavior_log.csv
    triggered==1), in session order.

    behavior_log.csv 'frame' is the online frameNumber (== processed_frame in
    frame_mapping.csv); its position in frame_mapping is the SLEAP triang row.
    The trigger frame is the LAST frame of the 30-frame online match window.
    """
    sd = os.path.join(DATA_ROOT, rat, session)
    bl = pd.read_csv(os.path.join(sd, "sleap", "behavior_log.csv"), skipinitialspace=True)
    log_frames = bl.loc[bl["triggered"] == 1, "frame"].to_numpy(dtype=int)
    fm = pd.read_csv(os.path.join(sd, "sleap", "frame_mapping.csv"), skipinitialspace=True)
    proc_to_row = {int(p): i for i, p in enumerate(fm["processed_frame"].to_numpy())}
    rows = np.array([proc_to_row.get(int(f), -1) for f in log_frames])
    n_trig = len(log_frames)
    n_unmapped = int((rows < 0).sum())
    return rows[rows >= 0], n_trig, n_unmapped


def training_window_feats(rot, match_rows):
    """Online-match rows -> 50-frame windows on the DANNCE-aligned rotated kp.

    Each match row m is the online match END (last of the 30-frame window
    [m-29, m]). Padding PAD before and after gives [m-39, m+10] (30+2*PAD=50);
    in window_feats' END-frame convention that window ends at m+PAD. Matches too
    close to a session edge to fit a full 50-frame window are dropped.
    """
    if len(match_rows) == 0:
        return np.zeros((0, WIN * N_KP * 3), np.float32), 0
    win_ends = np.asarray(match_rows, dtype=int) + PAD     # 50-frame window END
    n = rot.shape[0]
    valid = (win_ends - WIN + 1 >= 0) & (win_ends <= n - 1)
    n_drop = int((~valid).sum())
    feats = window_feats(rot, win_ends[valid])
    return feats, n_drop


# --------------------------------------------------------------------------- #
# density (KDE) + overlap (k-NN manifold precision)  — copied from reference
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
# figure  (style matched to rearing_manifold_plot.py make_figure)
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


def make_figure(out_dir, gx, gy, dens, labels, ov_full, ev, ramp, counts):
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
    comp = [(dens["full"], RGB_FULL)] + [(dens[l], ramp[i][0]) for i, l in enumerate(labels)]
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
    for i, lab in enumerate(labels):
        for frac, lw in HDR_LEVELS:
            ax.contour(cgx, cgy, dens[lab], levels=[hdr_level(dens[lab], frac, dA)],
                       colors=[ramp[i][1]], linewidths=lw, zorder=2)

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(f"Pose-shape PC 1 ({ev[0]*100:.0f}%)", fontsize=8, fontweight="bold", labelpad=1.5)
    ax.set_ylabel(f"Pose-shape PC 2 ({ev[1]*100:.0f}%)", fontsize=8, fontweight="bold", labelpad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=8, length=2.5, width=0.8, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8); ax.spines[sp].set_position(("outward", 3))

    rows = [TextArea("Full natural\nrepertoire",
                     textprops=dict(color=CLINE_FULL, size=8, linespacing=1.15))]
    for i, lab in enumerate(labels):
        col = ramp[i][1]
        rows.append(TextArea(f"{lab} training",
                             textprops=dict(color=col, size=8, linespacing=1.15)))
        rows.append(TextArea(f"{ov_full[lab]:.0%} in repertoire",
                             textprops=dict(color=_dim(col), size=6, linespacing=1.2)))
    ax.add_artist(AnchoredOffsetbox(loc="upper left",
                                    child=VPacker(children=rows, pad=0, sep=1.5, align="left"),
                                    frameon=False, borderpad=0.2, bbox_to_anchor=(1.02, 1.0),
                                    bbox_transform=ax.transAxes))
    fig.tight_layout(pad=0.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(out_dir, f"rearing_manifold_real.{ext}"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default="results/rearing_manifold_real")
    p.add_argument("--balance_cap", type=int, default=15000,
                   help="max windows per condition in the PCA pool")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # align template (early session's) defines the PC space + rear best-align
    align_template = dict(np.load(
        os.path.join(DATA_ROOT, RAT, "templates", ALIGN_TEMPLATE_FILE), allow_pickle=True))

    # ---- baseline: natural-repertoire REARS, best-aligned to the early template,
    #      padded to 50-frame DANNCE windows. Pooled over the 3 natural sessions.
    full_list, base_counts = [], {}
    for s in BASELINE_SESSIONS:
        rot = rotated_keypoints(load_dannce_aligned(RAT, s))
        feats, n_rears, n_aligned, n_drop = baseline_rear_window_feats(
            RAT, s, rot, align_template)
        full_list.append(feats)
        base_counts[s] = {"rears": n_rears, "aligned": n_aligned,
                          "dropped_edge": n_drop, "windows": feats.shape[0]}
        print(f"[full] {s}: rears={n_rears} aligned={n_aligned} "
              f"dropped_edge={n_drop} windows={feats.shape[0]}", flush=True)
    full_feats = np.concatenate(full_list, 0)
    print(f"[full] {full_feats.shape[0]} baseline rear windows ({full_feats.shape[1]} feats)",
          flush=True)

    # ---- training conditions: ONLINE matches (behavior_log triggered==1) ->
    #      50-frame DANNCE-keypoint windows centred on each match.
    policy, counts, sessions = {}, {}, {}
    for lab, s in TRAIN_SESSIONS:
        rows, n_trig, n_unmapped = online_match_rows(RAT, s)
        rot = rotated_keypoints(load_dannce_aligned(RAT, s))
        feats, n_drop = training_window_feats(rot, rows)
        policy[lab] = feats
        sessions[lab] = s
        counts[lab] = {"online_triggers": n_trig, "unmapped": n_unmapped,
                       "dropped_edge": n_drop, "windows": feats.shape[0]}
        print(f"[{lab}] {s}: online_triggers={n_trig} unmapped={n_unmapped} "
              f"dropped_edge={n_drop} windows={feats.shape[0]}", flush=True)

    # ---- pooled, class-balanced PCA(2) on baseline + early + late (ref recipe)
    pool_conds = {"full": full_feats, **policy}
    pool_conds = {k: v for k, v in pool_conds.items() if v.shape[0] > 0}
    n_bal = min(args.balance_cap, min(v.shape[0] for v in pool_conds.values()))
    if n_bal < 50:
        print(f"WARNING: smallest condition has {n_bal} windows; PCA/density noisy", flush=True)
    from sklearn.decomposition import PCA
    pool = np.concatenate([v[rng.choice(v.shape[0], n_bal, replace=False)]
                           for v in pool_conds.values()])
    pca = PCA(n_components=N_COMPONENTS, random_state=args.seed).fit(pool)
    ev = pca.explained_variance_ratio_
    print(f"PCA on {pool.shape[0]} balanced windows ({n_bal}/class); "
          f"PC1={ev[0]*100:.1f}% PC2={ev[1]*100:.1f}%", flush=True)

    scores = {"full": pca.transform(full_feats),
              **{k: pca.transform(v) for k, v in policy.items() if v.shape[0] > 0}}
    labels = [lab for lab, _ in TRAIN_SESSIONS if lab in scores]

    # ---- shared grid (0.3-99.7 pct of all embedded windows, padded 14%)
    allp = np.concatenate(list(scores.values()))
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
    dens = {k: kde_grid(cap(v), grid_xy, GRID_N, dA) for k, v in scores.items()}

    ov_full = {lab: overlap_knn(scores[lab], scores["full"], rng) for lab in labels}
    for lab in labels:
        print(f"[overlap] {lab}: repertoire={ov_full[lab]:.3f}", flush=True)

    make_figure(args.output_dir, gx, gy, dens, labels, ov_full, ev,
                blue_ramp(len(labels)), counts)

    json.dump({
        "rat": RAT,
        "baseline_sessions": BASELINE_SESSIONS,
        "train_sessions": {lab: s for lab, s in TRAIN_SESSIONS},
        "align_template_file": ALIGN_TEMPLATE_FILE,
        "window_frames": WIN, "pad_frames": PAD, "template_len": TEMPLATE_LEN,
        "feature_dim": int(WIN * N_KP * 3),
        "pc_variance": {"PC1": float(ev[0]), "PC2": float(ev[1])},
        "overlap_with_full_repertoire": ov_full,
        "window_counts": {"full": int(full_feats.shape[0]),
                          "full_per_session": base_counts, **counts},
        "balance_per_class": int(n_bal),
        "method": "50-frame egocentric-DANNCE windows (23x3x50=3450 feats); pooled "
                  "PCA(2) on baseline+early+late (no per-feature scaling). BASELINE = "
                  "natural-repertoire rears (SLEAP-PC0 hysteresis 200/150, >0.5s), each "
                  "best-aligned (min-MSE, 2-PC) to the early template R5_template_1 "
                  "inside [rear_start, rear_end+0.75s] then padded +/-10 -> 50 frames. "
                  "TRAINING = online template matches (behavior_log triggered==1, each "
                  "session's own template) padded +/-10. All windows from DANNCE "
                  "keypoints subsampled to the closest SLEAP frame. KDE bw 0.30 on "
                  "240x240 (0.3-99.7 pct +14%); overlap = kNN precision (k=5, "
                  "95th-pct cap, 10k subsample) vs full repertoire",
    }, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)

    import h5py
    with h5py.File(os.path.join(args.output_dir, "plot_data.h5"), "w") as f:
        for k, s in scores.items():
            f.create_dataset(f"scores/{k}", data=s)
    print(f"\nWrote figure + summary.json + plot_data.h5 to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
