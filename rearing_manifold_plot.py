"""Rearing-kinematics pose manifold: build the figure from collected rollouts + reference mocap.

Self-contained (numpy / h5py / scipy / scikit-learn / matplotlib only — no project imports, no
seaborn). Given a reference motion-capture file and one HDF5 per training checkpoint, it builds a
2D PCA "pose-shape" space, detects rears, and draws the region each condition occupies plus each
checkpoint's overlap with the natural-rear manifold. This is the shareable analysis/plot stage;
the rollouts are produced separately (any rollout writing the schema below works).

The method and figure style MATCH the published bowl-escape kinematic-manifold analysis
(figures-mimic-mjx `notebooks/fig5_09_bowl_escape_kinematic_analysis.ipynb`,
`docs/methods_kinematic_manifold.md`), with three deliberate rearing-specific adaptations:
  (a) conditions are a LEARNING PROGRESSION (early -> ... -> final checkpoints of one policy),
      shown as a blue sequential ramp, rather than two distinct policies;
  (b) only REAR poses enter the analysis (bowl-escape used all in-bowl poses); a rear is the full
      up-down head-height excursion at the task's own thresholds (below);
  (c) the reference is the FULL natural repertoire (all mocap poses, like bowl-escape); overlap is
      the share of the policy's rear poses that fall inside it. Only the POLICY side is filtered to
      rears — the reference is used whole.
Everything else — 91 pairwise distances over 14 sites, pooled StandardScaler + PCA(2) on the
reference + policies, 240x240 grid over the 0.3-99.7 percentile range padded 14%, gaussian_kde
bw 0.30, 50%/90% HDR contours, and k-NN manifold-precision overlap (k=5, 95th-pct-capped radii,
10k-pose subsamples) — matches.

--------------------------------------------------------------------------------------------------
WHAT A "REAR" IS
--------------------------------------------------------------------------------------------------
head_height = z(skull) - z(torso)  (a body-to-body vertical difference, in metres).
  * instantaneous rear : head_height >= REAR_THRESHOLD  (0.05 m)
  * a rear *segment*    : a maximal run of frames >= 0.05, extended outward to the nearest frame
                          <= RESET_THRESHOLD (0.01 m) on each side, i.e. the full up-down excursion
                          (ascent + hold + descent). These are the two thresholds the RodentRearing
                          task itself uses; change them only if your task differs.

--------------------------------------------------------------------------------------------------
POSE FEATURE  (translation- and rotation-invariant)
--------------------------------------------------------------------------------------------------
Each pose -> 91 pairwise Euclidean distances among 14 body sites (SITE_BODIES): head; three
axial-spine bodies; two tail bodies; the proximal + distal segment of each of the four limbs.
C(14,2) = 91. Because it uses only inter-site distances it needs no root/heading alignment. Sites
are resolved by NAME (a trailing "-rodent" suffix is tolerated), so the reference model and the
rollout model may differ in body count/order as long as they share bone geometry. (Sanity: the
script asserts a RIGID single-bone link — upper_arm<->lower_arm, upper_leg<->lower_leg — has the
same length in both sources; a multi-joint span such as scapula<->hand is posture-dependent and
must NOT be used for that check.)

--------------------------------------------------------------------------------------------------
ADAPTING THE POSE REPRESENTATION  (e.g. keypoints instead of FK body sites)
--------------------------------------------------------------------------------------------------
Everything below the feature vector is representation-agnostic: standardize -> PCA(2) -> KDE HDR ->
k-NN overlap operate on any (N, D) pose descriptor. To swap representations, replace ONLY
`site_feats()` (and `SITE_BODIES`); the rest is unchanged — and so is the rear detector, which needs
only skull/torso z, not the pose feature. Keypoints work in principle, with two requirements the
current FK body sites get for free:
  * Invariance. Raw world keypoints carry global translation + rotation. Either take PAIRWISE
    DISTANCES between them (K keypoints -> C(K,2) features; translation/rotation invariant, no
    alignment step — the direct analogue of what this script does), or egocentrically align
    (subtract the root, rotate to a canonical heading) and flatten. Prefer pairwise distances:
    alignment adds a root/heading definition and its own failure modes.
  * Shared geometry. Reference and policy must express the SAME keypoints off the SAME body. The
    mocap file already has markers (`kp_data` / `marker_sites`); a policy rollout has none, so you
    must read the equivalent VIRTUAL markers from the simulated state (the model's marker sites /
    stac-mjx offsets) with the SAME calibration. The bone-length guard here generalizes to a fixed
    marker-pair distance check; if the two marker placements differ, the distances differ
    systematically and the comparison is invalid.
Practical notes: keypoints are higher-D (23 markers -> 253 pairwise distances) — fine, PCA still
reduces to 2D. Mocap markers carry measurement noise (jitter, occlusion) that FK-from-qpos body
sites do not, so expect a slightly broader, noisier manifold. The 14 body sites used here are
body-frame origins (no marker offsets), which is exactly why they are trivially shared across
models; that convenience is what you trade away when moving to calibrated markers.

--------------------------------------------------------------------------------------------------
INPUT DATA SCHEMA
--------------------------------------------------------------------------------------------------
1) REFERENCE mocap file  (--reference, one HDF5):
     xpos        float32/64  (N_frames, n_bodies, 3)   world body positions per frame
     names_xpos  bytes|str   (n_bodies,)               body names, ordered as xpos axis 1
   All poses are used (the full natural repertoire); other datasets (qpos, xquat, ...) are ignored.
   Reference frames are subsampled by --ref_stride (default 2, e.g. 50 Hz -> 25 Hz).

2) ROLLOUT files  (--rollout_dir/step_*.h5, one per checkpoint):
     xpos         float32   (T, B, n_bodies, 3)   T control steps x B rollouts (envs)
     done         float32   (T, B)                termination flag in {0,1}; the FIRST done per
                                                   rollout truncates it (no autoreset assumed)
     names_xpos   bytes|str (n_bodies,)           body names, ordered as xpos axis 2
     head_height  float32   (T, B)   OPTIONAL     z(skull)-z(torso); recomputed from xpos if absent
     attrs["step"] int                            checkpoint step (used for ordering + labels)
   The filename must sort by step (e.g. step_15728640.h5). Rollout rear poses are subsampled by
   --rollout_stride (default 4, e.g. 100 Hz -> 25 Hz) after pooling across rollouts. Provide >= 2
   checkpoints; 3 (early/intermediate/final) is the intended case.

--------------------------------------------------------------------------------------------------
OUTPUTS  (--output_dir)
--------------------------------------------------------------------------------------------------
  rearing_manifold.{png,pdf,svg}   the figure (matplotlib; 600 dpi png, editable-text pdf/svg)
  summary.json                     steps, PC1/PC2 variance, overlap per checkpoint, pose counts,
                                   bone-length guard (mm)
  plot_data.h5                     per-condition 2D PC scores (to re-plot without recomputing)

USAGE
  python rearing_manifold_plot.py \
      --reference   rodent_reference_clips.h5 \
      --rollout_dir rollouts/ \
      --output_dir  out/

DEPENDENCIES  numpy, h5py, scipy, scikit-learn, matplotlib   (no seaborn)
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----- task thresholds (metres); the rear detector matches the RodentRearing task -----
REAR_THRESHOLD = 0.05
RESET_THRESHOLD = 0.01

# ----- the 14 pose sites (plain base names; a trailing "-rodent" is tolerated). Matches
#       figures-mimic-mjx fig5_09 / docs/methods_kinematic_manifold.md exactly. -----
SITE_BODIES = (
    "skull",                                   # head
    "torso", "vertebra_3", "pelvis",           # 3 axial-spine
    "vertebra_C1", "vertebra_C6",              # 2 tail (proximal, distal)
    "scapula_L", "hand_L",                     # forelimb L (proximal, distal)
    "scapula_R", "hand_R",                     # forelimb R
    "upper_leg_L", "foot_L",                   # hindlimb L (proximal, distal)
    "upper_leg_R", "foot_R",                   # hindlimb R
)
_PAIR_IU = np.triu_indices(len(SITE_BODIES), k=1)  # 91 (i,j) pairs
_SUFFIX = "-rodent"

# ----- analysis constants (match fig5_09 / methods doc) -----
GRID_N = 240          # KDE grid resolution
PAD = 0.14            # grid padding fraction beyond the 0.3-99.7 pct range
PCTL = (0.3, 99.7)    # robust extent percentiles for the grid
BW = 0.30             # gaussian_kde bandwidth factor (kernel cov = 0.09 x sample cov)
N_DENS = 40000        # max poses per condition into the KDE
K_NN, N_KNN, CAP_PCT = 5, 10000, 95   # overlap: k, subsample size, radius cap percentile
HDR_LEVELS = ((0.50, 1.8), (0.90, 1.1))  # (mass fraction, contour linewidth)

# ----- palette: grey for the natural repertoire + blue ramp for the ordered checkpoints -----
RGB_FULL = (0.80, 0.80, 0.82)      # full natural repertoire fill
CLINE_FULL = (0.34, 0.34, 0.37)    # full-repertoire contour + inline % tags


# ==================================================================================================
# body / feature helpers
# ==================================================================================================
def _names(h5_names) -> list[str]:
    return [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in h5_names]


def resolve_body_index(names: list[str], base: str) -> int:
    norm = [n[: -len(_SUFFIX)] if n.endswith(_SUFFIX) else n for n in names]
    for i, n in enumerate(norm):
        if n == base:
            return i
    raise KeyError(f"body {base!r} not in names (have e.g. {sorted(set(norm))[:8]})")


def site_indices(names: list[str]) -> np.ndarray:
    return np.array([resolve_body_index(names, b) for b in SITE_BODIES], dtype=int)


def pairwise_site_distances(xpos_sites: np.ndarray) -> np.ndarray:
    """(..., 14, 3) -> (..., 91) inter-site distances (indexes the 91 pairs directly)."""
    a = xpos_sites[..., _PAIR_IU[0], :]
    b = xpos_sites[..., _PAIR_IU[1], :]
    return np.linalg.norm(a - b, axis=-1)


def site_feats(xpos: np.ndarray, names: list[str]) -> np.ndarray:
    """(N, n_bodies, 3) -> (N, 91)."""
    return pairwise_site_distances(xpos[:, site_indices(names), :])


def head_height_from_xpos(xpos: np.ndarray, names: list[str]) -> np.ndarray:
    s, t = resolve_body_index(names, "skull"), resolve_body_index(names, "torso")
    return xpos[..., s, 2] - xpos[..., t, 2]


def bone_len(xpos: np.ndarray, names: list[str], a: str, b: str) -> float:
    ia, ib = resolve_body_index(names, a), resolve_body_index(names, b)
    return float(np.median(np.linalg.norm(xpos[:, ia, :] - xpos[:, ib, :], axis=-1)))


# Rigid single-bone parent->child links (fixed length regardless of posture) — used ONLY for the
# bone-length guard, NOT as feature sites. A multi-joint span like scapula<->hand is posture-
# dependent and would false-trip the guard.
_GUARD_BONES = (("upper_arm_L", "lower_arm_L"), ("upper_leg_L", "lower_leg_L"))


def _bone_lengths(xpos: np.ndarray, names: list[str]) -> dict:
    return {f"{a}-{b}": bone_len(xpos, names, a, b) for a, b in _GUARD_BONES}


# ==================================================================================================
# rear detection
# ==================================================================================================
def valid_before_done(done: np.ndarray) -> np.ndarray:
    """Mask of frames strictly before the first done==1 (frames after are post-termination)."""
    flagged = np.flatnonzero(done > 0.5)
    term = int(flagged[0]) if flagged.size else len(done)
    m = np.zeros(len(done), dtype=bool)
    m[:term] = True
    return m


def bracketed_rear_mask(head_height, valid, up=REAR_THRESHOLD, down=RESET_THRESHOLD):
    """Frames in a full up-down rear excursion, and the excursion count. See module docstring."""
    T = len(head_height)
    above = (head_height >= up) & valid
    reset_idx = np.flatnonzero((head_height <= down) & valid)
    mask = np.zeros(T, dtype=bool)
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return mask, 0
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        s, e = int(run[0]), int(run[-1])
        prior = reset_idx[reset_idx <= s]
        post = reset_idx[reset_idx >= e]
        mask[(int(prior[-1]) if prior.size else s):(int(post[0]) if post.size else e) + 1] = True
    mask &= valid
    return mask, len(runs)


# ==================================================================================================
# loaders
# ==================================================================================================
def load_reference(path, stride):
    """-> (full_feats (N,91), bones dict). ALL reference poses = the full natural repertoire.

    (Rears are detected only in the POLICY rollouts; the reference is used whole, as the natural
    behavioural manifold the policy's rears are compared against.)
    """
    with h5py.File(path, "r") as f:
        names = _names(f["names_xpos"][:])
        xpos = f["xpos"][::stride]
    print(f"[ref] {xpos.shape[0]} poses (full natural repertoire)", flush=True)
    return site_feats(xpos, names), _bone_lengths(xpos, names)


def load_rollout(path, stride):
    """-> (rear_feats (Nr,91), step int, bones dict, n_excursions)."""
    with h5py.File(path, "r") as f:
        names = _names(f["names_xpos"][:])
        xpos = f["xpos"][:]                       # (T, B, n_bodies, 3)
        done = f["done"][:]                       # (T, B)
        hh = f["head_height"][:] if "head_height" in f else None
        step = int(f.attrs.get("step", -1))
    if hh is None:
        hh = head_height_from_xpos(xpos, names)   # recompute if not stored
    T, B = hh.shape
    feats, nexc = [], 0
    for b in range(B):
        valid = valid_before_done(done[:, b])
        mask, k = bracketed_rear_mask(hh[:, b], valid)
        nexc += k
        if mask.any():
            feats.append(site_feats(xpos[mask, b], names))
    feats = np.concatenate(feats, 0)[::stride] if feats else np.zeros((0, 91))
    bones = _bone_lengths(xpos.reshape(-1, xpos.shape[-2], 3), names)
    print(f"[roll step={step}] rear poses={feats.shape[0]} excursions={nexc}", flush=True)
    return feats, step, bones, nexc


# ==================================================================================================
# density (KDE on the shared grid) + overlap (k-NN manifold precision)
# ==================================================================================================
def kde_grid(scores, grid_xy, n_grid, dA, bw=BW):
    """Unit-mass KDE density on the grid (240x240)."""
    Z = gaussian_kde(scores.T, bw_method=bw)(grid_xy).reshape(n_grid, n_grid)
    return Z / (Z.sum() * dA)


def hdr_level(Z, frac, dA):
    """Density threshold enclosing `frac` of the total probability mass."""
    v = np.sort(Z.ravel())[::-1]
    c = np.cumsum(v) * dA
    return float(v[min(np.searchsorted(c / c[-1], frac), len(v) - 1)])


def _subsample(x, n, rng):
    return x if x.shape[0] <= n else x[rng.choice(x.shape[0], n, replace=False)]


def overlap_knn(policy, ref, rng, k=K_NN, cap_pct=CAP_PCT, n_sub=N_KNN, batch=2000):
    """k-NN manifold precision (Kynkaanniemi 2019): fraction of policy poses inside the union of
    balls centred on each reference pose with radius = distance to its k-th NN (radii capped at the
    `cap_pct` percentile), evaluated on n_sub-pose subsamples in the 2D PCA plane."""
    if policy.shape[0] == 0 or ref.shape[0] < k + 1:
        return float("nan")
    ref = _subsample(ref, n_sub, rng)
    policy = _subsample(policy, n_sub, rng)
    r = cKDTree(ref).query(ref, k=k + 1)[0][:, k]          # k-th NN distance (self excluded)
    r = np.minimum(r, np.percentile(r, cap_pct))
    inside = np.zeros(policy.shape[0], dtype=bool)
    for s in range(0, policy.shape[0], batch):
        d = np.linalg.norm(policy[s:s + batch, None, :] - ref[None, :, :], axis=2)  # (b, M)
        inside[s:s + batch] = (d <= r[None, :]).any(axis=1)
    return float(inside.mean())


# ==================================================================================================
# figure  (style matched to figures-mimic-mjx fig5_09 make_panel)
# ==================================================================================================
def _weight(d, vmax, gamma, amax):
    return (amax * np.clip(d / vmax, 0.0, 1.0) ** gamma)[..., None]


def _composite(layers, vmax, gamma=0.55, amax=0.80):
    """layers = [(density, rgb), ...] multiply-blended over white (order-independent), so
    co-occupied regions darken into a distinct combined tone."""
    H, W = layers[0][0].shape
    out = np.ones((H, W, 3), np.float32)
    for d, rgb in layers:
        a = _weight(d, vmax, gamma, amax)
        out = out * (1.0 - a * (1.0 - np.asarray(rgb, np.float32)))
    return out


def _dim(rgb, f=0.35):
    return tuple(c + (1.0 - c) * f for c in rgb)


def blue_ramp(n):
    """n ordered (fill, line) blues: early=light -> late=dark."""
    import matplotlib as mpl
    blues = mpl.colormaps["Blues"]
    fills = [blues(x)[:3] for x in np.linspace(0.42, 0.86, n)]
    lines = [blues(x)[:3] for x in np.linspace(0.66, 1.0, n)]
    return list(zip(fills, lines))


def make_figure(out_dir, gx, gy, dens, labels, steps, ov_full, ev, ramp):
    """Full natural repertoire (grey) vs the policy checkpoints (blue ramp), with each checkpoint's
    overlap-with-repertoire in the legend."""
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
    mpl.rcParams.update({"figure.facecolor": "w", "savefig.dpi": 600, "savefig.transparent": True,
                         "font.size": 8, "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    dA = (gx[1] - gx[0]) * (gy[1] - gy[0])
    extent = [gx.min(), gx.max(), gy.min(), gy.max()]
    cgx = np.linspace(extent[0], extent[1], GRID_N)
    cgy = np.linspace(extent[2], extent[3], GRID_N)
    vmax = max(Z.max() for Z in dens.values())

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    # multiply-composite density: full repertoire (light grey) + checkpoints (blue ramp)
    comp = [(dens["full"], RGB_FULL)] + [(dens[l], ramp[i][0]) for i, l in enumerate(labels)]
    ax.imshow(_composite(comp, vmax), extent=extent, origin="lower", aspect="auto",
              interpolation="bilinear", zorder=0)

    # full-repertoire 50%/90% HDR contours with inline % tags — the natural behavioural manifold
    for frac, lw in HDR_LEVELS:
        cs = ax.contour(cgx, cgy, dens["full"], levels=[hdr_level(dens["full"], frac, dA)],
                        colors=[CLINE_FULL], linewidths=lw, zorder=3)
        t = ax.clabel(cs, inline=True, inline_spacing=1, fontsize=6, fmt=lambda _l, f=frac: f"{f:.0%}")
        for tt in t:
            tt.set_zorder(6)
            tt.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground="white")])
    # checkpoints
    for i, lab in enumerate(labels):
        for frac, lw in HDR_LEVELS:
            ax.contour(cgx, cgy, dens[lab], levels=[hdr_level(dens[lab], frac, dA)],
                       colors=[ramp[i][1]], linewidths=lw, zorder=2)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(f"Kinematics PC 1 ({ev[0]*100:.0f}%)", fontsize=8, fontweight="bold", labelpad=1.5)
    ax.set_ylabel(f"Kinematics PC 2 ({ev[1]*100:.0f}%)", fontsize=8, fontweight="bold", labelpad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=8, length=2.5, width=0.8, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8)
        ax.spines[sp].set_position(("outward", 3))

    # swatch-free text legend: the natural repertoire, then each checkpoint with its overlap
    rows = [TextArea("Full natural\nrepertoire", textprops=dict(color=CLINE_FULL, size=8, linespacing=1.15))]
    for i, (lab, st) in enumerate(zip(labels, steps)):
        col = ramp[i][1]
        rows.append(TextArea(f"{lab} — {st/1e6:.0f}M", textprops=dict(color=col, size=8, linespacing=1.15)))
        rows.append(TextArea(f"{ov_full[lab]:.0%} in repertoire",
                             textprops=dict(color=_dim(col), size=6, linespacing=1.2)))
    ax.add_artist(AnchoredOffsetbox(loc="upper left", child=VPacker(children=rows, pad=0, sep=1.5, align="left"),
                                    frameon=False, borderpad=0.2, bbox_to_anchor=(1.02, 1.0),
                                    bbox_transform=ax.transAxes))
    fig.tight_layout(pad=0.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(out_dir, f"rearing_manifold.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ==================================================================================================
# main
# ==================================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="reference mocap HDF5 (see schema in docstring)")
    p.add_argument("--rollout_dir", required=True, help="dir of step_*.h5 rollout files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ref_stride", type=int, default=2, help="reference subsample (50->25 Hz => 2)")
    p.add_argument("--rollout_stride", type=int, default=4, help="rollout rear-pose subsample (100->25 Hz => 4)")
    p.add_argument("--balance_cap", type=int, default=15000, help="max poses per class in the PCA pool")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    full_feats, ref_bones = load_reference(args.reference, args.ref_stride)
    files = sorted((os.path.join(args.rollout_dir, f) for f in os.listdir(args.rollout_dir)
                    if f.startswith("step_") and f.endswith(".h5")),
                   key=lambda pth: int(os.path.basename(pth)[5:-3]))
    if len(files) < 2:
        raise FileNotFoundError(f"need >=2 step_*.h5 in {args.rollout_dir}, found {len(files)}")
    rolls = [load_rollout(f, args.rollout_stride) for f in files]
    steps = [r[1] for r in rolls]
    labels = (["early", "intermediate", "final"] if len(rolls) == 3
              else [f"ckpt{n}" for n in range(len(rolls))])

    # bone-length guard: identical skeleton geometry => stored xpos ≡ shared-model FK
    guard = {}
    for bone in ref_bones:
        vals = [ref_bones[bone]] + [r[2][bone] for r in rolls]
        guard[bone] = [round(v * 1000, 3) for v in vals]
        spread = max(vals) - min(vals)
        print(f"[guard] {bone}: {guard[bone]} mm  spread={spread*1000:.2f} mm", flush=True)
        if spread > 0.002:
            raise RuntimeError(f"bone-length guard FAILED for {bone} (spread {spread*1000:.1f} mm > 2 mm): "
                               "reference and rollout models differ in geometry; features not comparable.")

    policy = {lab: r[0] for lab, r in zip(labels, rolls)}
    # PCA plane fit on the class-balanced pool of the FULL reference repertoire + all policies
    # (bowl-escape recipe: reference + rollouts). The natural-rear subset is a derived overlay.
    pool_conds = {"full": full_feats, **policy}
    n_bal = min(args.balance_cap, min(v.shape[0] for v in pool_conds.values()))
    if n_bal < 50:
        print(f"WARNING: smallest condition has {n_bal} poses; density/PCA will be noisy", flush=True)
    pool = np.concatenate([v[rng.choice(v.shape[0], n_bal, replace=False)] for v in pool_conds.values()])
    scaler = StandardScaler().fit(pool)
    pca = PCA(n_components=2, random_state=args.seed).fit(scaler.transform(pool))
    ev = pca.explained_variance_ratio_
    print(f"PCA on {pool.shape[0]} balanced poses ({n_bal}/class, full+policies); "
          f"PC1={ev[0]*100:.1f}% PC2={ev[1]*100:.1f}%", flush=True)
    proj = lambda x: pca.transform(scaler.transform(x))
    scores = {"full": proj(full_feats), **{k: proj(v) for k, v in policy.items()}}

    # shared grid over the 0.3-99.7 pct range of all embedded poses, padded 14%
    allp = np.concatenate(list(scores.values()))
    lo = np.percentile(allp, PCTL[0], axis=0)
    hi = np.percentile(allp, PCTL[1], axis=0)
    pad = PAD * (hi - lo)
    lo, hi = lo - pad, hi + pad
    gx = np.linspace(lo[0], hi[0], GRID_N)
    gy = np.linspace(lo[1], hi[1], GRID_N)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.vstack([GX.ravel(), GY.ravel()])
    dA = (gx[1] - gx[0]) * (gy[1] - gy[0])
    cap = lambda s: _subsample(s, N_DENS, rng)
    dens = {k: kde_grid(cap(v), grid_xy, GRID_N, dA) for k, v in scores.items()}

    # overlap per checkpoint: policy rears vs the full natural repertoire
    ov_full = {lab: overlap_knn(scores[lab], scores["full"], rng) for lab in labels}
    for lab in labels:
        print(f"[overlap] {lab} (step {dict(zip(labels, steps))[lab]}): "
              f"repertoire={ov_full[lab]:.3f}", flush=True)

    make_figure(args.output_dir, gx, gy, dens, labels, steps, ov_full, ev, blue_ramp(len(labels)))

    json.dump({"steps": dict(zip(labels, steps)),
               "pc_variance": {"PC1": float(ev[0]), "PC2": float(ev[1])},
               "overlap_with_full_repertoire": ov_full,
               "pose_counts": {"full": int(full_feats.shape[0]),
                               **{k: int(v.shape[0]) for k, v in policy.items()}},
               "balance_per_class": int(n_bal),
               "bone_length_guard_mm": guard,
               "method": "fig5_09 kinematic-manifold; 14 sites/91 dist; pooled StandardScaler+PCA on "
                         "full-reference+policies; KDE bw 0.30 on 240x240 (0.3-99.7 pct +14% pad); "
                         "overlap=kNN precision (k=5, 95th-pct cap, 10k subsample), policy rears vs "
                         "full natural repertoire"},
              open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)
    with h5py.File(os.path.join(args.output_dir, "plot_data.h5"), "w") as f:
        for k, s in scores.items():
            f.create_dataset(f"scores/{k}", data=s)
    print(f"\nWrote figure + summary.json + plot_data.h5 to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
