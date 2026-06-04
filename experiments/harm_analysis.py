"""
Track A — How harmful is the SLEAP/DANNCE template-match disagreement?

This script does NOT try to fix the disagreement. It measures whether the
disagreement is systematic (a stable, biased reward target — survivable for
training) or random (a noisy target — bad for learning).

Outputs go to results/figures/harm_analysis/<rat>/<session>/ and a single
results/metrics/<rat>_harm_summary.csv that aggregates per-session metrics.

Run:
    python harm_analysis.py --rat R1 --config primary
    python harm_analysis.py --rat R1 --config secondary
    python harm_analysis.py --rat R2 --config primary
    python harm_analysis.py --rat R3 --config primary
    python harm_analysis.py --rat all                  # loops the four configs above
"""
import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, mannwhitneyu

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from data_io import (
    get_sessions,
    load_aligned_data,
    load_opcon_events,
    load_template,
)
from exp_utils import (
    SLEAP_HZ,
    estimate_temporal_offset,
    load_session_data,
    run_template_matching,
    smooth_keypoints,
)
from skeleton import normalize_skeleton_batch, project_to_pcs
from config import NODES, NODE_IDX

FIGS_DIR = ROOT / "results" / "figures" / "harm_analysis"
METRICS_DIR = ROOT / "results" / "metrics"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Mirror v3 conventions
RAT_CONFIG = {
    "R1": {
        "primary":   {"template_file": "R1_template_2.npz", "bounds": 1.0},
        "secondary": {"template_file": "R1_template_1.npz", "bounds": 1.5},
    },
    "R2": {"primary": {"template_file": "R2_template_1.npz", "bounds": 1.0}},
    "R3": {"primary": {"template_file": "R3_template_1.npz", "bounds": 1.0}},
}

WIN = 30
SL_SMOOTH = ("median", 11)
DN_SMOOTH = ("median", 11)
PAIR_TOL_FRAMES = int(round(0.3 * SLEAP_HZ))  # 300 ms ≈ 6 SLEAP frames


# ─────────────────────────────────────────────────────────────────────────────
# Match-set partition: SLEAP-only / DANNCE-only / both, in DANNCE PC space
# ─────────────────────────────────────────────────────────────────────────────

def partition_matches(sleap_matches, dannce_matches, tol_frames=PAIR_TOL_FRAMES):
    """
    Greedy nearest-neighbour pairing: a SLEAP match is "both" if it has a DANNCE
    match within tol_frames. Returns three sets of frame indices (SLEAP-frame
    coordinates — both arrays already share the SLEAP time axis after
    load_session_data resamples DANNCE).
    """
    sl = np.asarray(sleap_matches, dtype=int)
    dn = np.asarray(dannce_matches, dtype=int)
    if len(sl) == 0 or len(dn) == 0:
        return sl.tolist(), [], dn.tolist()

    # Greedy: nearest unused DANNCE match per SLEAP match
    used = np.zeros(len(dn), dtype=bool)
    both_sl, both_dn, sl_only = [], [], []
    for s in sl:
        diffs = np.abs(dn - s).astype(float)
        diffs[used] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= tol_frames:
            used[j] = True
            both_sl.append(int(s))
            both_dn.append(int(dn[j]))
        else:
            sl_only.append(int(s))
    dn_only = [int(d) for d, u in zip(dn, used) if not u]
    return sl_only, dn_only, both_sl, both_dn


# ─────────────────────────────────────────────────────────────────────────────
# Mechanism 1: per-PC bias decomposition
# ─────────────────────────────────────────────────────────────────────────────

def per_pc_bias(sleap_pc, dannce_pc):
    """Decompose SLEAP-DANNCE PC error into mean (bias) and std (noise) per PC."""
    err = sleap_pc - dannce_pc
    mu = err.mean(axis=0)
    sd = err.std(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    bias_frac = np.where(rmse > 0, np.abs(mu) / rmse, 0.0)
    return dict(mu=mu, sd=sd, rmse=rmse, bias_fraction=bias_frac)


# ─────────────────────────────────────────────────────────────────────────────
# Mechanism 2: per-keypoint constant offset (post egocentric normalization)
# ─────────────────────────────────────────────────────────────────────────────

def per_keypoint_bias(sleap_rot, dannce_rot):
    """
    After egocentric normalization, mean displacement of each keypoint per axis.
    A non-zero mean in any (keypoint, axis) cell means that error is NOT noise —
    it's a fixed offset that flows linearly into PC space.

    Returns per-keypoint scalar |bias| and RMSE (Euclidean across xyz) so the
    bias_fraction is a single number per keypoint.
    """
    diff = sleap_rot - dannce_rot         # (T, 23, 3)
    mu = diff.mean(axis=0)                # (23, 3) — bias vector per keypoint
    sd = diff.std(axis=0)                 # (23, 3)
    bias_norm = np.linalg.norm(mu, axis=1)                       # (23,)
    rmse = np.sqrt((np.linalg.norm(diff, axis=2) ** 2).mean(axis=0))  # (23,)
    bias_frac = np.where(rmse > 0, bias_norm / rmse, 0.0)        # (23,)
    return dict(per_kp_mean=mu, per_kp_std=sd, per_kp_rmse=rmse,
                per_kp_bias_norm=bias_norm,
                per_kp_bias_fraction=bias_frac)


# ─────────────────────────────────────────────────────────────────────────────
# Pose-distribution analysis at event times (in DANNCE PC space)
# ─────────────────────────────────────────────────────────────────────────────

def windowed_features(pc, frames, win=WIN):
    """For each frame index, return the (win, n_pcs) window ending at that frame."""
    out = []
    for f in frames:
        s = f - win + 1
        if s < 0 or f >= pc.shape[0]:
            continue
        out.append(pc[s:f + 1])
    return np.array(out) if out else np.zeros((0, win, pc.shape[1]))


def pose_distribution_test(both_dn_pcs, sl_only_dn_pcs, dn_only_dn_pcs):
    """
    Compare the DISTRIBUTION of DANNCE PC values at:
      - both events (consensus reward)
      - SLEAP-only events (false-positive rewards from DANNCE's perspective)
      - DANNCE-only events (would-be rewards SLEAP missed)

    Returns per-PC KS statistics (sl_only vs both) and per-PC centroid offsets.
    A small KS stat + small centroid offset => SLEAP-only events draw poses
    similar in DANNCE-space to consensus events (systematic, survivable).
    A large KS stat or large centroid offset => SLEAP-only events sample a
    different region (random or biased target).
    """
    out = {"per_pc_ks_pvalue": [], "per_pc_centroid_offset": [],
           "per_pc_both_mean": [], "per_pc_sl_only_mean": []}
    if both_dn_pcs.size == 0 or sl_only_dn_pcs.size == 0:
        return out

    # Use the END-of-window pose (frame at the trigger) — most diagnostic
    both_end = both_dn_pcs[:, -1, :]          # (n_both, n_pcs)
    sl_only_end = sl_only_dn_pcs[:, -1, :]    # (n_sl_only, n_pcs)

    n_pcs = both_end.shape[1]
    for j in range(n_pcs):
        if len(sl_only_end) >= 5 and len(both_end) >= 5:
            stat, p = ks_2samp(both_end[:, j], sl_only_end[:, j])
        else:
            stat, p = np.nan, np.nan
        out["per_pc_ks_pvalue"].append(float(p) if not np.isnan(p) else np.nan)
        out["per_pc_both_mean"].append(float(both_end[:, j].mean()))
        out["per_pc_sl_only_mean"].append(float(sl_only_end[:, j].mean()))
        out["per_pc_centroid_offset"].append(
            float(sl_only_end[:, j].mean() - both_end[:, j].mean()))
    return out


def template_distance_distribution(both_dn_pcs, sl_only_dn_pcs, dn_only_dn_pcs,
                                   template):
    """
    Mahalanobis-like distance from each event window to the template.
    If SLEAP-only events sit roughly at the same distance from the template
    (in DANNCE PC space) as both-events, they are systematically off by a
    consistent amount but still in the right neighbourhood — survivable.
    """
    def d_to_tmpl(arr):
        if arr.size == 0:
            return np.array([])
        # Mean squared deviation per event window, summed over PC and time
        return np.sqrt(((arr - template[None, :, :]) ** 2).mean(axis=(1, 2)))

    return {
        "both": d_to_tmpl(both_dn_pcs),
        "sl_only": d_to_tmpl(sl_only_dn_pcs),
        "dn_only": d_to_tmpl(dn_only_dn_pcs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Behavior-state correlation
# ─────────────────────────────────────────────────────────────────────────────

def behavior_state_at_events(dannce_3d, frames):
    """At each event frame, compute COM speed and snout height from DANNCE."""
    frames = np.array(frames, dtype=int)
    if len(frames) == 0:
        return dict(com_speed=np.array([]), snout_z=np.array([]))
    com = dannce_3d.mean(axis=1)               # (T, 3)
    com_speed_full = np.linalg.norm(np.diff(com, axis=0), axis=1) * SLEAP_HZ
    com_speed_full = np.concatenate([[0], com_speed_full])
    snout_z = dannce_3d[:, NODE_IDX["Snout"], 2]
    valid = (frames >= 0) & (frames < len(com_speed_full))
    f = frames[valid]
    return dict(com_speed=com_speed_full[f], snout_z=snout_z[f])


# ─────────────────────────────────────────────────────────────────────────────
# Reward-rate impact (uses ratBoops if available)
# ─────────────────────────────────────────────────────────────────────────────

def reward_metrics(rat, session):
    """Return (n_tones, n_reward_starts, n_completions, reward_rate_per_min)."""
    try:
        ev = load_opcon_events(rat, session)
    except Exception:
        return None
    duration_min = float(np.ptp(ev["sleap_frame_times_ms"])) / 60_000.0 if len(
        ev["sleap_frame_times_ms"]) > 1 else np.nan
    return dict(
        n_tones=int(len(ev["tone_times_ms"])),
        n_reward_starts=int(len(ev["reward_start_times_ms"])),
        n_completions=int(len(ev["reward_end_times_ms"])),
        n_window_expired=int(len(ev["reward_window_end_times_ms"])),
        duration_min=duration_min,
        reward_rate_per_min=(len(ev["reward_start_times_ms"]) / duration_min
                             if duration_min and duration_min > 0 else np.nan),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-session orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_session(rat, session, t, bounds_scalar, fig_dir):
    """
    Returns a dict of per-session metrics OR a dict with 'error' set.
    Also writes diagnostic figures to fig_dir.
    """
    pcu = t["pcs_to_use"].ravel().astype(int)
    pw = t["pc_weights"]
    fm = t["feature_means"]
    xyz_stds = t["feature_stds"][pcu]
    tmpl = t["template"][:, pcu]                       # (WIN, n_pcs_used)

    sleap_3d, dannce_3d, aligned = load_session_data(rat, session)
    if aligned is None:
        return {"rat": rat, "session": session, "error": "no aligned"}
    st = np.array(aligned["sleap_times_ms"]).ravel()

    sl_sm = smooth_keypoints(sleap_3d, *SL_SMOOTH)
    dn_sm = smooth_keypoints(dannce_3d, *DN_SMOOTH)
    sl_rot, _, _ = normalize_skeleton_batch(sl_sm)
    dn_rot, _, _ = normalize_skeleton_batch(dn_sm)
    sl_pc = project_to_pcs(sl_rot, pw, fm)[:, pcu]
    dn_pc = project_to_pcs(dn_rot, pw, fm)[:, pcu]

    n_pcs = sl_pc.shape[1]

    # ---- Mechanism 1 & 2: per-PC bias and per-keypoint bias ----
    pc_bias = per_pc_bias(sl_pc, dn_pc)
    kp_bias = per_keypoint_bias(sl_rot, dn_rot)

    # ---- Match sets ----
    bounds = np.tile(xyz_stds * bounds_scalar, (WIN, 1))
    sl_m = run_template_matching(sl_pc, tmpl, bounds, max_outside=3)
    dn_m = run_template_matching(dn_pc, tmpl, bounds, max_outside=3)

    # Estimate temporal offset (SLEAP detects later) and shift SLEAP matches
    # for the partition. We already know offsets are typically ~500ms.
    if len(sl_m) >= 2 and len(dn_m) >= 2:
        offset_ms = estimate_temporal_offset(sl_m, dn_m, st, st)
    else:
        offset_ms = 0.0
    sl_m_aligned = [int(round(s - offset_ms / 1000.0 * SLEAP_HZ)) for s in sl_m]
    sl_m_aligned = [s for s in sl_m_aligned if 0 <= s < sl_pc.shape[0]]

    sl_only, dn_only, both_sl, both_dn = partition_matches(
        sl_m_aligned, dn_m, tol_frames=PAIR_TOL_FRAMES)

    # ---- Pose-distribution comparison in DANNCE PC space ----
    # For "both" we use the DANNCE-side timing; for sl_only we still use
    # SLEAP timing because there is no DANNCE counterpart.
    both_w = windowed_features(dn_pc, both_dn)
    sl_only_w = windowed_features(dn_pc, sl_only)
    dn_only_w = windowed_features(dn_pc, dn_only)

    pose_test = pose_distribution_test(both_w, sl_only_w, dn_only_w)
    tmpl_dist = template_distance_distribution(both_w, sl_only_w, dn_only_w, tmpl)

    # ---- Behavior state at event times ----
    state_both = behavior_state_at_events(dannce_3d, both_dn)
    state_sl_only = behavior_state_at_events(dannce_3d, sl_only)
    state_dn_only = behavior_state_at_events(dannce_3d, dn_only)

    # ---- Reward metrics ----
    rew = reward_metrics(rat, session) or {}

    # ---- Precision / recall at 300 ms ----
    n_both = len(both_sl)
    n_sl = len(sl_m)
    n_dn = len(dn_m)
    recall = n_both / n_dn if n_dn else 0.0
    precision = n_both / n_sl if n_sl else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)

    # ---- Plots ----
    _plot_session(fig_dir, rat, session, sl_pc, dn_pc, tmpl, pc_bias, kp_bias,
                  both_dn, sl_only, dn_only, tmpl_dist, state_both,
                  state_sl_only, state_dn_only, n_pcs)

    out = dict(
        rat=rat, session=session,
        n_dannce=n_dn, n_sleap=n_sl, n_both=n_both,
        n_sleap_only=len(sl_only), n_dannce_only=len(dn_only),
        recall=recall, precision=precision, f1=f1,
        temporal_offset_ms=float(offset_ms),
        # Mechanism summaries (compact scalars)
        pc_bias_fraction_max=float(np.max(pc_bias["bias_fraction"])),
        pc_bias_fraction_mean=float(np.mean(pc_bias["bias_fraction"])),
        kp_bias_fraction_max=float(np.max(kp_bias["per_kp_bias_fraction"])),
        kp_bias_fraction_mean=float(np.mean(kp_bias["per_kp_bias_fraction"])),
        # Distribution-test summary
        ks_pvalue_min=float(np.nanmin(pose_test["per_pc_ks_pvalue"]))
            if pose_test["per_pc_ks_pvalue"] else np.nan,
        centroid_offset_max=float(np.nanmax(np.abs(pose_test["per_pc_centroid_offset"])))
            if pose_test["per_pc_centroid_offset"] else np.nan,
        # Distance-to-template summary (median)
        dist_to_tmpl_both=float(np.median(tmpl_dist["both"]))
            if tmpl_dist["both"].size else np.nan,
        dist_to_tmpl_sl_only=float(np.median(tmpl_dist["sl_only"]))
            if tmpl_dist["sl_only"].size else np.nan,
        dist_to_tmpl_dn_only=float(np.median(tmpl_dist["dn_only"]))
            if tmpl_dist["dn_only"].size else np.nan,
        # Behavior state at events
        com_speed_both_mean=float(np.mean(state_both["com_speed"]))
            if state_both["com_speed"].size else np.nan,
        com_speed_sl_only_mean=float(np.mean(state_sl_only["com_speed"]))
            if state_sl_only["com_speed"].size else np.nan,
        snout_z_both_mean=float(np.mean(state_both["snout_z"]))
            if state_both["snout_z"].size else np.nan,
        snout_z_sl_only_mean=float(np.mean(state_sl_only["snout_z"]))
            if state_sl_only["snout_z"].size else np.nan,
        **{f"reward_{k}": v for k, v in rew.items()},
    )

    # Per-PC details to a JSON sidecar (so the CSV stays flat)
    detail = dict(
        pc_bias_mu=pc_bias["mu"].tolist(),
        pc_bias_rmse=pc_bias["rmse"].tolist(),
        pc_bias_fraction=pc_bias["bias_fraction"].tolist(),
        per_kp_bias_fraction=kp_bias["per_kp_bias_fraction"].tolist(),
        per_kp_mean_norm=kp_bias["per_kp_bias_norm"].tolist(),
        per_kp_node=NODES,
        pose_per_pc_ks_pvalue=pose_test["per_pc_ks_pvalue"],
        pose_per_pc_centroid_offset=pose_test["per_pc_centroid_offset"],
        pose_per_pc_both_mean=pose_test["per_pc_both_mean"],
        pose_per_pc_sl_only_mean=pose_test["per_pc_sl_only_mean"],
    )
    with open(fig_dir / "detail.json", "w") as f:
        json.dump(detail, f, indent=2)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_session(fig_dir, rat, session, sl_pc, dn_pc, tmpl, pc_bias, kp_bias,
                  both_dn, sl_only, dn_only, tmpl_dist, state_both,
                  state_sl_only, state_dn_only, n_pcs):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-PC bias decomposition
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(n_pcs)
    w = 0.4
    ax.bar(x - w / 2, np.abs(pc_bias["mu"]), w, label="|mean error|  (bias)",
           color="firebrick")
    ax.bar(x + w / 2, pc_bias["sd"], w, label="std error  (noise)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i + 1}" for i in x])
    ax.set_ylabel("(SLEAP − DANNCE) error in PC space")
    ax.set_title(f"{rat}/{session} — per-PC bias vs noise"
                 f"\nbias fraction: " + ", ".join(
                     [f"PC{i + 1}={pc_bias['bias_fraction'][i]:.2f}" for i in x]))
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "01_per_pc_bias.png", dpi=120)
    plt.close(fig)

    # 2) Per-keypoint bias map
    fig, ax = plt.subplots(figsize=(12, 4))
    bf = kp_bias["per_kp_bias_fraction"]
    colors = ["firebrick" if v > 0.5 else ("orange" if v > 0.3 else "steelblue")
              for v in bf]
    ax.bar(np.arange(len(bf)), bf, color=colors)
    ax.axhline(0.5, color="k", ls="--", alpha=0.5,
               label="bias > noise  (>0.5 ⇒ structured)")
    ax.set_xticks(np.arange(len(bf)))
    ax.set_xticklabels(NODES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|bias| / RMSE per keypoint")
    ax.set_title(f"{rat}/{session} — per-keypoint bias fraction "
                 "(after egocentric normalization)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "02_per_keypoint_bias.png", dpi=120)
    plt.close(fig)

    # 3) Match-set partition in DANNCE PC space (PC1 vs PC2)
    if n_pcs >= 2 and len(both_dn) + len(sl_only) + len(dn_only) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(dn_pc[::20, 0], dn_pc[::20, 1], s=2, color="lightgrey",
                   alpha=0.4, label="all frames (subsampled)")
        if len(both_dn):
            ax.scatter(dn_pc[both_dn, 0], dn_pc[both_dn, 1], s=30,
                       color="forestgreen", label=f"both (n={len(both_dn)})",
                       edgecolor="k", lw=0.3)
        if len(sl_only):
            ax.scatter(dn_pc[sl_only, 0], dn_pc[sl_only, 1], s=30,
                       color="firebrick", marker="^",
                       label=f"SLEAP-only (n={len(sl_only)})",
                       edgecolor="k", lw=0.3)
        if len(dn_only):
            ax.scatter(dn_pc[dn_only, 0], dn_pc[dn_only, 1], s=30,
                       color="royalblue", marker="s",
                       label=f"DANNCE-only (n={len(dn_only)})",
                       edgecolor="k", lw=0.3)
        ax.scatter(tmpl[-1, 0], tmpl[-1, 1] if n_pcs >= 2 else 0, s=200,
                   color="gold", marker="*", edgecolor="k", lw=1,
                   label="template end")
        ax.set_xlabel("DANNCE PC1")
        ax.set_ylabel("DANNCE PC2")
        ax.set_title(f"{rat}/{session} — events in DANNCE PC space")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "03_event_pose_distribution.png", dpi=120)
        plt.close(fig)

    # 4) Distance-to-template distributions
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, max(
        [tmpl_dist[k].max() if tmpl_dist[k].size else 1.0
         for k in ("both", "sl_only", "dn_only")] + [1.0]), 30)
    if tmpl_dist["both"].size:
        ax.hist(tmpl_dist["both"], bins=bins, alpha=0.6, color="forestgreen",
                label=f"both (n={tmpl_dist['both'].size})")
    if tmpl_dist["sl_only"].size:
        ax.hist(tmpl_dist["sl_only"], bins=bins, alpha=0.6, color="firebrick",
                label=f"SLEAP-only (n={tmpl_dist['sl_only'].size})")
    if tmpl_dist["dn_only"].size:
        ax.hist(tmpl_dist["dn_only"], bins=bins, alpha=0.6, color="royalblue",
                label=f"DANNCE-only (n={tmpl_dist['dn_only'].size})")
    ax.set_xlabel("RMSE to template (DANNCE PC space)")
    ax.set_ylabel("Count")
    ax.set_title(f"{rat}/{session} — how close is each event class to template?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "04_distance_to_template.png", dpi=120)
    plt.close(fig)

    # 5) Behavior state at events: COM speed and snout height
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(
            axes, ("com_speed", "snout_z"),
            ("COM speed (mm/s)", "Snout z (calibration units)")):
        groups = {
            "both": state_both[key],
            "SLEAP-only": state_sl_only[key],
            "DANNCE-only": state_dn_only[key],
        }
        data = [v for v in groups.values() if v.size]
        labels = [k for k, v in groups.items() if v.size]
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel(label)
    fig.suptitle(f"{rat}/{session} — behavior state at event times (from DANNCE)")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_behavior_state.png", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Across-session aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_summary(summary_csv):
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    # Drop error rows so subsequent column lookups don't fail
    df = df[df.get("error", pd.Series(index=df.index)).isna()] if "error" in df.columns else df
    if df.empty:
        print(f"  aggregate_summary: no successful sessions in {summary_csv.name}")
        return
    if "dist_to_tmpl_sl_only" not in df.columns:
        print(f"  aggregate_summary: required columns missing in {summary_csv.name}")
        return
    # Centroid stability: per-PC SLEAP-only mean across sessions
    # The summary_csv stem is "<rat>_<config>_harm_summary" — strip the suffix
    # to get the session-figure-folder prefix used by run_session().
    detail_root = summary_csv.parent.parent / "figures" / "harm_analysis"
    folder_prefix = summary_csv.stem.replace("_harm_summary", "")
    detail_dir = detail_root / folder_prefix
    out_dir = detail_root / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot recall vs precision colored by SLEAP/DANNCE-pose dist gap
    fig, ax = plt.subplots(figsize=(6, 5))
    gap = df["dist_to_tmpl_sl_only"] - df["dist_to_tmpl_both"]
    sc = ax.scatter(df["recall"], df["precision"], c=gap, cmap="coolwarm",
                    s=40, edgecolor="k", lw=0.3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("median(SLEAP-only dist − both dist) to template")
    ax.set_xlabel("Recall (300 ms tol)")
    ax.set_ylabel("Precision (300 ms tol)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(0.9, color="red", ls=":", alpha=0.4)
    ax.axvline(0.9, color="red", ls=":", alpha=0.4)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", alpha=0.4)
    ax.set_title(f"{summary_csv.stem} — per-session recall vs precision")
    fig.tight_layout()
    fig.savefig(out_dir / f"{summary_csv.stem}_recall_precision.png", dpi=120)
    plt.close(fig)

    # Centroid offset stability across sessions: load detail JSONs
    centroids = []
    sessions = []
    for _, row in df.iterrows():
        d = detail_dir / row["session"] / "detail.json"
        if d.exists():
            with open(d) as f:
                jd = json.load(f)
            if jd.get("pose_per_pc_centroid_offset"):
                centroids.append(jd["pose_per_pc_centroid_offset"])
                sessions.append(row["session"])

    if centroids:
        C = np.array(centroids)  # (n_sessions, n_pcs)
        fig, ax = plt.subplots(figsize=(8, 5))
        for j in range(C.shape[1]):
            ax.plot(C[:, j], "o-", label=f"PC{j + 1}", alpha=0.7)
        ax.axhline(0, color="k", ls="--", alpha=0.5)
        ax.set_xlabel("Session (chronological order in CSV)")
        ax.set_ylabel("SLEAP-only centroid − both centroid (DANNCE PC)")
        ax.set_title(f"{summary_csv.stem} — centroid stability across sessions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{summary_csv.stem}_centroid_stability.png", dpi=120)
        plt.close(fig)

    # Reward-rate vs alignment quality
    if "reward_reward_rate_per_min" in df.columns and df[
            "reward_reward_rate_per_min"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["f1"], df["reward_reward_rate_per_min"], s=40,
                   alpha=0.7, edgecolor="k", lw=0.3)
        ax.set_xlabel("SLEAP/DANNCE F1 (300 ms)")
        ax.set_ylabel("Reward rate per min")
        ax.set_title(f"{summary_csv.stem} — does disagreement hurt reward rate?")
        fig.tight_layout()
        fig.savefig(out_dir / f"{summary_csv.stem}_reward_rate.png", dpi=120)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_rat(rat, config_key="primary", max_sessions=None):
    cfg = RAT_CONFIG[rat][config_key]
    template_file = cfg["template_file"]
    bounds_scalar = cfg["bounds"]
    t = dict(load_template(rat, template_file))
    sessions = get_sessions(rat=rat)["session"].tolist()
    if max_sessions:
        sessions = sessions[:max_sessions]

    out_csv = METRICS_DIR / f"{rat}_{config_key}_harm_summary.csv"
    rows = []
    t0 = time.time()
    for i, session in enumerate(sessions):
        fd = FIGS_DIR / f"{rat}_{config_key}" / session
        try:
            t1 = time.time()
            row = run_session(rat, session, t, bounds_scalar, fd)
            print(f"  [{i+1}/{len(sessions)}] {session}: "
                  f"recall={row.get('recall', float('nan')):.2f} "
                  f"precision={row.get('precision', float('nan')):.2f} "
                  f"({time.time()-t1:.1f}s)")
            rows.append(row)
            # Incremental save so we can check progress
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        except Exception as e:
            traceback.print_exc()
            rows.append({"rat": rat, "session": session, "error": str(e)})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n{rat} {config_key}: {len(rows)} sessions, "
          f"saved to {out_csv} in {time.time()-t0:.1f}s")
    aggregate_summary(out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rat", required=True,
                        choices=["R1", "R2", "R3", "all"])
    parser.add_argument("--config", default="primary",
                        choices=["primary", "secondary"])
    parser.add_argument("--max_sessions", type=int, default=None)
    args = parser.parse_args()

    if args.rat == "all":
        for rat, cfg in [("R1", "primary"), ("R1", "secondary"),
                         ("R2", "primary"), ("R3", "primary")]:
            run_rat(rat, cfg, args.max_sessions)
    else:
        run_rat(args.rat, args.config, args.max_sessions)


if __name__ == "__main__":
    main()
