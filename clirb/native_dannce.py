"""Native (~50 Hz) DANNCE-space windows + reward-frame mapping.

Built for session_variability.ipynb and scripts/render_session_variability_video.py.

Rears are still detected and aligned in the validated 20 Hz (SLEAP-rate) domain
via clirb.events (that's what the online template-match rule and its 30-frame
template are calibrated for). This module maps the resulting per-rear anchor
frame to the native ~50 Hz DANNCE stream via dannce_idx_for_sleap_cams, then
extracts a wider +/-2s window directly from full-rate DANNCE keypoints for
downstream variability computation and visualization -- the analysis lives at
native temporal resolution even though detection/alignment does not.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT  # noqa: E402
from skeleton import normalize_skeleton_batch, project_to_pcs  # noqa: E402
from clirb.events import (causal_median_5_kp, project_session_to_pcs, detect_rears,  # noqa: E402
                          best_align_end, replay_rule_offline, rear_matched, TEMPLATE_LEN)
from clirb import templates as tmpl  # noqa: E402
import session_cache as sc  # noqa: E402
from data_io import load_aligned_data  # noqa: E402

DANNCE_FPS = 50.0
CAUSAL_MEDIAN_PAD = 4  # lead-in frames the 5-frame causal median needs


# --------------------------------------------------------------------------- #
# Reward events -> native DANNCE frames
# --------------------------------------------------------------------------- #
def reward_sleap_rows(rat, session, data_root=DATA_ROOT):
    """behavior_log.csv 'frame' (triggered==1, processed_frame domain) -> SLEAP
    /triang-array row indices (0..n_sleap-1), via frame_mapping.csv.

    GOTCHA: frame_mapping.csv has one row per PROCESSED frame -- fewer rows
    than n_sleap, since dropped frames are skipped entirely. `processed_frame`
    is a dense 1..N counter identifying the ROW; that row's `cam0_frame`
    column (NOT the row position itself) gives the actual triang/SLEAP array
    row. Verified on real data: using row position instead of cam0_frame is
    off by thousands of frames (~minutes) for late-session rewards, because
    frame_mapping.csv accumulates more dropped frames over the session.
    """
    sleap_dir = os.path.join(data_root, rat, session, "sleap")
    bl = pd.read_csv(os.path.join(sleap_dir, "behavior_log.csv"),
                      skipinitialspace=True)
    trig = bl.loc[bl["triggered"] == 1, "frame"].to_numpy(dtype=int)
    if not len(trig):
        return np.array([], dtype=int)
    fm = pd.read_csv(os.path.join(sleap_dir, "frame_mapping.csv"),
                      skipinitialspace=True)
    proc = fm["processed_frame"].to_numpy(dtype=int)
    cam0 = fm["cam0_frame"].to_numpy(dtype=int)
    rows = np.searchsorted(proc, trig)
    valid = (rows < len(proc)) & (proc[np.clip(rows, 0, len(proc) - 1)] == trig)
    if not valid.all():
        print(f"  {rat}/{session}: {int((~valid).sum())}/{len(trig)} triggered "
              f"frames not found in frame_mapping.csv (dropped before logging?)")
    return cam0[rows[valid]]


def reward_native_frames(rat, session, dannce_idx_for_sleap_cams, data_root=DATA_ROOT):
    """Reward events (same source as single_session.ipynb: behavior_log.csv
    triggered==1) -> native (~50 Hz) DANNCE frame indices."""
    sleap_rows = reward_sleap_rows(rat, session, data_root=data_root)
    return dannce_idx_for_sleap_cams[sleap_rows]


# --------------------------------------------------------------------------- #
# Native-rate windows
# --------------------------------------------------------------------------- #
def extract_native_window(dn_native, center_frame, template,
                          half_win_frames=100, median_pad=CAUSAL_MEDIAN_PAD):
    """+/-half_win_frames window around `center_frame` in the native DANNCE
    stream, projected through the session's own template PC weights.

    Returns None if the padded window would run off either edge of the
    session (excluded rather than clipped, per project convention).

    Returns dict:
      frames : (W,) int      -- native frame indices, W = 2*half_win_frames+1
      world  : (W, 23, 3)    -- raw world-frame keypoints (for arena/video)
      pcs    : (W, n_pcs)    -- egocentric + 5-frame causal median, projected

    The 5-frame causal median is computed with `median_pad` REAL leading
    frames (not the isolated-slice repeat-padding causal_median_5_kp would
    apply on its own), so the very first output frame is not artificially
    smoothed against itself.
    """
    lo, hi = center_frame - half_win_frames, center_frame + half_win_frames
    lo_pad = lo - median_pad
    if lo_pad < 0 or hi >= dn_native.shape[0]:
        return None
    world = dn_native[lo:hi + 1]
    buf = dn_native[lo_pad:hi + 1].astype(np.float32)
    rotated, _, _ = normalize_skeleton_batch(buf)
    rotated_med = causal_median_5_kp(rotated)[median_pad:]
    pcs = project_to_pcs(rotated_med, template["pc_weights"], template["feature_means"])
    return {"frames": np.arange(lo, hi + 1), "world": world, "pcs": pcs,
            "rotated_med": rotated_med}


# --------------------------------------------------------------------------- #
# Sliding-window variability
# --------------------------------------------------------------------------- #
def _window_stack(pc_windows, index, lookback=4):
    """pc_windows: list of (W, n_pcs) arrays, one per VALID (in-bounds) rear,
    in chronological order. index: position of the target rear in that list.

    Shrinking window: uses whatever's available up to `lookback` previous
    rears (minimum 1 = the rear alone). Boundary-excluded rears never enter
    `pc_windows` at all, so they neither get a variability value nor count
    toward any other rear's window -- the window is over the valid
    subsequence, not over raw rear index.

    Returns (n_in_window, W, n_pcs).
    """
    lo = max(0, index - lookback)
    return np.stack(pc_windows[lo:index + 1], axis=0)


def sliding_window_variability(pc_windows, index, lookback=4):
    """Scalar per PC: mean-over-time std-across-rears. See `_window_stack`."""
    std_t = _window_stack(pc_windows, index, lookback=lookback).std(axis=0)  # (W, n_pcs)
    return std_t.mean(axis=0)


def sliding_window_mean_std(pc_windows, index, lookback=4):
    """Full time-course mean/std (not collapsed to a scalar). Same window as
    `sliding_window_variability`. Returns (mean, std), each (W, n_pcs)."""
    stack = _window_stack(pc_windows, index, lookback=lookback)
    return stack.mean(axis=0), stack.std(axis=0)


def group_variability(pc_windows_subset):
    """Variability of a fixed, arbitrary GROUP of rears (not a rolling
    window): stack the given (W, n_pcs) trajectories, take the std across
    them at each timepoint, then average over time. Same math as
    `sliding_window_variability`, generalized to any subset -- used for
    "the first 10 matches" / "the last 10 matches" style comparisons where
    the group is fixed by position-in-session, not by a rolling lookback.
    """
    stack = np.stack(pc_windows_subset, axis=0)
    return stack.std(axis=0).mean(axis=0)


def first_last_match_variability(d, n=10, min_matches=20):
    """First-n vs last-n MATCHED (`rear_is_matched_valid`, standard bounds)
    rears in a session, by chronological position among valid rears.

    `d` is a dict from `build_session_variability`. Returns None if the
    session has fewer than `min_matches` matched (and valid) rears -- per
    the project convention of excluding rather than padding/clipping.

    Returns dict(n_matches, first, last), where `first`/`last` are (n_pcs,)
    group-variability arrays (see `group_variability`).
    """
    matched_idx = np.where(d["rear_is_matched_valid"])[0]
    if len(matched_idx) < min_matches:
        return None
    first_idx, last_idx = matched_idx[:n], matched_idx[-n:]
    pcs_windows = d["pcs_windows"]
    return dict(
        n_matches=len(matched_idx),
        first=group_variability([pcs_windows[i] for i in first_idx]),
        last=group_variability([pcs_windows[i] for i in last_idx]),
    )


# --------------------------------------------------------------------------- #
# Reward presence/absence labeling
# --------------------------------------------------------------------------- #
def label_rears_by_reward(anchor_native, reward_native, tol_frames=50):
    """Label each (chronologically-ordered, valid) rear by its relationship
    to reward events, both in native (~50 Hz) frames.

    `tol_frames` (default 50 = 1s @ 50Hz) is the tolerance for calling a rear
    itself "rewarded": the rear's own alignment anchor and the reward's
    trigger frame refer to the same physical match-completion moment via two
    different paths (offline replay vs. the online rule's own log), so they
    should land within a frame or two of each other, not exactly on it.

    Returns
    -------
    is_rewarded : (n,) bool -- a reward landed within `tol_frames` of this
        rear's anchor.
    rears_since_reward : (n,) int -- 0 for a rewarded rear itself, 1 for the
        next rear after the most recent reward, ...; -1 before any reward
        has occurred yet.
    frames_since_reward : (n,) float -- native frames since the most recent
        reward at or before this rear's anchor; NaN before any reward.
    """
    anchor_native = np.asarray(anchor_native)
    reward_native = np.sort(np.asarray(reward_native))
    n = len(anchor_native)
    is_rewarded = np.zeros(n, dtype=bool)
    if len(reward_native):
        for i, a in enumerate(anchor_native):
            is_rewarded[i] = bool(np.any(np.abs(reward_native - a) <= tol_frames))

    rears_since_reward = np.full(n, -1, dtype=int)
    frames_since_reward = np.full(n, np.nan)
    counter = -1
    last_reward_frame = None
    for i, a in enumerate(anchor_native):
        if is_rewarded[i]:
            counter = 0
            last_reward_frame = a
        elif counter >= 0:
            counter += 1
        rears_since_reward[i] = counter
        if last_reward_frame is not None:
            frames_since_reward[i] = a - last_reward_frame
    return is_rewarded, rears_since_reward, frames_since_reward


# --------------------------------------------------------------------------- #
# Consolidated per-session pipeline (used by batch/cross-session analyses)
# --------------------------------------------------------------------------- #
def build_session_variability(rat, session, half_win_s=2.0, lookback=4,
                              accepted_errors=3, reward_tol_s=1.0,
                              template_like_bound_mult=1.25,
                              n_pcs=3, data_root=DATA_ROOT):
    """Run the full rear-detection -> native-window -> variability -> reward-
    labeling pipeline for one session. Raises if the session lacks dual
    SLEAP+DANNCE data or a behavior log. Returns a dict of per-valid-rear
    arrays (chronological order) plus session metadata -- no video/plotting.

    `is_template_like` (recommendation #2: separate "rewarded" from "merely
    close to the template") is a SEPARATE, more permissive replay of the
    same matching rule with `bounds_to_use` widened by
    `template_like_bound_mult` (default 1.25x) at the same `accepted_errors`
    -- a strict superset of `rear_is_matched_valid` (standard bounds), since
    widening the band only ever admits more windows.
    """
    half_win_frames = int(round(half_win_s * DANNCE_FPS))
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw is None or not raw["has_dual"]:
        raise ValueError(f"{rat}/{session}: no dual SLEAP+DANNCE data cached")
    if not raw["has_task_log"]:
        raise ValueError(f"{rat}/{session}: no behavior_log")

    dn_native = raw["dannce_keys_3D"]
    dn_native = (dn_native.squeeze(1).transpose(0, 2, 1) if dn_native.ndim == 4
                else np.transpose(dn_native, [0, 2, 1]))
    di = raw["dannce_idx_for_sleap_cams"].astype(int).ravel()[1:]
    dn_at_sleap = dn_native[di]

    template = tmpl.load_template(rat, session)
    pcs_at_sleap, _ = project_session_to_pcs(dn_at_sleap, template, flip_z=False)
    rears = detect_rears(pcs_at_sleap[:, 0])
    rear_start = np.array([r[0] for r in rears])
    rear_end = np.array([r[1] for r in rears])

    match_frames, _ = replay_rule_offline(pcs_at_sleap, template, accepted_errors=accepted_errors)
    rear_is_matched, _ = rear_matched(rear_start, rear_end, match_frames, template_len=TEMPLATE_LEN)

    wide_template = dict(template)
    wide_template["bounds_to_use"] = template["bounds_to_use"] * template_like_bound_mult
    wide_match_frames, _ = replay_rule_offline(pcs_at_sleap, wide_template,
                                               accepted_errors=accepted_errors)
    rear_is_template_like, _ = rear_matched(rear_start, rear_end, wide_match_frames,
                                            template_len=TEMPLATE_LEN)

    pcs_windows, valid_rear_idx, anchor_native = [], [], []
    for k, (rs, re_, _n_above) in enumerate(rears):
        end20, _mse = best_align_end(rs, re_, pcs_at_sleap, template["template"][:, :2])
        if end20 < 0:
            continue
        a_native = int(di[end20])
        w = extract_native_window(dn_native, a_native, template, half_win_frames=half_win_frames)
        if w is None:
            continue  # +/-half_win_s window runs off the session boundary
        pcs_windows.append(w["pcs"][:, :n_pcs])
        valid_rear_idx.append(k)
        anchor_native.append(a_native)
    anchor_native = np.array(anchor_native, dtype=int)
    n_valid = len(pcs_windows)

    variability = np.array([sliding_window_variability(pcs_windows, i, lookback=lookback)
                            for i in range(n_valid)]) if n_valid else np.zeros((0, n_pcs))

    aligned = load_aligned_data(rat, session)
    dannce_times_ms = aligned["dannce_times_ms"].ravel()
    t0_ms = dannce_times_ms.min()
    t_rear_min = ((dannce_times_ms[anchor_native] - t0_ms) / 1000.0 / 60.0
                 if n_valid else np.zeros(0))

    reward_native = reward_native_frames(rat, session, di, data_root=data_root)
    reward_native = reward_native[reward_native < dn_native.shape[0]]
    is_rewarded, rears_since_reward, frames_since_reward = label_rears_by_reward(
        anchor_native, reward_native, tol_frames=int(round(reward_tol_s * DANNCE_FPS)))

    return dict(
        rat=rat, session=session,
        n_rears=len(rears), n_valid=n_valid, n_rewards=len(reward_native),
        rear_is_matched_valid=rear_is_matched[valid_rear_idx] if len(rears) else
            np.zeros(0, dtype=bool),
        rear_is_template_like_valid=rear_is_template_like[valid_rear_idx] if len(rears) else
            np.zeros(0, dtype=bool),
        pcs_windows=pcs_windows, variability=variability,
        anchor_native=anchor_native, t_rear_min=t_rear_min,
        reward_native=reward_native,
        is_rewarded=is_rewarded, rears_since_reward=rears_since_reward,
        frames_since_reward=frames_since_reward,
        session_duration_min=(dn_native.shape[0] / DANNCE_FPS / 60.0),
    )
