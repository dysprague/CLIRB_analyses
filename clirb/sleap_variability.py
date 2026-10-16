"""SLEAP-only (native 20 Hz) fallback variant of clirb.native_dannce's
per-session variability pipeline, for sessions that don't have DANNCE 3D
reconstruction yet.

Mirrors `build_session_variability`'s logic and return shape exactly, reusing
the same window/variability/reward-labeling helpers. The only real
differences are the data source (SLEAP `triang_keys_3D.npy`, which needs the
z-flip `project_session_to_pcs(flip_z=True)` applies) and the native rate
(20 Hz, SLEAP's own -- so unlike the DANNCE pipeline there is no upsampling
step: the 20 Hz alignment anchor from `best_align_end` already IS the
native-rate anchor here, with no `dannce_idx_for_sleap_cams` hop needed).

IMPORTANT: results from this module are NOT comparable in absolute units to
`build_session_variability`'s DANNCE-based numbers (different camera system,
frame rate, and effectively a different feature stream through the same
template weights). Only use this to compare two sessions run through this
SAME fallback against each other -- e.g. a session without DANNCE data yet
against a historical session reprocessed the same way for a fair baseline.
"""
import os

import numpy as np
import pandas as pd

from clirb.events import (detect_rears, best_align_end, replay_rule_offline,  # noqa: E402
                          rear_matched, project_session_to_pcs, TEMPLATE_LEN)
from clirb import templates as tmpl  # noqa: E402
from clirb.native_dannce import (extract_native_window, reward_sleap_rows,  # noqa: E402
                                 label_rears_by_reward, sliding_window_variability,
                                 DATA_ROOT)

SLEAP_FPS = 20.0


def build_session_variability_sleap(rat, session, half_win_s=2.0, lookback=4,
                                    accepted_errors=3, reward_tol_s=1.0,
                                    template_like_bound_mult=1.25, n_pcs=3,
                                    data_root=DATA_ROOT):
    half_win_frames = int(round(half_win_s * SLEAP_FPS))
    sleap_dir = os.path.join(data_root, rat, session, "sleap")
    triang_p = os.path.join(sleap_dir, "triang_keys_3D.npy")
    if not os.path.isfile(triang_p):
        raise ValueError(f"{rat}/{session}: no triang_keys_3D.npy")
    sleap_native = np.load(triang_p).astype(np.float32)
    sleap_flipped = sleap_native.copy()
    sleap_flipped[:, :, 2] = -sleap_flipped[:, :, 2]  # SLEAP z-flip convention

    template = tmpl.load_template(rat, session)
    pcs_native, _ = project_session_to_pcs(sleap_native, template, flip_z=True)

    rears = detect_rears(pcs_native[:, 0])
    rear_start = np.array([r[0] for r in rears])
    rear_end = np.array([r[1] for r in rears])

    match_frames, _ = replay_rule_offline(pcs_native, template, accepted_errors=accepted_errors)
    rear_is_matched, _ = rear_matched(rear_start, rear_end, match_frames, template_len=TEMPLATE_LEN)

    wide_template = dict(template)
    wide_template["bounds_to_use"] = template["bounds_to_use"] * template_like_bound_mult
    wide_match_frames, _ = replay_rule_offline(pcs_native, wide_template, accepted_errors=accepted_errors)
    rear_is_template_like, _ = rear_matched(rear_start, rear_end, wide_match_frames, template_len=TEMPLATE_LEN)

    # extract_native_window does no flipping itself (built for DANNCE, which
    # isn't flipped) -- feed it the pre-flipped stream so its output PCs
    # match the flip_z=True convention used for detection/alignment above.
    pcs_windows, valid_rear_idx, anchor_native = [], [], []
    for k, (rs, re_, _n_above) in enumerate(rears):
        end20, _mse = best_align_end(rs, re_, pcs_native, template["template"][:, :2])
        if end20 < 0:
            continue
        w = extract_native_window(sleap_flipped, end20, template, half_win_frames=half_win_frames)
        if w is None:
            continue  # +/-half_win_s window runs off the session boundary
        pcs_windows.append(w["pcs"][:, :n_pcs])
        valid_rear_idx.append(k)
        anchor_native.append(end20)
    anchor_native = np.array(anchor_native, dtype=int)
    n_valid = len(pcs_windows)

    variability = np.array([sliding_window_variability(pcs_windows, i, lookback=lookback)
                            for i in range(n_valid)]) if n_valid else np.zeros((0, n_pcs))

    # Wall-clock time axis via frame_mapping.csv's shared hardware clock
    # (same source single_session.ipynb uses), nearest-cam0-row per frame.
    fm = pd.read_csv(os.path.join(sleap_dir, "frame_mapping.csv"), skipinitialspace=True)
    cam0 = fm["cam0_frame"].to_numpy(dtype=int)
    acq = fm["cam0_acquisition_time"].to_numpy(dtype=float)
    order = np.argsort(cam0)
    cam0_sorted, acq_sorted = cam0[order], acq[order]
    idx = np.clip(np.searchsorted(cam0_sorted, np.arange(sleap_native.shape[0])),
                 0, len(cam0_sorted) - 1)
    t_all = acq_sorted[idx]
    t0 = t_all.min()
    t_rear_min = ((t_all[anchor_native] - t0) / 1000.0 / 60.0) if n_valid else np.zeros(0)

    reward_native = reward_sleap_rows(rat, session, data_root=data_root)
    reward_native = reward_native[reward_native < sleap_native.shape[0]]
    is_rewarded, rears_since_reward, frames_since_reward = label_rears_by_reward(
        anchor_native, reward_native, tol_frames=int(round(reward_tol_s * SLEAP_FPS)))

    return dict(
        rat=rat, session=session, space="sleap_native_20hz",
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
        session_duration_min=(sleap_native.shape[0] / SLEAP_FPS / 60.0),
    )
