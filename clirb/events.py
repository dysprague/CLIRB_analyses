"""Single source of truth for CLIRB behavioural event extraction.

Lifted VERBATIM from:
  multi_session.ipynb  cells 7 (projection/replay), 9 (rear detector), 24 (attempts)
  single_session.ipynb cells 3 (matcher), 4 (correct_matches)

Conventions that MUST hold (see the notebook comments and the project memory):

  * Rear detector is HYSTERESIS: enter above REAR_RISE (200), stay until below
    REAR_RESET (150). The min-duration filter counts frames strictly ABOVE the
    rise threshold (n_above). Rear DURATION for plots is (end - start)/FPS, NOT
    n_above.
  * Port-zone gating is evaluated at the MATCH-END (trigger) frame for matched
    rears, else the rear-END frame. NEVER the rear-start frame.
  * Online-equivalent pose pipeline: z-flip (SLEAP only) -> normalize_skeleton
    (egocentric) -> 5-frame CAUSAL median in ROTATED keypoint space -> project.
    DANNCE is NOT z-flipped.
"""
import numpy as np
from collections import deque

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skeleton import normalize_skeleton_batch  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants (must match the online rule)
# --------------------------------------------------------------------------- #
FPS = 20
TEMPLATE_LEN = 30
N_COMPONENTS = 2
ACCEPTED_ERRORS = 3

REAR_RISE = 200.0
REAR_RESET = 150.0
REAR_MIN_FRAMES = int(0.5 * FPS)

LICK_BOUT_GAP_S = 2.0
ATTEMPT_WINDOW_S = 10.0
ALIGN_BUFFER_S = 0.75

LICK_GAP_FR = int(round(LICK_BOUT_GAP_S * FPS))
ATT_WIN_FR = int(round(ATTEMPT_WINDOW_S * FPS))
ALIGN_BUF_FR = int(round(ALIGN_BUFFER_S * FPS))


# --------------------------------------------------------------------------- #
# Pose -> PC projection
# --------------------------------------------------------------------------- #
def causal_median_5_kp(triang):
    """5-frame CAUSAL median over frames, applied in ROTATED keypoint space."""
    pad = 4
    padded = np.concatenate([np.repeat(triang[:1], pad, axis=0), triang], axis=0)
    T = triang.shape[0]
    widx = np.arange(T)[:, None] + np.arange(5)[None, :]
    return np.median(padded[widx], axis=1).astype(np.float32)


def project_session_to_pcs(triang, template, flip_z=True):
    """kp -> PC projection in the order the online rule uses.

    flip_z=True for SLEAP, False for DANNCE.
    Returns (T, n_pcs_total) and the rotated+median keypoints (T, 23, 3).
    """
    x = np.asarray(triang, dtype=np.float32).copy()
    if flip_z:
        x[:, :, 2] = -x[:, :, 2]
    rotated, _, _ = normalize_skeleton_batch(x)
    rotated_med = causal_median_5_kp(rotated)
    flat = rotated_med.reshape(rotated_med.shape[0], -1)
    centered = flat - template['feature_means']
    return centered @ template['pc_weights'].T, rotated_med


# --------------------------------------------------------------------------- #
# Template matching (the online rule)
# --------------------------------------------------------------------------- #
def check_template_match(pc_buffer, pc_template, pc_template_bounds,
                         accepted_errors=ACCEPTED_ERRORS):
    pc_array = np.array(pc_buffer)
    outside = ((pc_array >= pc_template + pc_template_bounds) |
               (pc_array <= pc_template - pc_template_bounds))
    num_outside = np.sum(outside)
    return num_outside <= accepted_errors, num_outside


def get_template_match_indices(keys_pcs, pc_template, pc_template_bounds,
                               refractory_frames=None,
                               accepted_errors=ACCEPTED_ERRORS):
    """Deque implementation, kept for exact parity with single_session cell 3."""
    buffer = deque()
    template_length = pc_template.shape[0]
    match_idxs = []
    num_outside = np.full(keys_pcs.shape[0], np.nan)
    if refractory_frames is None:
        refractory_frames = template_length
    last_match = -refractory_frames
    for i in range(keys_pcs.shape[0]):
        buffer.append(keys_pcs[i, :])
        while buffer and len(buffer) > template_length:
            buffer.popleft()
        if len(buffer) < template_length:
            continue
        is_match, num_outside[i] = check_template_match(
            buffer, pc_template, pc_template_bounds, accepted_errors)
        if is_match and (i - last_match >= refractory_frames):
            match_idxs.append(i)
            last_match = i
    return match_idxs, num_outside


def replay_rule_offline(pcs_full, template, accepted_errors=ACCEPTED_ERRORS):
    """Vectorized equivalent of get_template_match_indices, from multi_session
    cell 7. Returns (match_frames, num_outside_per_frame).

    `template` is the raw template npz dict.
    """
    pcs_to_use = np.asarray(template['pcs_to_use']).ravel().astype(int)
    tmpl = template['template'][:, :len(pcs_to_use)]
    bounds_arr = template['bounds_to_use'] if 'bounds_to_use' in template else (
        np.tile(template['feature_stds'][pcs_to_use] * float(template['bounds']),
                (tmpl.shape[0], 1)))
    pcs = pcs_full[:, pcs_to_use]
    T = pcs.shape[0]
    win = tmpl.shape[0]
    if T < win:
        return np.array([], dtype=int), np.full(T, np.nan)
    base = np.arange(T - win + 1, dtype=np.int64)[:, None]
    windows = pcs[base + np.arange(win, dtype=np.int64)[None, :]]
    above = windows >= (tmpl + bounds_arr)
    below = windows <= (tmpl - bounds_arr)
    outside_count = (above | below).reshape(windows.shape[0], -1).sum(axis=1)
    num_outside = np.full(T, np.nan)
    num_outside[win - 1:] = outside_count
    matches, last = [], -win
    for t in range(win - 1, T):
        if outside_count[t - (win - 1)] <= accepted_errors and (t - last) >= win:
            matches.append(t)
            last = t
    return np.array(matches, dtype=int), num_outside


def correct_matches(pcs, pc_template, matches, buffer_frames=30):
    """Re-center each match on the locally best-aligned template window."""
    pcs = np.asarray(pcs)
    pc_template = np.asarray(pc_template)
    matches = np.asarray(list(matches), dtype=int)
    template_length = pc_template.shape[0]
    n_pcs = pc_template.shape[1]
    T = pcs.shape[0]
    corr = matches.copy()
    for i, match in enumerate(matches):
        e_min = max(match - buffer_frames, template_length - 1)
        e_max = min(match + buffer_frames, T - 1)
        if e_max < e_min:
            continue
        ends = np.arange(e_min, e_max + 1)
        starts = ends - (template_length - 1)
        idx = starts[:, None] + np.arange(template_length)[None, :]
        windows = pcs[idx, :n_pcs]
        mse = ((windows - pc_template[None, :, :]) ** 2).mean(axis=(1, 2))
        corr[i] = int(ends[int(np.argmin(mse))])
    return corr


# --------------------------------------------------------------------------- #
# Rears
# --------------------------------------------------------------------------- #
def detect_rears(pc0, rise=REAR_RISE, reset=REAR_RESET, min_frames=REAR_MIN_FRAMES):
    """Hysteresis rear detector. Returns list of (start, end, n_above)."""
    rears = []
    state, start, n_above = "down", None, 0
    n = len(pc0)
    for i in range(n):
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
        rears.append((start, n - 1, n_above))
    return [r for r in rears if r[2] > min_frames]


def rear_matched(rear_start, rear_end, match_frames, template_len=TEMPLATE_LEN):
    """(hit, match_end_frame). hit[k] True if any match window overlaps rear k."""
    n_rears = len(rear_start)
    hit = np.zeros(n_rears, dtype=bool)
    match_end_frame = np.full(n_rears, -1, dtype=int)
    m_end = np.asarray(match_frames, dtype=int)
    if not n_rears or not len(m_end):
        return hit, match_end_frame
    m_start = m_end - (template_len - 1)
    for k in range(n_rears):
        ov = np.where((m_start <= rear_end[k]) & (m_end >= rear_start[k]))[0]
        if ov.size:
            hit[k] = True
            match_end_frame[k] = int(m_end[ov[0]])
    return hit, match_end_frame


def port_gate_frame(rear_end, match_end_frame, n_frames):
    """The frame the online rig evaluated the port gate on: MATCH-END when the
    rear was matched, else the rear-END frame. NEVER rear-start."""
    gate = np.where(match_end_frame >= 0, match_end_frame, rear_end)
    return np.clip(gate, 0, n_frames - 1)


def best_align_end(rs, re, pcs, template, buf=ALIGN_BUF_FR):
    """End-frame in [rs, re+buf] whose template-length window minimizes MSE."""
    L = template.shape[0]
    e_min = max(rs, L - 1)
    e_max = min(re + buf, len(pcs) - 1)
    if e_max < e_min:
        return -1, np.nan
    ends = np.arange(e_min, e_max + 1)
    starts = ends - (L - 1)
    idx = starts[:, None] + np.arange(L)[None, :]
    win = pcs[idx, :template.shape[1]]
    mse = ((win - template[None]) ** 2).mean(axis=(1, 2))
    k = int(np.argmin(mse))
    return int(ends[k]), float(mse[k])


# --------------------------------------------------------------------------- #
# Licks and attempts
# --------------------------------------------------------------------------- #
def lick_frames_from_idx(lick_idx, n_frames):
    """Lick ONSET frames = where the nearest-lick index increments."""
    lidx = np.asarray(lick_idx, dtype=int).ravel()[:n_frames]
    if not len(lidx):
        return np.array([], dtype=int)
    diff = np.diff(lidx, prepend=lidx[:1])
    return np.where(diff > 0)[0].astype(int)


def lick_bouts(lick_frames, gap_fr=LICK_GAP_FR):
    if not len(lick_frames):
        return lick_frames
    gaps = np.diff(lick_frames, prepend=lick_frames[:1] - gap_fr - 1)
    return lick_frames[gaps >= gap_fr]


def attempts(rear_end, lick_bout_frames, att_win_fr=ATT_WIN_FR):
    """Each lick bout attaches to the most-recent unclaimed rear ending in
    [lf - att_win_fr, lf]. Returns (attempt_rear_idx, attempt_lick_frame)."""
    n_rears = len(rear_end)
    attempt_ri, attempt_lf = [], []
    used = np.zeros(n_rears, dtype=bool)
    for lf in lick_bout_frames:
        cand = (rear_end >= lf - att_win_fr) & (rear_end <= lf) & ~used
        if cand.any():
            idx = np.where(cand)[0]
            j = idx[int(np.argmax(rear_end[idx]))]
            used[j] = True
            attempt_ri.append(j)
            attempt_lf.append(int(lf))
    return np.asarray(attempt_ri, dtype=int), np.asarray(attempt_lf, dtype=int)
