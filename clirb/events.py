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
  * The online rule leaves the first 4 frames of a session UNFILTERED (its
    buffer is not yet 5 deep). Use warmup='online' to reproduce that; the
    'pad' warm-up exists only for callers that slice off leading context
    themselves (see clirb/native_dannce.py).
  * A template match is NOT a reward. The rig applies, in this order:
    buffer-full -> interpolated-frame gate -> port-distance gate ->
    check_buffer -> refractory (SECONDS, on camera acquisition times) ->
    task-context gate. Use replay_online_rule() rather than reimplementing
    any subset of that stack.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass

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
def causal_median_5_kp(triang, warmup="pad"):
    """5-frame CAUSAL median over frames, applied in ROTATED keypoint space.

    Mirrors TemplateMatch.corrections (campy/behavior_rule.py): the rule's
    buffer already holds the current frame, so the median window is
    [t-4 ... t] with every sample counted once.

    warmup
      'pad'    : repeat frame 0 four times so frames 0-3 are also filtered.
                 Only correct when the caller supplied real leading context
                 and slices it off (clirb/native_dannce.py does this).
      'online' : leave frames 0-3 as-is, exactly as the rule does while its
                 buffer fills. Use this for session-level replay.
    """
    triang = np.asarray(triang)
    T = triang.shape[0]
    pad = 4
    padded = np.concatenate([np.repeat(triang[:1], pad, axis=0), triang], axis=0)
    widx = np.arange(T)[:, None] + np.arange(5)[None, :]
    out = np.median(padded[widx], axis=1).astype(np.float32)
    if warmup == "online":
        out[:pad] = triang[:pad]
    elif warmup != "pad":
        raise ValueError(f"warmup must be 'pad' or 'online', got {warmup!r}")
    return out


def project_session_to_pcs(triang, template, flip_z=True, warmup="online"):
    """kp -> PC projection in the order the online rule uses.

    flip_z=True for SLEAP, False for DANNCE.
    warmup='online' reproduces the rule's unfiltered first 4 frames; this
    only ever affects frames 1-3, well before the first checkable window.
    Returns (T, n_pcs_total) and the rotated+median keypoints (T, 23, 3).
    """
    x = np.asarray(triang, dtype=np.float32).copy()
    if flip_z:
        x[:, :, 2] = -x[:, :, 2]
    rotated, _, _ = normalize_skeleton_batch(x)
    rotated_med = causal_median_5_kp(rotated, warmup=warmup)
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
# The online rule, end to end
# --------------------------------------------------------------------------- #
@dataclass
class OnlineParams:
    """Everything the rig needed to turn a template match into a reward.

    Build one with data_io.load_session_config(); do not hand-type these.
    """
    fps: float = float(FPS)
    accepted_errors: int = ACCEPTED_ERRORS
    min_port_distance: float | None = None
    task_id: int = 1
    context_switch_time: float = 0.0
    reward_threshold: int | None = None
    train_state: int | None = None
    template_file: str | None = None


def template_bounds(template):
    """Per-timepoint, per-PC half-width of the accepted band.

    Mirrors TemplateMatch.current_bounds: explicit bounds_to_use when the
    template carries it, else feature_stds[pcs_to_use] * bounds.
    """
    pcs_to_use = np.asarray(template['pcs_to_use']).ravel().astype(int)
    tmpl = template['template'][:, :len(pcs_to_use)]
    if 'bounds_to_use' in template:
        b = np.asarray(template['bounds_to_use'])
        return b[:, :len(pcs_to_use)] if b.ndim == 2 else b
    return np.tile(np.asarray(template['feature_stds'])[pcs_to_use]
                   * float(template['bounds']), (tmpl.shape[0], 1))


def windowed_out_of_bounds(pcs_full, template):
    """check_buffer evaluated on every window, vectorized.

    Returns num_outside (T,) with NaN before the first full window; entry t is
    the count for the window ENDING at t.
    """
    pcs_to_use = np.asarray(template['pcs_to_use']).ravel().astype(int)
    tmpl = np.asarray(template['template'])[:, :len(pcs_to_use)]
    band = template_bounds(template)
    pcs = np.asarray(pcs_full)[:, pcs_to_use]
    T, W = pcs.shape[0], tmpl.shape[0]
    num_outside = np.full(T, np.nan)
    if T < W:
        return num_outside
    base = np.arange(T - W + 1, dtype=np.int64)[:, None]
    win = pcs[base + np.arange(W, dtype=np.int64)[None, :]]
    oob = (win >= (tmpl + band)) | (win <= (tmpl - band))
    num_outside[W - 1:] = oob.reshape(win.shape[0], -1).sum(axis=1)
    return num_outside


def online_com_trailing_mean(triang, window=TEMPLATE_LEN):
    """The rig's COM_buffer: trailing mean of the per-frame COM.

    behavior.py appends mean(keypoints, axis=0) once per CAMERA frame -- real
    and interpolated alike -- and trims to `window`. The port gate then uses
    the mean of that buffer, NOT the instantaneous COM. Returns (T, 3).
    """
    com = np.asarray(triang).mean(axis=1)
    cs = np.cumsum(np.vstack([np.zeros((1, com.shape[1])), com]), axis=0)
    idx = np.arange(com.shape[0])
    lo = np.maximum(0, idx + 1 - window)
    return (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)[:, None]


def online_port_distance(triang, port_xy, window=TEMPLATE_LEN):
    """x-y distance from the trailing-mean COM to the port, per frame.

    This is what behavior.get_port_distance computes. Using the instantaneous
    COM instead can disagree by hundreds of units mid-rear.
    """
    m = online_com_trailing_mean(triang, window)
    port_xy = np.asarray(port_xy).ravel()[:2]
    return np.hypot(m[:, 0] - port_xy[0], m[:, 1] - port_xy[1])


# Rejection reasons, in the order the rig applies them.
REASONS = ('buffer_not_full', 'interpolated', 'port_gate', 'no_match',
           'refractory', 'task_context', 'reward')


def replay_online_rule(pcs_full, template, params=None, is_real=None,
                       acq_times=None, port_dist=None):
    """Single-pass replay of ProcessBehavior + TemplateMatch.update.

    Gates are applied in the rig's order, and only a frame that clears all of
    them advances the refractory clock -- a port-gated or interpolated frame
    does NOT, because behavior.py never reaches the trigger check on it.

    pcs_full  : (T, n_pcs) from project_session_to_pcs
    template  : template npz dict (or Mapping)
    params    : OnlineParams; defaults are permissive (no gates)
    is_real   : (T,) bool, False on interpolated fills. None => all real.
    acq_times : (T,) camera acquisition time in seconds. None => t/fps.
    port_dist : (T,) from online_port_distance. None => port gate disabled.

    Returns dict:
      reward           : frames where water was actually delivered
      fired            : frames where the rule fired (reward or task-suppressed)
      passing          : every window that passed check_buffer, no gates
      matches_ungated  : passing + frame-refractory only (the legacy count)
      num_outside      : (T,) per-window error count
      reason           : (T,) object array, one of REASONS
      counts           : {reason: n}

    Note: the rig skips rule.update on the session's very first queue item
    (first_run_done), so its buffer fills one frame later than this replay.
    That shifts only windows ending at frame W-1.
    """
    params = params or OnlineParams()
    W = int(np.asarray(template['template']).shape[0])
    T = int(np.asarray(pcs_full).shape[0])
    fps = float(params.fps)

    num_outside = windowed_out_of_bounds(pcs_full, template)
    if is_real is None:
        is_real = np.ones(T, dtype=bool)
    if acq_times is None:
        acq_times = np.arange(T) / fps
    is_real = np.asarray(is_real, dtype=bool)
    acq_times = np.asarray(acq_times, dtype=float)

    # TemplateMatch.__init__ OVERRIDES reward_params['refractory'] with
    # pc_buffer_duration/fps, and compares against camera acquisition times.
    refractory_s = W / fps

    passing = np.where(num_outside <= params.accepted_errors)[0]

    matches_ungated, last = [], -W
    for t in passing:
        if t - last >= W:
            matches_ungated.append(int(t))
            last = t

    reason = np.empty(T, dtype=object)
    reward, fired = [], []
    last_trigger = -np.inf
    for f in range(T):
        if not np.isfinite(num_outside[f]):
            reason[f] = 'buffer_not_full'
            continue
        if not is_real[f]:
            reason[f] = 'interpolated'          # rule.update(check_time=False)
            continue
        if (params.min_port_distance is not None and port_dist is not None
                and port_dist[f] < params.min_port_distance):
            reason[f] = 'port_gate'             # rule.update(check_time=False)
            continue
        if num_outside[f] > params.accepted_errors:
            reason[f] = 'no_match'
            continue
        if (acq_times[f] - last_trigger) < refractory_s:
            reason[f] = 'refractory'
            continue
        # The rule sets _last_trigger whenever it fires, even when the task
        # context suppresses the actual water delivery.
        last_trigger = acq_times[f]
        fired.append(f)
        if not _task_active(f, params):
            reason[f] = 'task_context'
            continue
        reason[f] = 'reward'
        reward.append(f)

    counts = {r: int((reason == r).sum()) for r in REASONS}
    return {
        'reward': np.asarray(reward, dtype=int),
        'fired': np.asarray(fired, dtype=int),
        'passing': passing.astype(int),
        'matches_ungated': np.asarray(matches_ungated, dtype=int),
        'num_outside': num_outside,
        'reason': reason,
        'counts': counts,
    }


def _task_active(frame, params):
    """task_id 0 = never reward, 1 = always, 2 = only after context_switch_time."""
    if params.task_id == 0:
        return False
    if params.task_id == 2:
        return (frame / float(params.fps)) > float(params.context_switch_time)
    return True


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
