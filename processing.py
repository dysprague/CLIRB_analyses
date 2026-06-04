"""
Processing utilities: time alignment, template matching, feature extraction.
"""
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------
def closest_indices(source_times, target_times):
    """
    For each t in target_times, return the index of the closest value in source_times.
    Returns empty array if source_times is empty.
    """
    if len(source_times) == 0:
        return np.array([], dtype=int)
    return np.array(
        [np.argmin(np.abs(source_times - t)) for t in target_times], dtype=int
    )


def align_session_frames(opcon_events, n_frames_dannce, n_frames_sleap_cams):
    """
    Compute frame alignment indices between SLEAP and DANNCE cameras.

    Parameters
    ----------
    opcon_events : dict from load_opcon_events()
    n_frames_dannce : int — total DANNCE video frames
    n_frames_sleap_cams : int — total SLEAP video frames (add 1 for starter frame)

    Returns
    -------
    dict with alignment arrays and event indices
    """
    sleap_raw = opcon_events["sleap_frame_times_ms"]
    dannce_raw = opcon_events["dannce_frame_times_ms"]

    # Find session start from SLEAP trigger gaps (>200 ms)
    diff_sl = np.diff(sleap_raw)
    first_sl_idx = np.where(diff_sl > 200)[0][-1]

    session_start_sl = int(sleap_raw[first_sl_idx])
    sleap_times = sleap_raw[first_sl_idx:][:n_frames_sleap_cams]

    # DANNCE frame start from gaps (>25 ms)
    diff_dn = np.diff(dannce_raw)
    first_dn_idx = np.where(diff_dn > 25)[0][-1] + 1
    dannce_times = dannce_raw[first_dn_idx : first_dn_idx + n_frames_dannce]

    tone_times = opcon_events["tone_times_ms"]
    lick_times = opcon_events["lick_times_ms"]
    reward_times = opcon_events["reward_start_times_ms"]

    return {
        "sleap_times_ms": sleap_times,
        "dannce_times_ms": dannce_times,
        "session_start_sl_ms": session_start_sl,
        "sleap_idx_for_dannce_cams": closest_indices(sleap_times, dannce_times),
        "dannce_idx_for_sleap_cams": closest_indices(dannce_times, sleap_times),
        "tone_idx_dannce_cams": closest_indices(tone_times, dannce_times),
        "lick_idx_dannce_cams": closest_indices(lick_times, dannce_times),
        "reward_idx_dannce_cams": closest_indices(reward_times, dannce_times),
        "tone_idx_sleap_cams": closest_indices(tone_times, sleap_times),
        "lick_idx_sleap_cams": closest_indices(lick_times, sleap_times),
        "reward_idx_sleap_cams": closest_indices(reward_times, sleap_times),
        "tone_times_ms": tone_times,
        "lick_times_ms": lick_times,
        "reward_times_ms": reward_times,
    }


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------
def check_template_match(pc_buffer, pc_template, pc_template_bounds):
    """
    Check if PC values in buffer match the template within bounds.

    Parameters
    ----------
    pc_buffer : (buffer_length, n_components) — recent PC values
    pc_template : (template_length, n_components) — template
    pc_template_bounds : (template_length, n_components) — per-timepoint bounds

    Returns
    -------
    bool — True if all timepoints are within bounds
    """
    diff = np.abs(np.array(pc_buffer) - pc_template)
    return np.all(diff <= pc_template_bounds)


def check_template_match(pc_buffer, pc_template, pc_template_bounds):
    """
    Check if PC values in buffer match the template within specified bounds.
    
    Parameters:
    - pc_buffer: Array of PC values, shape (buffer_length, n_components)
    - pc_template: Template PC trajectory, shape (template_length, n_components)
    - pc_template_bounds: Tolerance bounds for each PC dimension
    
    Returns:
    - True if buffer matches template within bounds, False otherwise
    """
    pc_array = np.array(pc_buffer)
    
    # Check how many points fall outside the bounds
    outside_bounds = (pc_array >= pc_template + pc_template_bounds) | (pc_array <= pc_template - pc_template_bounds)
    
    # If fewer than 3 points are outside bounds, consider it a match
    num_outside = np.sum(outside_bounds)
    
    return num_outside <= 3, num_outside


def get_template_match_indices(keys_pcs, pc_template, pc_template_bounds, refractory_frames=None):
    """
    Search through PC data to find frames matching a behavioral template.
    
    Parameters:
    - keys_pcs: Full PC trajectory data, shape (n_frames, n_components)
    - pc_template: Template PC trajectory to match, shape (template_length, n_components)
    - pc_template_bounds: Tolerance bounds for matching, same shape as pc_template
    - refractory_frames: Minimum frames between matches (default: template_length)
    
    Returns:
    - match_idxs: List of frame indices where template matches were found
    """
    buffer = deque()
    template_length = pc_template.shape[0]
    match_idxs = []
    num_outside = np.full(keys_pcs.shape[0], np.nan)
    
    # Set refractory period to template length if not specified
    if refractory_frames is None:
        refractory_frames = template_length
    
    last_match = -refractory_frames
    
    # Slide through all frames
    for i in range(keys_pcs.shape[0]):
        # Add current frame's PC values to buffer
        buffer.append(keys_pcs[i, :])
        
        # Remove old frames that exceed template length
        while buffer and len(buffer) > template_length:
            buffer.popleft()
        
        # Wait until buffer is full before checking
        if len(buffer) < template_length:
            continue
        
        # Check if buffer matches template AND refractory period has passed
        is_match, num_outside[i] = check_template_match(buffer, pc_template, pc_template_bounds)
        if is_match and (i - last_match >= refractory_frames):
            match_idxs.append(i)
            last_match = i
    
    return match_idxs, num_outside


# ---------------------------------------------------------------------------
# Feature extraction (for full behavior analysis)
# ---------------------------------------------------------------------------
def get_rears(keys_3D, start_thresh, nose_point=0,
              stop_thresh=None, min_len_frames=1):
    """
    Detect rear bouts using nose height thresholding with hysteresis.

    Parameters
    ----------
    keys_3D : (n_frames, n_keypoints, 3)
    start_thresh : float — z-height to start a rear
    nose_point : int — keypoint index for nose (default 0 = Snout)
    stop_thresh : float, optional — z-height to end a rear (default = start_thresh)
    min_len_frames : int — minimum bout length

    Returns
    -------
    bouts : list of (start, end) frame tuples
    """
    if stop_thresh is None:
        stop_thresh = start_thresh

    nose_z = keys_3D[:, nose_point, 2]
    in_rear = False
    bouts = []
    start = 0

    for i in range(len(nose_z)):
        if not in_rear and nose_z[i] > start_thresh:
            in_rear = True
            start = i
        elif in_rear and nose_z[i] < stop_thresh:
            in_rear = False
            if (i - start) >= min_len_frames:
                bouts.append((start, i))

    if in_rear and (len(nose_z) - start) >= min_len_frames:
        bouts.append((start, len(nose_z)))

    return bouts


def compute_speed(keys_3D, keypoint_idx=4, dt=1.0):
    """
    Compute speed of a keypoint (default SpineM=4) across frames.

    Returns (n_frames,) array with speed[0] = 0.
    """
    pos = keys_3D[:, keypoint_idx, :]
    diffs = np.diff(pos, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / dt
    return np.concatenate([[0], speeds])


def compute_com(keys_3D):
    """Compute center of mass across all keypoints. Returns (n_frames, 3)."""
    return np.mean(keys_3D, axis=1)


def extract_session_features(keys_3D, rear_thresh=100, dt=1.0):
    """
    Extract a standard set of behavioral features from 3D keypoints.

    Returns dict with arrays for each feature.
    """
    rears = get_rears(keys_3D, start_thresh=rear_thresh)
    speed = compute_speed(keys_3D, dt=dt)
    com = compute_com(keys_3D)

    # Per-rear features
    rear_heights = []
    rear_lengths = []
    rise_times = []
    for start, end in rears:
        nose_z = keys_3D[start:end, 0, 2]
        rear_heights.append(np.max(nose_z))
        rear_lengths.append(end - start)
        peak_idx = np.argmax(nose_z)
        rise_times.append(peak_idx)

    # Hand-to-nose distances
    lh_nose = np.linalg.norm(
        keys_3D[:, 10, :] - keys_3D[:, 0, :], axis=1  # HandL - Snout
    )
    rh_nose = np.linalg.norm(
        keys_3D[:, 14, :] - keys_3D[:, 0, :], axis=1  # HandR - Snout
    )

    return {
        "rears": rears,
        "rear_height": np.array(rear_heights),
        "rear_length": np.array(rear_lengths),
        "rise_time": np.array(rise_times),
        "speed": speed,
        "com": com,
        "LH_nose_dist": lh_nose,
        "RH_nose_dist": rh_nose,
        "nose_speed": compute_speed(keys_3D, keypoint_idx=0, dt=dt),
    }
