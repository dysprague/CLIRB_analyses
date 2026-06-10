"""
Shared configuration: paths, event codes, keypoint definitions.
"""
import os
import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data"
SKELETON_FILE = "/home/yutaka-sprague/CLIRB/rat23.mat"
SESSION_CSV = os.path.join(DATA_ROOT, "good_sessions_through_2026_02_18_clean.csv")

# ---------------------------------------------------------------------------
# Event codes  (MUST match Arduino firmware)
# ---------------------------------------------------------------------------
TONE_CODE = 11
LICK_CODE = 20
REWARD_ONSET_CODE = 50
REWARD_WINDOW_EXPIRED_CODE = 51
REWARD_COMPLETION_CODE = 52
DANNCE_CODE = 100
SLEAP_CODE = 101
SYNC_CODE = 102

EVENT_CODES = {
    "tone": TONE_CODE,
    "lick": LICK_CODE,
    "reward_onset": REWARD_ONSET_CODE,
    "reward_window_expired": REWARD_WINDOW_EXPIRED_CODE,
    "reward_completion": REWARD_COMPLETION_CODE,
    "dannce_frame": DANNCE_CODE,
    "sleap_frame": SLEAP_CODE,
    "sync": SYNC_CODE,
}

# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------
def load_skeleton(matfile=SKELETON_FILE):
    """Load skeleton node names and edge indices from the rat23.mat file."""
    s = sio.loadmat(matfile, simplify_cells=True)
    nodes = list(map(str, s["joint_names"]))
    edges = (s["joints_idx"] - 1).tolist()  # convert to 0-indexed
    color = s.get("color", None)
    return nodes, edges, color

NODES, EDGES, SKELETON_COLOR = load_skeleton()

# Convenience index lookup
NODE_IDX = {name: i for i, name in enumerate(NODES)}

N_KEYPOINTS = len(NODES)  # 23

# ---------------------------------------------------------------------------
# Standard session path helpers
# ---------------------------------------------------------------------------
def session_path(rat, session):
    return os.path.join(DATA_ROOT, rat, session)

def dannce_path(rat, session):
    return os.path.join(session_path(rat, session), "dannce")

SLEAP_LOCAL_CACHE = os.environ.get(
    "SLEAP_LOCAL_CACHE",
    "/home/yutaka-sprague/CLIRB/data/sleap_2d_cache_2026_05_21",
)


def sleap_path(rat, session):
    """SLEAP per-session folder.

    Prefer the local NVMe cache at SLEAP_LOCAL_CACHE/<rat>/<session>/sleap if
    the directory exists (much faster than SMB). Fall back to the SMB-mounted
    DATA_ROOT path. Override the cache location with the SLEAP_LOCAL_CACHE
    environment variable; clear it ("") to disable.
    """
    if SLEAP_LOCAL_CACHE:
        local = os.path.join(SLEAP_LOCAL_CACHE, rat, session, "sleap")
        if os.path.isdir(local):
            return local
    return os.path.join(session_path(rat, session), "sleap")

def processed_path(rat, session):
    """Per-session "processed" folder holding sleap_dannce_keys.* and
    aligned_data.* outputs. Prefer the local NVMe cache used for the
    Phase G/H/I work (R1/R2/R3 through 2026-02-18); fall back to the
    per-session `processed/` subdir on the SMB share for any session
    not present in the cache (e.g. new rats R4/R5/R6, fresh data).
    """
    local = os.path.join(
        "/home/yutaka-sprague/CLIRB/data/sleap_dannce_keys_2026_02_18",
        rat, session)
    if os.path.isdir(local):
        return local
    return os.path.join(session_path(rat, session), "processed")

def template_path(rat, template_file):
    return os.path.join(DATA_ROOT, rat, "templates", template_file)

def sleap_video_path(rat, session, camera="Camera1"):
    """Path to a SLEAP camera video (0.mp4).

    The local NVMe cache (SLEAP_LOCAL_CACHE) holds only keypoints + calibration,
    NOT the videos — those live on the SMB share under DATA_ROOT. So unlike
    sleap_path (keypoint-oriented), resolve the video by preferring the cache
    only if the .mp4 actually exists there, otherwise fall back to DATA_ROOT.
    """
    cached = os.path.join(sleap_path(rat, session), camera, "0.mp4")
    if os.path.isfile(cached):
        return cached
    return os.path.join(session_path(rat, session), "sleap", camera, "0.mp4")

def dannce_video_path(rat, session, camera="Camera1"):
    return os.path.join(dannce_path(rat, session), "videos", camera, "0.mp4")

def calibration_path(rat, session):
    """Return the calibration folder inside the sleap session directory."""
    cal_dir = os.path.join(sleap_path(rat, session), "calibration")
    if os.path.isdir(cal_dir):
        # Return the first (usually only) subfolder
        subs = sorted(os.listdir(cal_dir))
        if subs:
            return os.path.join(cal_dir, subs[0])
    return cal_dir

# ---------------------------------------------------------------------------
# Calibration date placeholders — fill in with actual calibration folder names
# These are used in QC analyses to mark calibration change boundaries.
# Format: list of date strings matching calibration folder names, e.g.
#   CALIBRATION_DATES = ["2025_07_27", "2025_09_15", "2026_01_10"]
# ---------------------------------------------------------------------------
CALIBRATION_DATES = []
