"""Local disk cache for the expensive raw per-session inputs used by
multi_session.ipynb and interactive_rat.ipynb.

Both notebooks re-read large files from the slow SMB mount every run:
  - sleap/triang_keys_3D.npy                (~13-20 MB)
  - sleap/frame_mapping.csv                 (~2-3 MB)
  - processed/aligned_data.mat              (~55-81 MB, for two index vectors)
  - processed/sleap_dannce_keys.mat         (~87 MB, section-2 only)

This module caches the RAW arrays those files provide to a local .npz per
session, so subsequent runs read locally and only hit SMB for sessions not yet
cached (or whose source files changed). The notebooks keep doing their own
cheap CPU derivation (rears / attempts / gating / PC projection) from these raw
inputs -- we cache the INPUTS, not the derived plot data, so any future plot
still works offline.

Mirrors the prefer-local idiom already in config.py (SLEAP_LOCAL_CACHE /
processed_path). Cache location overridable via $CLIRB_CACHE_ROOT.

Usage in a notebook:

    import session_cache as sc
    raw = sc.get_raw_session(rat, session)          # dict of raw arrays, cached
    triang = raw['triang_3d']                        # instead of np.load(SMB)
    lick_idx = raw['lick_idx_sleap_cams']            # instead of loadmat(81MB)
    # section 2:
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw['has_dual']:
        sleap_3d, dannce_3d = raw['sleap_keys_3D'], raw['dannce_keys_3D']

Template npz is per-RAT (not per-session) and already tiny (~45 KB), so it is
NOT cached here -- the notebooks keep loading it directly.
"""
import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_ROOT = "/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data"

# Bump whenever the SET of cached fields or how they are extracted changes, to
# force every session to rebuild. (Not tied to the rear/gating analysis logic --
# that is derived in the notebooks from these raw inputs, so it does not affect
# the cache.)
CACHE_VERSION = 2   # v2: added sleap_keys_2D (section 2.2) to the dual cache

CACHE_ROOT = os.environ.get(
    "CLIRB_CACHE_ROOT",
    "/home/yutaka-sprague/CLIRB_analyses/data/cache",
)

# Store the big (n,23,3) arrays as float32 (source is float64). Halves size to
# ~30 MB/session with negligible precision impact for PC projection / port
# distance. Set to False to keep float64.
STORE_FLOAT32 = True


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def cache_path(rat, session):
    return os.path.join(CACHE_ROOT, rat, f"{session}.npz")


def _sleap_dir(rat, session):
    return os.path.join(DATA_ROOT, rat, session, "sleap")


def _processed_dir(rat, session):
    return os.path.join(DATA_ROOT, rat, session, "processed")


def _aligned_path(rat, session):
    return os.path.join(_processed_dir(rat, session), "aligned_data.mat")


def _dual_keys_path(rat, session):
    return os.path.join(_processed_dir(rat, session), "sleap_dannce_keys.mat")


def _keys2d_path(rat, session):
    return os.path.join(_sleap_dir(rat, session), "sleap_keys_2D.npy")


def _port_path(rat, session):
    m = sorted(glob.glob(os.path.join(
        _sleap_dir(rat, session), "calibration", "*", "water_port_location.npy")))
    return m[0] if m else None


def source_files(rat, session, want_dual=True):
    """The SMB files this cache entry depends on (existing ones only)."""
    sd = _sleap_dir(rat, session)
    cands = [
        os.path.join(sd, "triang_keys_3D.npy"),
        os.path.join(sd, "frame_mapping.csv"),
        os.path.join(sd, "campy_config.yaml"),
        _aligned_path(rat, session),
    ]
    p = _port_path(rat, session)
    if p:
        cands.append(p)
    if want_dual:
        cands.append(_dual_keys_path(rat, session))
        cands.append(_keys2d_path(rat, session))  # section 2.2 reprojection
    return [c for c in cands if os.path.isfile(c)]


def _src_signature(rat, session, want_dual=True):
    """dict {path: [mtime, size]} for every existing source file."""
    sig = {}
    for f in source_files(rat, session, want_dual=want_dual):
        st = os.stat(f)
        sig[f] = [float(st.st_mtime), int(st.st_size)]
    return sig


# --------------------------------------------------------------------------- #
# Validity
# --------------------------------------------------------------------------- #
def is_valid(rat, session, want_dual=True):
    """True iff a cache file exists, its version matches, it has the dual data
    if requested, and no source file is newer/bigger than when it was built."""
    cp = cache_path(rat, session)
    if not os.path.isfile(cp):
        return False
    try:
        with np.load(cp, allow_pickle=True) as z:
            if int(z["cache_version"]) != CACHE_VERSION:
                return False
            # Was the cache BUILT with dual intent? (Not whether dual data
            # existed -- a session with no DANNCE is still validly "dual-built".)
            built_dual = bool(z["built_dual"]) if "built_dual" in z.files \
                else bool(z["has_dual"])
            if want_dual and not built_dual:
                return False
            cached_sig = z["src_signature"].item()  # dict stored as 0-d object
    except Exception:
        return False
    # Re-stat the SAME source set the cache was built with (built_dual), so a
    # dual-built cache also satisfies a want_dual=False request (it is a
    # superset). Compare against the recorded signature.
    cur_sig = _src_signature(rat, session, want_dual=built_dual)
    if set(cur_sig) != set(cached_sig):
        return False
    for f, (mt, sz) in cur_sig.items():
        old = cached_sig.get(f)
        if old is None or sz != old[1] or mt > old[0] + 1e-6:
            return False
    return True


# --------------------------------------------------------------------------- #
# Build (read SMB)
# --------------------------------------------------------------------------- #
def _f(arr):
    a = np.asarray(arr)
    if STORE_FLOAT32 and a.dtype == np.float64:
        return a.astype(np.float32)
    return a


def _read_min_port_and_train_state(rat, session):
    """Parse campy_config.yaml -> (min_port_distance, train_state). Matches the
    notebooks' own parsing (min_port via line scan; train_state via yaml)."""
    min_port = 200.0
    train_state = np.nan
    cfg = os.path.join(_sleap_dir(rat, session), "campy_config.yaml")
    if not os.path.isfile(cfg):
        return min_port, train_state
    # min_port_distance: simple line scan (matches load_session cell 4)
    try:
        with open(cfg) as f:
            for line in f:
                if "min_port_distance" in line and ":" in line:
                    try:
                        min_port = float(line.split(":", 1)[1].split("#")[0].strip())
                    except Exception:
                        pass
                    break
    except Exception:
        pass
    # train_state: yaml (matches _load_train_state cell 22)
    try:
        import yaml
        with open(cfg) as f:
            c = yaml.safe_load(f)
        ts = (c.get("behavior_params", {}) or {}).get("train_state", None)
        train_state = int(ts) if ts is not None else np.nan
    except Exception:
        pass
    return min_port, train_state


def _build(rat, session, want_dual=True):
    """Read the raw inputs from SMB and return the cache dict, or None if the
    essential SLEAP inputs are missing/corrupt."""
    sd = _sleap_dir(rat, session)
    triang_p = os.path.join(sd, "triang_keys_3D.npy")
    fm_p = os.path.join(sd, "frame_mapping.csv")
    if not (os.path.isfile(triang_p) and os.path.isfile(fm_p)):
        return None
    try:
        triang = np.load(triang_p)
    except Exception as e:
        print(f"  {rat}/{session}: triang_keys_3D.npy unreadable ({e!r})")
        return None

    fm = pd.read_csv(fm_p, skipinitialspace=True)
    frame_mapping_cam0 = fm["cam0_frame"].to_numpy(dtype=np.int64)
    # multi_session cell 6 also needs the acquisition-time column.
    cam0_acq_time = (fm["cam0_acquisition_time"].to_numpy(dtype=np.float64)
                     if "cam0_acquisition_time" in fm.columns
                     else np.full(len(fm), np.nan))

    # behavior_log: multi_session cell 6 uses the frames where triggered==1.
    # Cache just those frames (+ a has_task_log flag) instead of the CSV.
    bl_p = os.path.join(sd, "behavior_log.csv")
    has_task_log = os.path.isfile(bl_p)
    behavior_log_frames = np.zeros(0, dtype=np.int64)
    if has_task_log:
        try:
            bl = pd.read_csv(bl_p, skipinitialspace=True)
            if "triggered" in bl.columns and "frame" in bl.columns:
                behavior_log_frames = bl.loc[bl["triggered"] == 1, "frame"].to_numpy(dtype=np.int64)
        except Exception as e:
            print(f"  {rat}/{session}: behavior_log.csv unreadable ({e!r})")
            has_task_log = False

    n_frames = len(triang)
    min_port, train_state = _read_min_port_and_train_state(rat, session)

    port_p = _port_path(rat, session)
    port_xy = (np.asarray(np.load(port_p)).ravel()[:3].astype(np.float64)
               if port_p else np.array([np.nan, np.nan, np.nan]))

    # aligned_data.mat -> two index vectors (the 55-81 MB read for ~0.3 MB used)
    lick_idx = np.zeros(0, dtype=np.int64)
    dannce_idx = np.zeros(0, dtype=np.int64)
    ap = _aligned_path(rat, session)
    if os.path.isfile(ap):
        try:
            al = sio.loadmat(ap, simplify_cells=True)
            if "lick_idx_sleap_cams" in al:
                lick_idx = np.asarray(al["lick_idx_sleap_cams"], dtype=np.int64).ravel()
            if "dannce_idx_for_sleap_cams" in al:
                dannce_idx = np.asarray(al["dannce_idx_for_sleap_cams"], dtype=np.int64).ravel()
        except Exception as e:
            print(f"  {rat}/{session}: aligned_data.mat unreadable ({e!r})")

    out = {
        "cache_version": np.int64(CACHE_VERSION),
        "rat": rat,
        "session": session,
        "n_frames": np.int64(n_frames),
        "triang_3d": _f(triang),
        "frame_mapping_cam0": frame_mapping_cam0,
        "cam0_acq_time": cam0_acq_time,
        "has_task_log": bool(has_task_log),
        "behavior_log_frames": behavior_log_frames,
        "lick_idx_sleap_cams": lick_idx,
        "dannce_idx_for_sleap_cams": dannce_idx,
        "port_xy": port_xy,
        "min_port_distance": np.float64(min_port),
        "train_state": np.float64(train_state),
        "has_dual": False,
        "has_keys2d": False,
    }

    # section-2 SLEAP+DANNCE keys (the 87 MB read)
    if want_dual:
        dp = _dual_keys_path(rat, session)
        if os.path.isfile(dp):
            try:
                keys = sio.loadmat(dp, simplify_cells=True)
                out["sleap_keys_3D"] = _f(keys["sleap_keys_3D"])
                out["dannce_keys_3D"] = _f(keys["dannce_keys_3D"])
                out["has_dual"] = True
            except Exception as e:
                print(f"  {rat}/{session}: sleap_dannce_keys.mat unreadable ({e!r})")
        # section-2.2 reprojection reads sleap_keys_2D.npy (~89 MB) independently
        # of the dual-keys mat -- (T, 3, 23, 3) = (x, y, conf). Stored float32:
        # cell 48 re-triangulates in float64 from the x,y coords, and f32 on
        # pixel coords is < 1e-3 px, negligible vs the plotted residual.
        k2 = _keys2d_path(rat, session)
        if os.path.isfile(k2):
            try:
                out["sleap_keys_2D"] = _f(np.load(k2))
                out["has_keys2d"] = True
            except Exception as e:
                print(f"  {rat}/{session}: sleap_keys_2D.npy unreadable ({e!r})")
    return out


def _atomic_savez(path, data, built_dual):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # store src_signature as a 0-d object array so it round-trips as a dict.
    # Signature covers the source set for the intent this cache was BUILT with
    # (built_dual), so is_valid re-stats exactly those files.
    payload = dict(data)
    payload["built_dual"] = bool(built_dual)
    payload["src_signature"] = np.array(
        _src_signature(data["rat"], data["session"], want_dual=built_dual),
        dtype=object)
    # np.savez appends '.npz' to a path without that suffix; give it a temp base
    # ending in .npz so the written file is exactly tmp, then atomically rename.
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_raw_session(rat, session, want_dual=True, verbose=False):
    """Return the raw-input dict for (rat, session), reading from the local
    cache if valid, otherwise building it from SMB and writing the cache.
    Returns None if the essential SLEAP inputs are missing/corrupt.

    Keys: triang_3d, frame_mapping_cam0, lick_idx_sleap_cams,
    dannce_idx_for_sleap_cams, port_xy, min_port_distance, train_state,
    n_frames, has_dual, and (if has_dual) sleap_keys_3D, dannce_keys_3D.
    """
    if is_valid(rat, session, want_dual=want_dual):
        with np.load(cache_path(rat, session), allow_pickle=True) as z:
            out = {k: z[k] for k in z.files
                   if k not in ("src_signature", "built_dual", "cache_version")}
        # unwrap scalars / strings
        out["rat"] = str(out["rat"]); out["session"] = str(out["session"])
        out["n_frames"] = int(out["n_frames"])
        out["has_dual"] = bool(out["has_dual"])
        out["has_keys2d"] = bool(out.get("has_keys2d", False))
        out["has_task_log"] = bool(out["has_task_log"])
        out["min_port_distance"] = float(out["min_port_distance"])
        out["train_state"] = float(out["train_state"])
        if verbose:
            print(f"  {rat}/{session}: cache hit")
        return out

    if verbose:
        print(f"  {rat}/{session}: building cache from SMB...")
    built = _build(rat, session, want_dual=want_dual)
    if built is None:
        return None
    _atomic_savez(cache_path(rat, session), built, built_dual=want_dual)
    # normalize return types to match the cache-hit path
    built["min_port_distance"] = float(built["min_port_distance"])
    built["train_state"] = float(built["train_state"])
    built["n_frames"] = int(built["n_frames"])
    built.pop("cache_version", None)
    return built


def warm_cache(sessions, want_dual=True):
    """Pre-build the cache for a list of (rat, session) tuples. Prints progress."""
    n_built = n_hit = n_skip = 0
    for rat, session in sessions:
        if is_valid(rat, session, want_dual=want_dual):
            n_hit += 1
            continue
        r = get_raw_session(rat, session, want_dual=want_dual)
        if r is None:
            print(f"  skipped {rat}/{session} (no/corrupt source)")
            n_skip += 1
            continue
        mb = os.path.getsize(cache_path(rat, session)) / 1e6
        print(f"  built {rat}/{session} ({mb:.1f} MB, dual={r['has_dual']})")
        n_built += 1
    print(f"warm_cache done: {n_built} built, {n_hit} already cached, {n_skip} skipped")
