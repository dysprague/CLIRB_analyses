"""Offline re-creation of the live CLIRB pipeline.

Mirrors what runs in production via campy-CLIRB:
  unicam.GrabFrames    -> tf.image.resize(BGR_from_video -> RGB, 600x960) -> CHW
  process.ProcessFrames -> SLEAP NN forward pass -> find_peaks (argmax over conf
                           map) -> *4 / 0.5 + 0.5 (back to full-res px coords)
  behavior.ProcessBehavior -> per-cam COM distance filter -> NaN->prev fallback
                              -> triangulate (undistort + linear SVD)
                              -> normalize_skeleton -> rule.update

Public entry points used by the offline driver scripts:

  read_session_frames(rat, session, frame_indices) -> (n_frames, n_cams, 1200, 1920, 3) uint8 RGB
  preprocess_for_sleap(frames_rgb)                  -> (n_frames, n_cams, 3, 600, 960) float32
  find_peaks_from_confmaps(confmaps, threshold)     -> (n_frames, n_cams, 23, 2/3)
  postprocess_peaks(peaks, peak_vals)               -> (n_frames, n_cams, 23, 3) px in full-res
  preprocess_2d_behavior(peaks_with_vals)           -> per-cam COM filter + NaN->prev
  triangulate_session(peaks_2d, calib)              -> (n_frames, 23, 3) world coords
  build_calibration_for_session(rat, session)       -> dict with P/K/r/t/dist per cam
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from data_io import sleap_path  # noqa: E402
from corrector.data_world_2d import load_session_calibration  # noqa: E402

VIDEO_HEIGHT = 1200
VIDEO_WIDTH = 1920
MODEL_H = 600
MODEL_W = 960
N_KEYPOINTS = 23
# Used in process.py to recover full-res coords from confmap argmax.
# Confmap stride from input is 4, so peaks are in *quarter-resolution-of-input*
# pixel space. (input is 600x960 -> confmap 150x240) Multiplying by 4 gets you
# back to input pixel space (which is the downsampled 600x960). Then the
# divide-by-0.5 + 0.5 in process.py:455-457 looks like an off-by-one
# subpixel correction (centers the argmax cell). We replicate it.
PEAK_OUT_STRIDE = 4

CAMERA_DIRS = ("Camera0", "Camera1", "Camera2")


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------
def _open_session_videos(rat: str, session: str):
    """Return list of cv2.VideoCapture objects, one per camera, ordered cam0..2."""
    sp = sleap_path(rat, session)
    caps = []
    for cd in CAMERA_DIRS:
        path = os.path.join(sp, cd, "0.mp4")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open {path}")
        caps.append(cap)
    return caps


def read_session_frames(rat: str, session: str, frame_indices) -> np.ndarray:
    """Read the same frame_indices from all three cameras and return as
    (n_frames, n_cams, 1200, 1920, 3) uint8 in RGB order.

    Same video frame index for all cameras. Use read_session_frames_per_cam
    if you need per-camera index control (as the live pipeline does)."""
    frame_indices = list(frame_indices)
    n_cams = len(CAMERA_DIRS)
    per_cam = np.tile(np.asarray(frame_indices, dtype=int)[:, None],
                       (1, n_cams))
    return read_session_frames_per_cam(rat, session, per_cam)


def read_session_frames_per_cam(rat: str, session: str,
                                  per_cam_indices: np.ndarray) -> np.ndarray:
    """Per-camera control over which video frame to read.

    per_cam_indices: (n_samples, n_cams) int — video frame index per camera
    per processed sample. This matches the live pipeline, where each cam's
    `frame_id` for a given processed sample is the *latest* frame that camera
    had buffered when the processor pulled from the queue, and these can
    differ by 1-2 frames between cameras when frames are dropped.

    Returns (n_samples, n_cams, 1200, 1920, 3) uint8 RGB (cv2 BGR -> RGB)."""
    per_cam_indices = np.asarray(per_cam_indices, dtype=int)
    n_samples, n_cams = per_cam_indices.shape
    assert n_cams == len(CAMERA_DIRS)
    caps = _open_session_videos(rat, session)
    out = np.empty((n_samples, n_cams, VIDEO_HEIGHT, VIDEO_WIDTH, 3),
                   dtype=np.uint8)
    try:
        for ci, cap in enumerate(caps):
            for si in range(n_samples):
                fi = int(per_cam_indices[si, ci])
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr_bgr = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"cam{ci} frame {fi}: read failed")
                if fr_bgr.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                    raise RuntimeError(
                        f"cam{ci} frame {fi}: unexpected shape "
                        f"{fr_bgr.shape}")
                out[si, ci] = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
    finally:
        for cap in caps:
            cap.release()
    return out


def load_frame_mapping(rat: str, session: str):
    """Returns a DataFrame with columns
    cam0_frame, cam1_frame, cam2_frame, processed_frame
    indexed by `processed_frame` for O(1) lookup."""
    import pandas as pd
    path = os.path.join(sleap_path(rat, session), "frame_mapping.csv")
    df = pd.read_csv(path)
    return df.set_index("processed_frame", drop=False)


# ---------------------------------------------------------------------------
# Preprocess (matches unicam.GrabFrames)
# ---------------------------------------------------------------------------
def preprocess_for_sleap(frames_rgb: np.ndarray) -> np.ndarray:
    """(n_frames, n_cams, 1200, 1920, 3) uint8 RGB
       -> (n_frames * n_cams, 600, 960, 3) float32 in [0, 1].

    The .og SavedModel signature is (None, 600, 960, 3) -> HWC. The live
    pipeline does an extra CHW transpose for the TRT export, but for direct
    .og inference we want HWC.

    Resize uses tf.image.resize(method='bilinear', antialias=False) to
    exactly mirror unicam.GrabFrames line 327."""
    import tensorflow as tf
    n_frames, n_cams, h, w, c = frames_rgb.shape
    assert (h, w, c) == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)
    flat = frames_rgb.reshape(n_frames * n_cams, h, w, c)
    flat_tf = tf.constant(flat)
    resized = tf.cast(
        tf.image.resize(flat_tf, size=[MODEL_H, MODEL_W],
                         method="bilinear",
                         preserve_aspect_ratio=False, antialias=False),
        tf.uint8,
    )
    out = tf.cast(resized, tf.float32) / 255.0
    return out.numpy()


# ---------------------------------------------------------------------------
# Peak finding (mirrors process.find_peaks_cpu_no_transpose)
# ---------------------------------------------------------------------------
def find_peaks_from_confmaps(confmaps: np.ndarray,
                              threshold: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """confmaps: (B, n_kp, H_cm, W_cm) float32 in [0,1]
    Returns peaks (B, n_kp, 2) in (x, y) confmap-pixel coords (NaN if conf<threshold),
            vals  (B, n_kp) float32.
    Exact replica of process.find_peaks_cpu_no_transpose."""
    B, C, H, W = confmaps.shape
    flat = confmaps.reshape(B, C, H * W)
    idx = np.argmax(flat, axis=2)
    vals = np.max(flat, axis=2)
    y = (idx // W).astype(np.float32)
    x = (idx % W).astype(np.float32)
    peaks = np.stack([x, y], axis=-1)
    mask = (vals >= threshold)[..., np.newaxis]
    peaks = np.where(mask, peaks, np.nan)
    return peaks, vals


def postprocess_peaks(peaks: np.ndarray, peak_vals: np.ndarray) -> np.ndarray:
    """Recover full-res pixel coords from confmap argmax.

    Mirrors ProcessFrames lines 455-457:
        np.multiply(peaks_numpy, 4.0, out=peaks_numpy)
        np.divide(peaks_numpy, 0.5, out=peaks_numpy)   # i.e. *2
        np.add(peaks_numpy, 0.5, out=peaks_numpy)

    The *4 maps confmap-px back to model-input px (600x960). The *2 + 0.5
    looks like an off-by-half centering — but: the saved sleap_keys_2D.npy
    is in *full-resolution video* px coords (1920x1200). So an additional
    scale by (1920/960, 1200/600) = (2, 2) should be applied to get to
    full-res. That's likely what the /0.5 is doing in production.

    Returns (B, n_kp, 3) where [:,:,:2] is (x, y) full-res px, [:,:,2] is conf.
    """
    out = np.empty(peaks.shape[:-1] + (3,), dtype=np.float32)
    p = peaks.astype(np.float32) * float(PEAK_OUT_STRIDE)  # *4 -> 600x960 px
    p = p / 0.5                                              # *2 -> 1920x1200
    p = p + 0.5                                              # half-pixel center
    out[..., :2] = p
    out[..., 2] = peak_vals
    return out


# ---------------------------------------------------------------------------
# Behavior-side cleanup (mirrors behavior.ProcessBehavior lines 394-405)
# ---------------------------------------------------------------------------
def preprocess_2d_behavior(peaks_with_vals: np.ndarray,
                            com_distance_thresh: float = 400.0) -> np.ndarray:
    """peaks_with_vals: (T, n_cams, n_kp, 3) where last dim is (x, y, conf).
    For each frame:
      - per-camera COM-distance filter: distance > 400 px -> NaN
      - NaN -> previous-frame value
    Returns a fresh array of the same shape. Confidence values are passed
    through unchanged."""
    T, C, K, _ = peaks_with_vals.shape
    out = peaks_with_vals.copy()
    prev = np.zeros((C, K, 2), dtype=out.dtype)
    for t in range(T):
        xy = out[t, :, :, :2]
        # COM distance per cam
        # Treat conf==0 / NaN xy as missing — exclude from COM calc to avoid
        # COM being pulled to (0,0).
        for c in range(C):
            valid = np.isfinite(xy[c]).all(axis=1) & (out[t, c, :, 2] > 0)
            if valid.sum() == 0:
                continue
            com = xy[c, valid].mean(axis=0)
            dist = np.linalg.norm(xy[c] - com, axis=1)
            far = dist > com_distance_thresh
            xy[c, far] = np.nan
        # NaN -> prev
        nan_mask = np.isnan(xy).any(axis=-1)
        xy[nan_mask] = prev[nan_mask]
        out[t, :, :, :2] = xy
        prev = xy.copy()
    return out


# ---------------------------------------------------------------------------
# Triangulation (mirrors behavior.triangulate)
# ---------------------------------------------------------------------------
def _undistort_points(points_2d: np.ndarray, K: np.ndarray,
                       dist: np.ndarray) -> np.ndarray:
    """OpenCV undistortPoints: pixel -> normalized undistorted coords."""
    pts = points_2d.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, np.asarray(dist, dtype=np.float64))
    return undist.reshape(-1, 2)


def _triangulate_point(undist_points, P_list):
    """SVD-DLT triangulation of a single 3D point from m views.
    undist_points: list/array (m, 2). P_list: list of (3, 4)."""
    m = len(P_list)
    A = np.zeros((2 * m, 4), dtype=np.float64)
    for i in range(m):
        x, y = undist_points[i]
        P = P_list[i]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    try:
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        if abs(Xh[3]) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        Xh = Xh / Xh[3]
        return Xh[:3]
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float64)


def build_calibration_for_session(rat: str, session: str):
    """Returns a dict suitable for triangulate_session:
        {'P_list': [3x(3,4)], 'K_list': [3x(3,3)],
         'dist_coefs': [3x(4,)], 'cal_date': str}
    The P matrix here exactly matches what behavior.py builds via
    build_projection_matrix(K=identity, R, t)."""
    calib_dicts, cal_date = load_session_calibration(rat, session)
    P_list, K_list, dist = [], [], []
    for cd in calib_dicts:
        # behavior.build_projection_matrix(K, R, t) returns I @ [R | t]
        P = np.hstack([cd["r"], cd["t"].reshape(3, 1)])
        P_list.append(P.astype(np.float64))
        K_list.append(cd["K"].astype(np.float64))
        dist.append(cd["dist"].astype(np.float64))
    return {"P_list": P_list, "K_list": K_list, "dist_coefs": dist,
            "cal_date": cal_date}


def triangulate_frame(keys_2d: np.ndarray, calib: dict) -> np.ndarray:
    """keys_2d: (n_cams, n_kp, 2) — full-res pixel coords. NaN entries are
    triangulated as-is and will produce nonsense; rely on the upstream
    preprocess_2d_behavior to have already replaced them.
    Returns (n_kp, 3)."""
    P_list = calib["P_list"]
    K_list = calib["K_list"]
    dist = calib["dist_coefs"]
    n_cams, n_kp, _ = keys_2d.shape
    undist = np.zeros_like(keys_2d, dtype=np.float64)
    for ci in range(n_cams):
        undist[ci] = _undistort_points(keys_2d[ci], K_list[ci], dist[ci])
    pts3d = np.zeros((n_kp, 3), dtype=np.float64)
    for j in range(n_kp):
        pts3d[j] = _triangulate_point(undist[:, j, :], P_list)
    return pts3d


def triangulate_session(peaks_2d: np.ndarray, calib: dict) -> np.ndarray:
    """peaks_2d: (T, n_cams, n_kp, 2) full-res px. Returns (T, n_kp, 3)."""
    T = peaks_2d.shape[0]
    out = np.zeros((T, peaks_2d.shape[2], 3), dtype=np.float64)
    for t in range(T):
        out[t] = triangulate_frame(peaks_2d[t], calib)
    return out


# ---------------------------------------------------------------------------
# SLEAP forward pass (light wrapper around tf.saved_model.load)
# ---------------------------------------------------------------------------
class SleapModel:
    """Loads a SavedModel and exposes a numpy-in numpy-out predict()."""

    def __init__(self, model_dir: str):
        import tensorflow as tf
        self._tf = tf
        loaded = tf.saved_model.load(model_dir)
        # Try serving_default; otherwise pick the only signature.
        sigs = loaded.signatures
        if "serving_default" in sigs:
            self._fn = sigs["serving_default"]
        else:
            keys = list(sigs.keys())
            if len(keys) != 1:
                raise RuntimeError(f"Multiple signatures available: {keys}")
            self._fn = sigs[keys[0]]
        # Discover input name
        sig = self._fn.structured_input_signature[1]
        self._input_name = next(iter(sig.keys()))
        # Discover output names
        self._output_names = list(self._fn.structured_outputs.keys())

    def predict(self, x_chw_float: np.ndarray) -> dict:
        """x_chw_float: (B, 3, 600, 960) float32 in [0,1].
        Returns dict of output_name -> numpy array."""
        x = self._tf.constant(x_chw_float)
        outs = self._fn(**{self._input_name: x})
        return {k: v.numpy() for k, v in outs.items()}

    @property
    def output_names(self):
        return list(self._output_names)

    @property
    def input_name(self):
        return self._input_name
