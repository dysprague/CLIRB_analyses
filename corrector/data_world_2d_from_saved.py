"""Loader for the 2D-input corrector that uses the *saved* SLEAP outputs
directly (no video re-inference). Bridges the two index spaces:

  - sleap_keys_2D.npy  is indexed by *processed_frame* (one row per SLEAP
                       forward pass)
  - triang_keys_3D.npy is indexed by *cam0_frame*       (one row per video
                       frame; dropped frames are linearly interpolated by
                       behavior.py)

The mapping `processed_frame -> (cam0_frame, cam1_frame, cam2_frame)` lives
in `<session>/sleap/frame_mapping.csv`.

For each processed sample p we return:
    x_2d        (3, 23, 2)  — pixel coords per camera, post live cleanup
    x_conf      (3, 23)     — SLEAP detection confidences per camera
    x_triang_3d (23, 3)     — saved triangulated SLEAP 3D, native SLEAP world
    y_dannce_3d (23, 3)     — DANNCE 3D for the matching video frame, in
                              DANNCE world (median-filtered, same as
                              load_paired_world)
    cam_frames  (3,)        — per-camera video-frame index used by this sample
                              (mostly identical across cams; differs by 1-2
                              in ~5% of samples when frames were dropped)

Procrustes alignment and smoothing are deliberately NOT applied here — those
are training-pipeline concerns. The caller can use the existing
corrector.world_alignment.fit_procrustes on x_triang_3d / y_dannce_3d for the
calibration window.

`load_session_2d` returns everything for a session in one batch.
`PairedSessionDataset` is a torch Dataset wrapper that concatenates many
sessions for training, mirroring the layout of corrector.data_world.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from data_io import (load_aligned_data, load_sleap_dannce_keys,  # noqa: E402
                      load_sleap_keys_2d, load_sleap_keys_3d, sleap_path)

from corrector.data_world import DANNCE_MEDFILT  # noqa: E402
from corrector.data_world_2d import load_session_calibration  # noqa: E402


@dataclass
class SavedSession2D:
    """Container for one session's 2D-input corrector data."""
    rat: str
    session: str
    x_2d: np.ndarray        # (P, 3, 23, 2) float32 pixel coords
    x_conf: np.ndarray      # (P, 3, 23)    float32
    x_triang_3d: np.ndarray # (P, 23, 3)    float32 SLEAP world
    y_dannce_3d: np.ndarray # (P, 23, 3)    float32 DANNCE world
    cam_frames: np.ndarray  # (P, 3) int — per-cam video frame idx
    calibration: list       # list of 3 cam dicts (K, r, t, dist, cal_date)
    cal_date: str
    # Diagnostics:
    n_processed_raw: int    # original rows in 2D (before pruning)
    n_dropped_no_dannce: int  # samples dropped because matching DANNCE was OOB


def _load_frame_mapping(rat: str, session: str):
    """Returns a 2D int ndarray (P, 3) of [cam0_frame, cam1_frame, cam2_frame]
    indexed by row order, where row i corresponds to sleap_keys_2D.npy[i].

    The CSV's `processed_frame` column is 1-indexed (production code logs it
    AFTER incrementing the counter at the end of each loop), while
    sleap_keys_2D.npy is 0-indexed. We verify that the CSV is contiguous
    starting at 1 — i.e., row i has processed_frame == i+1."""
    import pandas as pd
    path = os.path.join(sleap_path(rat, session), "frame_mapping.csv")
    df = pd.read_csv(path, usecols=["processed_frame",
                                     "cam0_frame", "cam1_frame", "cam2_frame"])
    pf = df["processed_frame"].values
    if not (pf[0] == 1 and np.array_equal(pf, np.arange(1, len(pf) + 1))):
        raise RuntimeError(
            f"{rat}/{session}: frame_mapping.processed_frame is not "
            f"contiguous-from-1. Got "
            f"{pf[:5]}..{pf[-5:]}"
        )
    return df[["cam0_frame", "cam1_frame", "cam2_frame"]].values.astype(np.int64)


def load_session_2d(rat: str, session: str,
                     smooth_dannce: bool = True,
                     drop_first_seconds: float = 0.0,
                     skip_dannce: bool = False) -> SavedSession2D:
    """Load one session's 2D-input corrector data.

    smooth_dannce: apply median-25 filter to DANNCE 3D (consistent with
                   corrector.data_world.load_paired_world). Defaults True.
    drop_first_seconds: drop the first N seconds of processed samples
                       (caller typically does its own warmup handling).
    skip_dannce: for inference-only use on sessions that do not have DANNCE
                 keypoints yet. y_dannce_3d is filled with zeros and no
                 DANNCE-based drop filtering is applied. The corrector
                 inference path (correct_temporal_mlp_2d_reproj) never reads
                 y_dannce_3d, only the 2D/conf/triang/calib fields.
    """
    cam_frames = _load_frame_mapping(rat, session)  # (P, 3) int
    P = len(cam_frames)

    # --- 2D side: indexed by processed_frame ---
    sleap_2d = load_sleap_keys_2d(rat, session).astype(np.float32)  # (T_max, 3, 23, 3)
    if sleap_2d.shape[0] < P:
        raise RuntimeError(f"{rat}/{session}: sleap_keys_2D has {sleap_2d.shape[0]} "
                            f"rows, fewer than frame_mapping ({P})")
    sleap_2d = sleap_2d[:P]
    x_2d = sleap_2d[..., :2]                              # (P, 3, 23, 2)
    x_conf = sleap_2d[..., 2]                              # (P, 3, 23)

    # --- 3D triangulated (SLEAP world): indexed by video frame ---
    triang_full = load_sleap_keys_3d(rat, session).astype(np.float32)  # (V, 23, 3)
    # Index by cam0_frame for each processed sample.
    cam0_fr = cam_frames[:, 0]
    if cam0_fr.max() >= len(triang_full):
        raise RuntimeError(f"{rat}/{session}: cam0_frame max {cam0_fr.max()} "
                            f">= triang len {len(triang_full)}")
    x_triang_3d = triang_full[cam0_fr]                     # (P, 23, 3)

    if skip_dannce:
        y_dannce_3d = np.zeros_like(x_triang_3d)
        keep_mask = (cam0_fr >= 0)
        valid_dn = np.ones(P, dtype=bool)
        n_oob_sleap = 0
        n_dropped = 0
    else:
        # --- DANNCE 3D (target): indexed by SLEAP video frame via aligned_data ---
        keys = load_sleap_dannce_keys(rat, session)
        aligned = load_aligned_data(rat, session)
        dn = keys["dannce_keys_3D"]
        if dn.ndim == 4:
            dn = dn.squeeze(axis=1).transpose(0, 2, 1)
        else:
            dn = dn.transpose(0, 2, 1)
        dn = dn.astype(np.float32)
        if smooth_dannce:
            dn = median_filter(dn, size=(DANNCE_MEDFILT, 1, 1))
        didx_full = np.asarray(aligned["dannce_idx_for_sleap_cams"]).astype(np.int64).ravel()
        # didx_full is indexed by SLEAP video frame. Look up DANNCE row for each
        # cam0_frame, with clipping for out-of-range entries.
        keep_mask = (cam0_fr >= 0) & (cam0_fr < len(didx_full))
        n_oob_sleap = int((~keep_mask).sum())
        cam0_fr_clip = np.clip(cam0_fr, 0, len(didx_full) - 1)
        dn_rows = didx_full[cam0_fr_clip]
        valid_dn = (dn_rows >= 0) & (dn_rows < len(dn))
        n_dropped = int((~valid_dn | ~keep_mask).sum())
        dn_rows_clip = np.clip(dn_rows, 0, len(dn) - 1)
        y_dannce_3d = dn[dn_rows_clip]                         # (P, 23, 3)

    # Final keep mask: in-range on both sides + drop_first_seconds prefix
    final_keep = keep_mask & valid_dn
    if drop_first_seconds > 0:
        drop_n = int(round(drop_first_seconds * 20.0))  # SLEAP_HZ
        final_keep[:drop_n] = False

    x_2d = x_2d[final_keep]
    x_conf = x_conf[final_keep]
    x_triang_3d = x_triang_3d[final_keep]
    y_dannce_3d = y_dannce_3d[final_keep]
    cam_frames = cam_frames[final_keep]

    calib, cal_date = load_session_calibration(rat, session)

    return SavedSession2D(
        rat=rat, session=session,
        x_2d=x_2d, x_conf=x_conf,
        x_triang_3d=x_triang_3d, y_dannce_3d=y_dannce_3d,
        cam_frames=cam_frames,
        calibration=calib, cal_date=cal_date,
        n_processed_raw=P,
        n_dropped_no_dannce=n_dropped + n_oob_sleap,
    )


# ---------------------------------------------------------------------------
# Multi-session dataset wrapper (matches corrector.data_world layout)
# ---------------------------------------------------------------------------
class PairedSession2DDataset:
    """Wraps multiple sessions, exposing flat per-sample access. Mirrors
    corrector.data_world.WorldPairedDataset but keeps 2D + conf alongside
    the 3D pair.

    For each sample i it returns a dict:
        {
          'x_2d':      (3, 23, 2) float32
          'x_conf':    (3, 23)    float32
          'x_triang':  (23, 3)    float32  raw SLEAP triangulated 3D
          'y_dannce':  (23, 3)    float32  median-filtered DANNCE 3D
          'rat':       str
          'session':   str
          'cam_frame': int  cam0_frame (video frame index)
        }

    No Procrustes alignment is applied — let the trainer fit per-session
    Procrustes from x_triang on the calibration window and apply it on the
    fly. We expose `session_transforms` for trainers that want to cache.
    """

    def __init__(self, rat_to_sessions: dict[str, list[str]],
                 drop_first_seconds: float = 0.0,
                 verbose: bool = True):
        self.sessions: list[SavedSession2D] = []
        offsets = [0]
        for rat, sess_list in rat_to_sessions.items():
            for s in sess_list:
                try:
                    sd = load_session_2d(rat, s,
                                          drop_first_seconds=drop_first_seconds)
                except Exception as e:
                    if verbose:
                        print(f"  SKIP {rat}/{s}: {e}", flush=True)
                    continue
                if len(sd.x_2d) == 0:
                    if verbose:
                        print(f"  SKIP {rat}/{s}: empty after filtering",
                              flush=True)
                    continue
                self.sessions.append(sd)
                offsets.append(offsets[-1] + len(sd.x_2d))
                if verbose:
                    print(f"  loaded {rat}/{s}: P={len(sd.x_2d)}  "
                           f"cal={sd.cal_date}  "
                           f"dropped_no_dannce={sd.n_dropped_no_dannce}",
                           flush=True)
        self._offsets = np.asarray(offsets, dtype=np.int64)

    def __len__(self):
        return int(self._offsets[-1])

    def session_index(self, i: int) -> tuple[int, int]:
        """Returns (session_idx, local_idx) for a flat sample index."""
        si = int(np.searchsorted(self._offsets, i, side="right") - 1)
        return si, int(i - self._offsets[si])

    def __getitem__(self, i):
        si, li = self.session_index(i)
        sd = self.sessions[si]
        return {
            "x_2d": sd.x_2d[li],
            "x_conf": sd.x_conf[li],
            "x_triang": sd.x_triang_3d[li],
            "y_dannce": sd.y_dannce_3d[li],
            "rat": sd.rat,
            "session": sd.session,
            "cam_frame": int(sd.cam_frames[li, 0]),
        }
