"""
World-space paired (SLEAP, DANNCE) dataset for the corrector.

Per session, we:
  1. Load raw SLEAP and DANNCE 3D keypoints, smooth with median filters
     (SLEAP-11, DANNCE-25) and resample DANNCE to SLEAP frame rate.
  2. Fit a 7-DoF Procrustes alignment on a calibration epoch (first 5 min).
  3. Pre-align SLEAP into DANNCE world space using the fitted transform.
  4. Drop the first `drop_first_seconds` of each session (rat acclimation).
  5. Optionally drop sessions whose Procrustes residual is above a threshold
     (these are sessions where the SLEAP/DANNCE calibration disagrees badly,
     usually because one system isn't calibrated yet).

Each sample is one frame of (aligned_SLEAP_world, DANNCE_world), shape (23, 3).

The fitted Procrustes transforms are returned alongside so the renderer can
apply the same pipeline at inference time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import median_filter

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from data_io import get_sessions, load_aligned_data, load_sleap_dannce_keys
from corrector.world_alignment import (calibration_indices, fit_procrustes)

SLEAP_MEDFILT = 11
DANNCE_MEDFILT = 25
SLEAP_HZ = 20.0


def load_paired_world(rat: str, session: str):
    """Returns (sleap_world, dannce_world_at_sleap_times) shape (T, 23, 3) each.

    DANNCE has been resampled to SLEAP frame rate using dannce_idx_for_sleap_cams.
    Both arrays have been median-filtered. No alignment is applied.
    """
    keys = load_sleap_dannce_keys(rat, session)
    aligned = load_aligned_data(rat, session)
    sl = keys["sleap_keys_3D"].astype(np.float32)
    dn = keys["dannce_keys_3D"].astype(np.float32)
    if dn.ndim == 4:
        dn = dn.squeeze(axis=1).transpose(0, 2, 1)
    else:
        dn = dn.transpose(0, 2, 1)
    sl = median_filter(sl, size=(SLEAP_MEDFILT, 1, 1))
    dn = median_filter(dn, size=(DANNCE_MEDFILT, 1, 1))
    aidx = aligned["dannce_idx_for_sleap_cams"].astype(int).ravel()
    aidx = np.clip(aidx[: len(sl)], 0, len(dn) - 1)
    return sl, dn[aidx]


class WorldPairedDataset(Dataset):
    """Per-session Procrustes pre-alignment + flat frame index."""

    def __init__(self, rat_to_sessions: dict[str, list[str]],
                 calibration_minutes: float = 5.0,
                 calibration_n_sample: int = 1000,
                 drop_first_seconds: float = 30.0,
                 max_residual: float = 60.0,
                 verbose: bool = True):
        self._sl_aligned: list[np.ndarray] = []
        self._dn: list[np.ndarray] = []
        # Stash the Procrustes transforms for re-use at inference time
        # Keyed as (rat, session) -> dict.
        self.transforms: dict[tuple[str, str], dict] = {}
        self.session_residuals: list[tuple[str, str, float]] = []
        drop_n = int(round(drop_first_seconds * SLEAP_HZ))

        for rat, sessions in rat_to_sessions.items():
            for s in sessions:
                try:
                    sl, dn = load_paired_world(rat, s)
                except Exception as e:
                    if verbose:
                        print(f"  skip {rat}/{s}: load failed: {e}", flush=True)
                    continue
                if len(sl) < 1000:
                    if verbose:
                        print(f"  skip {rat}/{s}: only {len(sl)} frames", flush=True)
                    continue

                idx = calibration_indices(len(sl), calibration_minutes,
                                           SLEAP_HZ, calibration_n_sample, seed=0)
                if len(idx) < 100:
                    if verbose:
                        print(f"  skip {rat}/{s}: too few calibration frames",
                              flush=True)
                    continue
                tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
                self.transforms[(rat, s)] = tx
                self.session_residuals.append((rat, s, tx["residual"]))
                if tx["residual"] > max_residual:
                    if verbose:
                        print(f"  skip {rat}/{s}: residual={tx['residual']:.1f} mm "
                              f"> {max_residual} (uncalibrated)", flush=True)
                    continue

                sl_aligned = tx["apply"](sl).astype(np.float32)
                # Drop first `drop_first_seconds`
                if len(sl_aligned) <= drop_n:
                    continue
                self._sl_aligned.append(sl_aligned[drop_n:])
                self._dn.append(dn[drop_n:].astype(np.float32))

        # Build flat (session_idx, frame_idx) index
        self._index = []
        for i, arr in enumerate(self._sl_aligned):
            for f in range(len(arr)):
                self._index.append((i, f))
        if verbose:
            print(f"WorldPairedDataset: {len(self._sl_aligned)} sessions kept, "
                  f"{len(self._index):,} frames", flush=True)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        si, fi = self._index[idx]
        x = torch.from_numpy(self._sl_aligned[si][fi])
        y = torch.from_numpy(self._dn[si][fi])
        return x, y


class WindowedWorldDataset(Dataset):
    """Same data as WorldPairedDataset but each sample is a CTX-frame window.

    For each frame t in a session (after warmup), returns:
        x : (CTX, 23, 3)  Procrustes-aligned SLEAP at frames [t - CTX + 1, ..., t]
        y : (23, 3)       DANNCE at frame t  (regression target)

    Windows that would extend before the session start are skipped.
    """

    def __init__(self, rat_to_sessions: dict[str, list[str]],
                 ctx: int = 5,
                 calibration_minutes: float = 5.0,
                 calibration_n_sample: int = 1000,
                 drop_first_seconds: float = 30.0,
                 max_residual: float = 60.0,
                 verbose: bool = True):
        self.ctx = ctx
        self._sl_aligned: list[np.ndarray] = []
        self._dn: list[np.ndarray] = []
        self.transforms: dict[tuple[str, str], dict] = {}
        self.session_residuals: list[tuple[str, str, float]] = []
        drop_n = int(round(drop_first_seconds * SLEAP_HZ))

        for rat, sessions in rat_to_sessions.items():
            for s in sessions:
                try:
                    sl, dn = load_paired_world(rat, s)
                except Exception as e:
                    if verbose:
                        print(f"  skip {rat}/{s}: load failed: {e}", flush=True)
                    continue
                if len(sl) < 1000:
                    if verbose:
                        print(f"  skip {rat}/{s}: only {len(sl)} frames", flush=True)
                    continue
                idx = calibration_indices(len(sl), calibration_minutes,
                                           SLEAP_HZ, calibration_n_sample, seed=0)
                if len(idx) < 100:
                    continue
                tx = fit_procrustes(sl[idx], dn[idx], try_z_flip=True)
                self.transforms[(rat, s)] = tx
                self.session_residuals.append((rat, s, tx["residual"]))
                if tx["residual"] > max_residual:
                    if verbose:
                        print(f"  skip {rat}/{s}: residual={tx['residual']:.1f} mm "
                              f"> {max_residual} (uncalibrated)", flush=True)
                    continue
                sl_aligned = tx["apply"](sl).astype(np.float32)
                if len(sl_aligned) <= drop_n:
                    continue
                self._sl_aligned.append(sl_aligned[drop_n:])
                self._dn.append(dn[drop_n:].astype(np.float32))

        # Flat (session_idx, frame_idx) index — frames [ctx-1 .. T-1] are valid
        self._index = []
        for i, arr in enumerate(self._sl_aligned):
            for f in range(ctx - 1, len(arr)):
                self._index.append((i, f))
        if verbose:
            print(f"WindowedWorldDataset(ctx={ctx}): {len(self._sl_aligned)} sessions kept, "
                  f"{len(self._index):,} windows", flush=True)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        si, fi = self._index[idx]
        # Window from fi-ctx+1 to fi (inclusive)
        start = fi - self.ctx + 1
        x = torch.from_numpy(self._sl_aligned[si][start:fi + 1])  # (ctx, 23, 3)
        y = torch.from_numpy(self._dn[si][fi])                    # (23, 3)
        return x, y


def session_split_multi(rats: list[str], train_frac: float = 0.7,
                        val_frac: float = 0.15, seed: int = 0):
    """Per-rat chronological split, then concatenate splits across rats.

    Returns dict with 'train', 'val', 'test' keys, each mapping rat -> [sessions].
    """
    out = {"train": {}, "val": {}, "test": {}}
    rng = np.random.default_rng(seed)
    for rat in rats:
        sessions = sorted(get_sessions(rat=rat)["session"].tolist())
        n = len(sessions)
        n_test = int(round((1 - train_frac - val_frac) * n))
        test = sessions[-n_test:] if n_test > 0 else []
        rest = sessions[: n - n_test]
        rest_shuf = list(rest)
        rng.shuffle(rest_shuf)
        n_train = int(round(train_frac / (train_frac + val_frac) * len(rest_shuf)))
        train = sorted(rest_shuf[:n_train])
        val = sorted(rest_shuf[n_train:])
        out["train"][rat] = train
        out["val"][rat] = val
        out["test"][rat] = test
    return out
