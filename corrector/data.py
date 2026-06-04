"""
Paired (SLEAP, DANNCE) keypoint dataset for the corrector.

Inputs are loaded once from disk into a tensor cache, then served by frame.
Each sample is a single frame of egocentric-normalized keypoints:
    x  : (23, 3)  SLEAP egocentric coords
    y  : (23, 3)  DANNCE egocentric coords (regression target)

Sessions are split deterministically. The split is per-rat — corrections are
expected to be rat-specific until proven otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))                # for skeleton, data_io, config
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))  # for exp_utils

from exp_utils import load_session_data, smooth_keypoints
from skeleton import normalize_skeleton_batch
from data_io import get_sessions

SL_SMOOTH = ("median", 11)
DN_SMOOTH = ("median", 11)


def load_session_egocentric(rat: str, session: str):
    """Return (sleap_egocentric, dannce_egocentric) shape (T, 23, 3) each.

    Both arrays share the SLEAP time axis (DANNCE has been resampled by
    load_session_data via dannce_idx_for_sleap_cams).
    Returns (None, None) if the session is unloadable.
    """
    try:
        sleap_3d, dannce_3d, _ = load_session_data(rat, session)
    except Exception:
        return None, None
    sl = smooth_keypoints(sleap_3d, *SL_SMOOTH)
    dn = smooth_keypoints(dannce_3d, *DN_SMOOTH)
    sl_rot, _, _ = normalize_skeleton_batch(sl)
    dn_rot, _, _ = normalize_skeleton_batch(dn)
    return sl_rot.astype(np.float32), dn_rot.astype(np.float32)


def session_split(rat: str, train_frac=0.7, val_frac=0.15, seed=0):
    """Deterministic session-level split.

    Returns (train_sessions, val_sessions, test_sessions).
    Sessions are sorted chronologically before splitting so the test set
    contains the most recent sessions — this also tests for temporal drift.
    """
    sessions = get_sessions(rat=rat)["session"].tolist()
    sessions = sorted(sessions)  # chronological because sessions are YYYY_MM_DD_N
    rng = np.random.default_rng(seed)
    # Train/val drawn from the earlier portion; test = most recent sessions
    n = len(sessions)
    n_test = int(round((1 - train_frac - val_frac) * n))
    test = sessions[-n_test:] if n_test > 0 else []
    rest = sessions[:n - n_test]
    rng.shuffle(rest)
    n_train = int(round(train_frac / (train_frac + val_frac) * len(rest)))
    train = sorted(rest[:n_train])
    val = sorted(rest[n_train:])
    return train, val, test


class PairedKeypointDataset(Dataset):
    """One sample = one frame from one session.

    Pre-loads all session arrays into memory (rat × ~30 min × 20 Hz × 23 × 3 × 4 B
    ≈ 200 MB, fits easily in RAM).
    """

    def __init__(self, rat: str, sessions: list[str], drop_first_seconds: float = 30.0,
                 sleap_hz: float = 20.0, verbose: bool = True):
        self.rat = rat
        self.sessions = list(sessions)
        self._sleap = []  # list of (T_i, 23, 3) numpy
        self._dannce = []
        drop_n = int(round(drop_first_seconds * sleap_hz))
        for s in self.sessions:
            sl, dn = load_session_egocentric(rat, s)
            if sl is None or dn is None:
                if verbose:
                    print(f"  skip {rat}/{s}: load failed")
                continue
            T = min(len(sl), len(dn)) - drop_n
            if T <= 0:
                continue
            self._sleap.append(sl[drop_n:drop_n + T])
            self._dannce.append(dn[drop_n:drop_n + T])
        # Build a flat index: each entry maps -> (session_idx, frame_idx)
        self._index = []
        for i, arr in enumerate(self._sleap):
            for f in range(len(arr)):
                self._index.append((i, f))
        if verbose:
            print(f"  PairedKeypointDataset {rat}: {len(self._sleap)} sessions, "
                  f"{len(self._index):,} frames")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        si, fi = self._index[idx]
        x = torch.from_numpy(self._sleap[si][fi])    # (23, 3)
        y = torch.from_numpy(self._dannce[si][fi])
        return x, y


def make_loaders(rat: str, batch_size: int = 1024, num_workers: int = 0,
                 train_frac=0.7, val_frac=0.15, seed=0):
    """Convenience: returns (train_loader, val_loader, test_loader, splits).

    splits is a dict with the session lists for reporting.
    """
    from torch.utils.data import DataLoader
    train_s, val_s, test_s = session_split(rat, train_frac, val_frac, seed)
    print(f"{rat}: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test sessions")
    print("  building datasets...")
    train = PairedKeypointDataset(rat, train_s)
    val = PairedKeypointDataset(rat, val_s)
    test = PairedKeypointDataset(rat, test_s)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                   pin_memory=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=True),
        {"train": train_s, "val": val_s, "test": test_s},
    )
