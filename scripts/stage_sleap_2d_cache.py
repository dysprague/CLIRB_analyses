"""Stage the SLEAP per-session files needed by the 2D-input corrector into a
local NVMe cache directory.

Reads the canonical session CSV, walks every (rat, session), and copies:
    <session>/sleap/sleap_keys_2D.npy
    <session>/sleap/triang_keys_3D.npy
    <session>/sleap/frame_mapping.csv
    <session>/sleap/calibration/<YYYY_MM_DD>/hires_cam{0,1,2}_params.mat

into:
    $SLEAP_LOCAL_CACHE/<rat>/<session>/sleap/...

After staging, config.sleap_path() returns the local path for every session
that's been copied — the loaders pick it up transparently.

Skip files that already exist with non-zero size.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from config import DATA_ROOT, SLEAP_LOCAL_CACHE
from data_io import get_sessions

FILES_FLAT = ["sleap_keys_2D.npy", "triang_keys_3D.npy", "frame_mapping.csv"]


def smb_sleap_dir(rat: str, session: str) -> str:
    return os.path.join(DATA_ROOT, rat, session, "sleap")


def local_sleap_dir(rat: str, session: str) -> str:
    return os.path.join(SLEAP_LOCAL_CACHE, rat, session, "sleap")


def needs_copy(src: str, dst: str) -> bool:
    if not os.path.isfile(src):
        return False
    if not os.path.isfile(dst):
        return True
    try:
        if os.path.getsize(dst) == 0:
            return True
    except OSError:
        return True
    return False


def copy_session(rat: str, session: str, verbose: bool = True) -> dict:
    src = smb_sleap_dir(rat, session)
    dst = local_sleap_dir(rat, session)
    os.makedirs(dst, exist_ok=True)

    counts = {"copied": 0, "skipped_existing": 0, "skipped_missing": 0,
              "bytes": 0}

    for fname in FILES_FLAT:
        s = os.path.join(src, fname)
        d = os.path.join(dst, fname)
        if not os.path.isfile(s):
            counts["skipped_missing"] += 1
            if verbose:
                print(f"  [missing] {fname}", flush=True)
            continue
        if needs_copy(s, d):
            shutil.copy2(s, d)
            counts["copied"] += 1
            counts["bytes"] += os.path.getsize(d)
        else:
            counts["skipped_existing"] += 1

    # Calibration sub-tree
    cal_src = os.path.join(src, "calibration")
    if os.path.isdir(cal_src):
        for sub in sorted(os.listdir(cal_src)):
            sub_src = os.path.join(cal_src, sub)
            if not os.path.isdir(sub_src):
                continue
            sub_dst = os.path.join(dst, "calibration", sub)
            os.makedirs(sub_dst, exist_ok=True)
            for f in os.listdir(sub_src):
                if not (f.startswith("hires_cam") and f.endswith("_params.mat")):
                    continue
                s = os.path.join(sub_src, f)
                d = os.path.join(sub_dst, f)
                if needs_copy(s, d):
                    shutil.copy2(s, d)
                    counts["copied"] += 1
                    counts["bytes"] += os.path.getsize(d)
                else:
                    counts["skipped_existing"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rats", nargs="+", default=["R1", "R2", "R3"])
    ap.add_argument("--limit", type=int, default=None,
                    help="copy at most this many sessions per rat (debug)")
    args = ap.parse_args()

    if not SLEAP_LOCAL_CACHE:
        print("SLEAP_LOCAL_CACHE is unset/empty — refusing to stage. Set it in "
              "config.py or as an env var.", flush=True)
        sys.exit(1)
    os.makedirs(SLEAP_LOCAL_CACHE, exist_ok=True)
    print(f"Cache root: {SLEAP_LOCAL_CACHE}", flush=True)

    total = {"copied": 0, "skipped_existing": 0, "skipped_missing": 0, "bytes": 0}
    t_start = time.time()
    for rat in args.rats:
        sessions = sorted(get_sessions(rat=rat)["session"].tolist())
        if args.limit is not None:
            sessions = sessions[:args.limit]
        print(f"\n=== {rat}: {len(sessions)} sessions ===", flush=True)
        for i, s in enumerate(sessions, 1):
            t0 = time.time()
            try:
                c = copy_session(rat, s, verbose=False)
            except Exception as e:
                print(f"  {rat}/{s}: ERROR {e}", flush=True)
                continue
            for k, v in c.items():
                total[k] = total.get(k, 0) + v
            mb = c["bytes"] / 1e6
            elapsed = time.time() - t0
            print(f"  [{i:3d}/{len(sessions)}] {rat}/{s}: "
                  f"copied={c['copied']} skip_existing={c['skipped_existing']} "
                  f"skip_missing={c['skipped_missing']} {mb:6.1f} MB {elapsed:5.1f}s",
                  flush=True)

    total_s = time.time() - t_start
    print(f"\nDone in {total_s:.1f}s. "
          f"copied={total['copied']} skip_existing={total['skipped_existing']} "
          f"skip_missing={total['skipped_missing']} "
          f"total={total['bytes']/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
