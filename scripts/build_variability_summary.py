"""Build cross-session/cross-animal rear-variability summary tables.

Runs clirb.native_dannce.build_session_variability ONCE per qualifying
session (has dual SLEAP+DANNCE data + a behavior log) across the given rats,
and writes two combined tables from that single pass. Feeds
multi_session_variability.ipynb.

  per_rear.csv            one row per valid (in-bounds) rear: sliding-window
                          variability (PC1-3), is_matched (standard bounds),
                          is_template_like (recommendation #2: matched at
                          1.25x-wide bounds -- a superset of is_matched),
                          reward labeling, session metadata.
  per_session_learning.csv  one row per session with >= 20 MATCHED (standard-
                          bounds) rears: first-10-match vs last-10-match
                          group variability (PC1-3), for the
                          learning-across-training analysis. Sessions with
                          fewer than 20 matches are absent, not padded.

Sessions with fewer than --min_rewards online-triggered rewards are skipped
entirely (excluded from both tables), not merely filtered downstream.

Usage:
  .../envs/clirb_analysis/bin/python scripts/build_variability_summary.py \
      --rats R4 R5 R6 --min_rewards 10 --out data/variability_summary/per_rear.csv \
      --learning_out data/variability_summary/per_session_learning.csv

Sessions are independent and mostly SMB-cache hits (session_cache.py), so a
full run across ~150 sessions is expected to take on the order of minutes,
not hours -- but run it in the background regardless, since any individual
uncached session still pays a real SMB read.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT  # noqa: E402
from clirb.native_dannce import (build_session_variability,  # noqa: E402
                                 first_last_match_variability)


def discover_qualifying_sessions(rats, data_root=DATA_ROOT):
    """(rat, session) pairs with DANNCE keys, a behavior log, and SLEAP triang
    data -- the minimum for build_session_variability to have a chance."""
    out = []
    for rat in rats:
        rat_dir = os.path.join(data_root, rat)
        if not os.path.isdir(rat_dir):
            continue
        for session in sorted(os.listdir(rat_dir)):
            sd = os.path.join(rat_dir, session)
            has_dual = os.path.isfile(os.path.join(sd, "processed", "sleap_dannce_keys.mat"))
            has_log = os.path.isfile(os.path.join(sd, "sleap", "behavior_log.csv"))
            has_tri = os.path.isfile(os.path.join(sd, "sleap", "triang_keys_3D.npy"))
            if has_dual and has_log and has_tri:
                out.append((rat, session))
    return out


def parse_session_date(session):
    """'2026_07_27_1' -> '2026-07-27' (drop the trailing same-day run index)."""
    parts = session.split("_")
    return "-".join(parts[:3]) if len(parts) >= 3 else session


def rear_rows_from(d, rat, session):
    n = d["n_valid"]
    if n == 0:
        return []
    date = parse_session_date(session)
    return [{
        "rat": rat, "session": session, "session_date": date,
        "rear_idx": i, "t_min": d["t_rear_min"][i],
        "is_matched": bool(d["rear_is_matched_valid"][i]),
        "is_template_like": bool(d["rear_is_template_like_valid"][i]),
        "is_rewarded": bool(d["is_rewarded"][i]),
        "rears_since_reward": int(d["rears_since_reward"][i]),
        "frames_since_reward": d["frames_since_reward"][i],
        "var_pc1": d["variability"][i, 0],
        "var_pc2": d["variability"][i, 1],
        "var_pc3": d["variability"][i, 2],
        "n_rears_detected": d["n_rears"], "n_rewards_session": d["n_rewards"],
        "session_duration_min": d["session_duration_min"],
    } for i in range(n)]


def learning_row_from(d, rat, session, n_match=10, min_matches=20):
    r = first_last_match_variability(d, n=n_match, min_matches=min_matches)
    if r is None:
        return None
    date = parse_session_date(session)
    row = {"rat": rat, "session": session, "session_date": date,
          "n_matches": r["n_matches"]}
    for k in range(3):
        row[f"first10_var_pc{k + 1}"] = r["first"][k]
        row[f"last10_var_pc{k + 1}"] = r["last"][k]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rats", nargs="+", default=["R4", "R5", "R6"])
    ap.add_argument("--out", default="data/variability_summary/per_rear.csv")
    ap.add_argument("--learning_out", default="data/variability_summary/per_session_learning.csv")
    ap.add_argument("--half_win_s", type=float, default=2.0)
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--reward_tol_s", type=float, default=1.0)
    ap.add_argument("--template_like_bound_mult", type=float, default=1.25)
    ap.add_argument("--n_match", type=int, default=10)
    ap.add_argument("--min_matches", type=int, default=20)
    ap.add_argument("--min_rewards", type=int, default=0,
                    help="sessions with fewer online-triggered rewards than this are "
                         "skipped entirely -- excluded from BOTH output tables, not "
                         "just filtered downstream.")
    args = ap.parse_args()

    sessions = discover_qualifying_sessions(args.rats)
    print(f"{len(sessions)} qualifying sessions across {args.rats}")

    rear_rows, learning_rows, failures, skipped_low_reward = [], [], [], []
    t_start = time.time()
    for i, (rat, session) in enumerate(sessions):
        try:
            d = build_session_variability(
                rat, session, half_win_s=args.half_win_s, lookback=args.lookback,
                reward_tol_s=args.reward_tol_s,
                template_like_bound_mult=args.template_like_bound_mult)
            if d["n_rewards"] < args.min_rewards:
                skipped_low_reward.append((rat, session, d["n_rewards"]))
                continue
            rear_rows.extend(rear_rows_from(d, rat, session))
            lrow = learning_row_from(d, rat, session, n_match=args.n_match,
                                     min_matches=args.min_matches)
            if lrow is not None:
                learning_rows.append(lrow)
        except Exception as e:
            failures.append((rat, session, repr(e)))
            print(f"  [{i + 1}/{len(sessions)}] {rat}/{session}: FAILED -- {e!r}")
            continue
        if (i + 1) % 10 == 0 or i == len(sessions) - 1:
            elapsed = time.time() - t_start
            print(f"  [{i + 1}/{len(sessions)}] {rat}/{session}: ok "
                 f"({len(rear_rows)} rear-rows, {len(learning_rows)} learning-rows, "
                 f"{len(skipped_low_reward)} skipped (<{args.min_rewards} rewards) "
                 f"so far, {elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rear_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rear-rows from {len(sessions) - len(failures)}/"
         f"{len(sessions)} sessions -> {args.out}")

    ldf = pd.DataFrame(learning_rows)
    os.makedirs(os.path.dirname(args.learning_out), exist_ok=True)
    ldf.to_csv(args.learning_out, index=False)
    print(f"wrote {len(ldf)} sessions with >= {args.min_matches} matches "
         f"-> {args.learning_out}")

    print(f"{len(skipped_low_reward)} sessions skipped for <{args.min_rewards} rewards")
    if failures:
        print(f"{len(failures)} sessions failed:")
        for rat, session, err in failures:
            print(f"  {rat}/{session}: {err}")


if __name__ == "__main__":
    main()
