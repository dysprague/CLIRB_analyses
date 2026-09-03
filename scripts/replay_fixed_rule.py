#!/usr/bin/env python
"""Replay recorded sessions through the CURRENT (fixed) online rule.

Answers: how many rewards would each session have delivered if the rig had been
running today's campy/behavior_rule.py -- the true 5-frame causal median and the
corrected COM_buffer -- against how many it actually delivered?

Usage:
    python scripts/replay_fixed_rule.py R4 R5 R6 --last 10 --out results/fixed_rule_replay.csv
"""
import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_ROOT  # noqa: E402
from clirb.session_replay import replay_session, summarize  # noqa: E402
import data_io as dio  # noqa: E402


def sessions_for(rat, last=None):
    root = os.path.join(DATA_ROOT, rat)
    s = sorted(d for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d)) and d[:2] == "20")
    return s[-last:] if last else s


def one(rat, session):
    rep = replay_session(rat, session)
    row = summarize(rep)
    p = rep["params"]
    row.update({
        # The template ACTUALLY archived into the session folder -- the config's
        # rat/template_num fields go stale, so do not report p.template_file.
        "template": dio.session_template_name(rat, session),
        "template_in_config": p.template_file,
        "task_id": p.task_id,
        "accepted_errors": p.accepted_errors,
        "min_port_distance": p.min_port_distance,
        "context_switch_s": p.context_switch_time,
        "reward_threshold": p.reward_threshold,
        "bounds_scalar": float(rep["template"]["bounds"]),
    })
    online = row.pop("online_rewards")
    fixed = row.pop("rewards")
    row["online_rewards"] = online
    row["fixed_rule_rewards"] = fixed
    row["delta"] = None if online is None else fixed - online
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rats", nargs="+")
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, failures = [], []
    for rat in args.rats:
        for session in sessions_for(rat, args.last):
            try:
                rows.append(one(rat, session))
                r = rows[-1]
                print(f"{rat}/{session:14s} online={str(r['online_rewards']):>4} "
                      f"fixed={r['fixed_rule_rewards']:>4} "
                      f"delta={str(r['delta']):>5}  "
                      f"matches={r['matches_ungated']:>4} "
                      f"task_id={r['task_id']}", flush=True)
            except Exception as e:
                failures.append((rat, session, f"{type(e).__name__}: {e}"))
                print(f"{rat}/{session:14s} SKIPPED -- {type(e).__name__}: {e}",
                      flush=True)

    df = pd.DataFrame(rows)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    if failures:
        print(f"\n{len(failures)} session(s) skipped:")
        for rat, s, why in failures:
            print(f"  {rat}/{s}: {why}")
    return df


if __name__ == "__main__":
    main()
