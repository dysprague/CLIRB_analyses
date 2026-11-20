"""Session-level rear variability computed over ALL matched rears at once.

Difference from `data/variability_summary/per_rear_all.csv`: that table carries
the ROLLING statistic (`sliding_window_variability`) -- std across the rear and
its ≤4 predecessors, averaged over time -- and a session summary then averages
those per-rear scalars. Here the session statistic is computed directly:

    stack every matched rear's (201, n_pcs) window for the session
    -> std ACROSS rears at each of the 201 timepoints
    -> mean over the 201 timepoints
    -> one scalar per PC

which is `clirb.native_dannce.group_variability` applied to the session's whole
matched set. Three consequences versus the rolling version:

  * it is the within-session spread over all matched rears, not local
    reproducibility over a 5-rear neighbourhood;
  * no unmatched rear can enter it (the rolling window mixed them in);
  * no autocorrelation between values and no shrinking-window edge case,
    because each session yields exactly one number per PC.

Caveat kept in the output: `group_variability` uses numpy's ddof=0, so a
session with few matched rears is biased low by sqrt(1 - 1/N) -- 2.4% at N=21,
0.25% at N=200. `n_matched` is written out so this is checkable.

Also emits the same statistic over all VALID rears and over template-like rears
(1.25x bounds), since they cost nothing once the session is loaded.

Usage
  .../envs/clirb_analysis/bin/python scripts/session_group_variability.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)

from clirb.native_dannce import (build_session_variability,   # noqa: E402
                                 group_variability)
import era_timeline as et                                      # noqa: E402
from build_variability_summary import (discover_qualifying_sessions,  # noqa: E402
                                       parse_session_date)

OUT = 'results/era_concentration/session_group_variability.csv'
N_PCS = 3


def _group(windows, mask, n_pcs=N_PCS):
    sel = [w[:, :n_pcs] for w, m in zip(windows, mask) if m]
    if len(sel) < 2:
        return None
    return group_variability(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rats', nargs='+', default=['R4', 'R5', 'R6'])
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    sessions = discover_qualifying_sessions(args.rats)
    print(f'{len(sessions)} qualifying sessions across {args.rats}', flush=True)

    rows, failures = [], []
    t0 = time.time()
    for i, (rat, session) in enumerate(sessions):
        try:
            d = build_session_variability(rat, session)
        except Exception as e:                         # noqa: BLE001
            failures.append((rat, session, repr(e)))
            print(f'  [{i+1}/{len(sessions)}] {rat}/{session}: FAILED -- {e!r}',
                  flush=True)
            continue
        w = d['pcs_windows']
        matched = np.asarray(d['rear_is_matched_valid'], dtype=bool)
        tlike = np.asarray(d['rear_is_template_like_valid'], dtype=bool)
        allv = np.ones(len(w), dtype=bool)

        row = dict(rat=rat, session=session, session_date=parse_session_date(session),
                   era=et.assign_era(rat, session),
                   n_valid=len(w), n_matched=int(matched.sum()),
                   n_template_like=int(tlike.sum()),
                   n_rewards=d['n_rewards'])
        for tag, mask in (('matched', matched), ('all', allv), ('tlike', tlike)):
            g = _group(w, mask)
            for k in range(N_PCS):
                row[f'{tag}_var_pc{k+1}'] = float(g[k]) if g is not None else np.nan
        rows.append(row)
        if (i + 1) % 20 == 0 or i == len(sessions) - 1:
            print(f'  [{i+1}/{len(sessions)}] {rat}/{session}: ok '
                  f'({len(rows)} sessions, {time.time()-t0:.0f}s)', flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nwrote {args.out}: {len(df)} sessions ({len(failures)} failed)')
    if failures:
        for f in failures:
            print('  FAILED:', f)
    print(df.groupby(['rat', 'era']).agg(
        n=('session', 'size'), matched=('n_matched', 'median'),
        pc1=('matched_var_pc1', 'median'), pc2=('matched_var_pc2', 'median')
    ).round(2).to_string())


if __name__ == '__main__':
    main()
