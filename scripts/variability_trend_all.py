"""Per-session rear-variability trend, template-matched rears only, across ALL
sessions -- with the pre/post-lesion boundary marked.

Reproduces `multi_session_variability.ipynb` §3b/§3c (Fig. 9b/9c) but without
the 2026-08-07 cutoff: every session with more than MIN_MATCHED template-matched
rears is included, and the lesion boundary is drawn per rat.

Each point is the mean sliding-window variability over that session's MATCHED
rears only (standard bounds), in RAW PC units -- deliberately not
session-normalised, since normalising by a session's own median would remove
exactly the between-session drift this panel exists to show.

**Raw PC units are only comparable within a template epoch.** The PCA basis
comes from the session's own template, and templates are refit in place, so a
refit introduces a scale discontinuity unrelated to behaviour. Template-content
changes are therefore drawn as vertical rules (▲ above the top row): read
within-epoch trends, never a jump across a rule. The line itself is broken
only at recording gaps longer than GAP_DAYS. That also means the
pre/post-lesion contrast is NOT a clean statistical comparison in these units,
which is why no p-value is drawn.

Input comes from `scripts/build_variability_summary.py` run over all sessions:

  python scripts/build_variability_summary.py --rats R4 R5 R6 --min_rewards 0 \\
      --out data/variability_summary/per_rear_all.csv \\
      --learning_out data/variability_summary/per_session_learning_all.csv

Usage
  .../envs/clirb_analysis/bin/python scripts/variability_trend_all.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)

import era_timeline as et                       # noqa: E402

PER_REAR = 'data/variability_summary/per_rear_all.csv'
GROUP = 'results/era_concentration/session_group_variability.csv'
TEMPLATES = 'results/era_timeline/session_counts.csv'   # carries template_sha
OUT = 'results/era_concentration'
RATS = ['R4', 'R5', 'R6']
MIN_MATCHED = 20          # strictly MORE than this many matched rears
GAP_DAYS = 6

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880', '#e4e3df'
SERIES = {'R4': '#2a78d6', 'R5': '#eb6834', 'R6': '#1baf7a'}
ERA_BAND = et.ERA_BAND


def load(stat='group'):
    """stat='group'  -> one number per session: std across ALL matched rears at
                        each timepoint, averaged over timepoints
                        (clirb.native_dannce.group_variability).
       stat='rolling'-> the per-rear 5-rear rolling statistic, averaged over
                        the session's matched rears (the original Fig. 9b form).
    """
    if stat == 'group':
        g = pd.read_csv(GROUP)
        g = g.rename(columns={f'matched_var_pc{k}': f'var_pc{k}' for k in (1, 2, 3)})
        allsess = g[['rat', 'session', 'n_matched', 'era']].copy()
    elif stat == 'rolling':
        d = pd.read_csv(PER_REAR)
        m = d[d.is_matched]
        g = (m.groupby(['rat', 'session', 'session_date'])
               .agg(n_matched=('var_pc1', 'size'),
                    var_pc1=('var_pc1', 'mean'),
                    var_pc2=('var_pc2', 'mean'),
                    var_pc3=('var_pc3', 'mean'))
               .reset_index())
        g['era'] = [et.assign_era(r, s) for r, s in zip(g.rat, g.session)]
        allsess = (d.groupby(['rat', 'session'])
                     .agg(n_rears=('is_matched', 'size'),
                          n_matched=('is_matched', 'sum')).reset_index())
        allsess['era'] = [et.assign_era(r, s) for r, s in zip(allsess.rat, allsess.session)]
    else:
        raise ValueError(stat)

    g = g[g.n_matched > MIN_MATCHED].copy()
    g['date_dt'] = pd.to_datetime(g.session_date)
    tmpl = pd.read_csv(TEMPLATES)[['rat', 'session', 'template_sha']]
    g = g.merge(tmpl, on=['rat', 'session'], how='left')
    g = g.sort_values(['rat', 'date_dt', 'session']).reset_index(drop=True)
    return g, allsess


def figure(g, allsess, stat='group'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.transforms as mtransforms

    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
        'legend.frameon': False, 'figure.dpi': 170})

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4), constrained_layout=True)
    xlo = g.date_dt.min() - pd.Timedelta(days=4)
    xhi = g.date_dt.max() + pd.Timedelta(days=4)

    for col, rat in enumerate(RATS):
        sub = g[g.rat == rat].reset_index(drop=True)
        runs = et._era_runs(sub)
        edges = [xlo]
        for i in range(len(runs) - 1):
            edges.append(runs[i][2] + (runs[i + 1][1] - runs[i][2]) / 2)
        edges.append(xhi)

        # break the line at multi-day gaps AND at every template-content change
        day_gap = sub.date_dt.diff().dt.days.to_numpy()
        sha = sub.template_sha.to_numpy()
        new_tmpl = np.r_[False, sha[1:] != sha[:-1]]
        # Break only at real recording gaps. Template changes get a vertical
        # rule instead: fragmenting the series at every refit destroys the
        # trend the panel exists to show, but the reader still needs to see
        # where raw PC units stop being comparable.
        brk = np.isfinite(day_gap) & (day_gap > GAP_DAYS)

        for row, pc in enumerate(['var_pc1', 'var_pc2']):
            ax = axes[row][col]
            blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            for (era, _, _), lo, hi in zip(runs, edges[:-1], edges[1:]):
                ax.axvspan(lo, hi, color=ERA_BAND.get(era, '#eee'), lw=0, zorder=0)
            for i, (era, _, _) in enumerate(runs[1:], start=1):
                if era == 'post_lesion':
                    ax.axvline(edges[i], color='#c0392b', lw=2.0, ls='--', zorder=4)
                    ax.text(edges[i], 0.985, '  lesion', transform=blend, ha='left',
                            va='top', fontsize=9, color='#c0392b', fontweight='bold',
                            zorder=6)

            y = sub[pc].astype(float).to_numpy().copy()
            y[brk] = np.nan
            ax.plot(sub.date_dt, y, color=SERIES[rat], lw=1.4, zorder=3,
                    label=('std across all matched rears' if stat == 'group'
                           else 'mean of 5-rear rolling variability')
                          if (row == 0 and col == 0) else None)
            ax.plot(sub.date_dt, sub[pc], color=SERIES[rat], marker='o', ms=3.4,
                    ls='none', zorder=3)

            ch = np.where(new_tmpl)[0]
            if len(ch):
                for x in sub.date_dt.iloc[ch]:
                    ax.axvline(x, color='#7a5fb0', lw=0.8, alpha=0.5, zorder=2)
                if row == 0:
                    ax.plot(sub.date_dt.iloc[ch], np.full(len(ch), 1.015),
                            transform=blend, marker='^', ms=5, ls='none',
                            color='#7a5fb0', zorder=6, clip_on=False,
                            label='template changed' if col == 0 else None)

            ax.set_xlim(xlo, xhi)
            ax.grid(axis='y', alpha=0.9, zorder=0)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(f'mean {"PC1" if row == 0 else "PC2"} variability\n'
                              '(matched rears only, raw)')
            if row == 0:
                npre = int((sub.era == 'training').sum())
                npost = int((sub.era == 'post_lesion').sum())
                a = allsess[(allsess.rat == rat) & (allsess.era == 'post_lesion')]
                pre_sha = sub[sub.era == 'training'].template_sha
                post_sha = sub[sub.era == 'post_lesion'].template_sha
                same = (len(pre_sha) and len(post_sha)
                        and pre_sha.iloc[-1] == post_sha.iloc[0])
                note = ('same template across lesion' if same
                        else '⚠ template changed at lesion')
                ax.set_title(f'{rat}   {npre} pre / {npost} post'
                             f'   ({npost}/{len(a)} post pass)\n{note}',
                             fontsize=10, color=INK if same else '#c0392b')
            if row == 1:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.set_xlabel('session date (2026)')
            else:
                ax.set_xticklabels([])

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=3, fontsize=9)
    how = ('std across ALL matched rears at each timepoint, averaged over time'
           if stat == 'group' else
           'mean over matched rears of the 5-rear rolling variability')
    fig.suptitle(f'Per-session variability, template-matched rears, ALL sessions with '
                 f'>{MIN_MATCHED} matched rears — {how}\n'
                 'Raw PC units are comparable only within a template epoch — vertical '
                 'rules mark ▲ template changes; the line breaks only at recording gaps > 6 days.',
                 fontsize=11.5, ha='left', x=0.008, color=INK2, linespacing=1.5)
    os.makedirs(OUT, exist_ok=True)
    stem = 'F5_variability_trend_all' if stat == 'rolling' else 'F6_variability_group_all'
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{stem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT}/{stem}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stat', choices=['group', 'rolling'], default='group')
    args = ap.parse_args()
    g, allsess = load(args.stat)
    g.to_csv(f'{OUT}/variability_trend_sessions_{args.stat}.csv', index=False)
    print(f'wrote {OUT}/variability_trend_sessions_{args.stat}.csv ({len(g)} sessions)')
    print(g.groupby(['rat', 'era']).agg(
        n=('session', 'size'), matched=('n_matched', 'median'),
        pc1=('var_pc1', 'median'), pc2=('var_pc2', 'median'),
        templates=('template_sha', 'nunique')).round(2).to_string())
    figure(g, allsess, args.stat)
