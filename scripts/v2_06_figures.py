"""Corrector v2, step 6 — figures for the summary interface.

Palette: dataviz reference instance, first three categorical slots
(blue/orange/aqua), the documented all-pairs-validated subset.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from config import NODES

RES = 'results/corrector_v2'
FIG = 'results/figures/corrector_v2'
os.makedirs(FIG, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880', '#e4e3df'
SERIES = {'deployed': '#4a3aa7', 'uncorrected': MUTED, 'temporal': '#2a78d6', 'velacc': '#eb6834', 'aspect': '#1baf7a'}
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.8, 'lines.linewidth': 2.0,
    'grid.color': GRID, 'grid.linewidth': 0.7, 'legend.frameon': False, 'figure.dpi': 160})


def tidy(ax):
    ax.grid(axis='y', alpha=0.9, zorder=0); ax.set_axisbelow(True)


def fig_tier1():
    """Keypoint error: baseline vs each corrector, overall and per keypoint."""
    tr = json.load(open(f'{RES}/train_results.json'))
    names = [n for n in ['temporal', 'velacc', 'aspect'] if n in tr]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                             gridspec_kw={'width_ratios': [1, 2.4]},
                             constrained_layout=True)
    ax = axes[0]
    base = tr[names[0]]['test_med_baseline']
    vals = [base] + [tr[n]['test_med_corrected'] for n in names]
    labs = ['uncorrected'] + names
    cols = [SERIES[l] for l in labs]
    b = ax.bar(range(len(vals)), vals, color=cols, edgecolor=SURFACE, lw=1.2)
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.2f}', (i, v), textcoords='offset points', xytext=(0, 4),
                    ha='center', fontsize=9, weight='bold' if i else 'normal', color=INK)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=15)
    ax.set_ylabel('median |pose − DANNCE| (mm)')
    ax.set_title('Held-out keypoint error', fontsize=10, color=INK); tidy(ax)

    ax = axes[1]
    kb = np.array(tr[names[0]]['per_kp_baseline'])
    order = np.argsort(-kb)
    x = np.arange(len(NODES))
    ax.plot(x, kb[order], 'o-', color=MUTED, ms=4, label='uncorrected')
    for n in names:
        kc = np.array(tr[n]['per_kp_corrected'])
        ax.plot(x, kc[order], 'o-', color=SERIES[n], ms=4, label=n)
    ax.set_xticks(x); ax.set_xticklabels([NODES[i] for i in order], rotation=70, fontsize=7)
    ax.set_ylabel('median error (mm)')
    ax.set_title('Per keypoint (sorted by uncorrected error)', fontsize=10, color=INK)
    ax.legend(fontsize=8, ncol=4); tidy(ax)
    fig.suptitle('Tier 1 — does the corrector reduce keypoint error?', fontsize=11.5, color=INK)
    fig.savefig(f'{FIG}/V1_tier1_keypoint_error.png', bbox_inches='tight'); plt.close(fig)


def fig_tier2():
    """The headline: spatial reward bias, per source x template option."""
    f = f'{RES}/tier2_summary.csv'
    if not os.path.isfile(f):
        return
    d = pd.read_csv(f)
    opts = [o for o in ['existing', 'rebuilt'] if o in set(d.option)]
    metrics = [('sector_spread', 'sector spread in P(match)'),
               ('mu_range', 'μ(θ) range (PC0 units)'),
               ('cond_var', 'behaviour-conditioned sector variance')]
    fig, axes = plt.subplots(len(metrics), len(opts),
                             figsize=(4.6 * len(opts), 3.1 * len(metrics)),
                             squeeze=False, constrained_layout=True)
    order = [s for s in ['deployed', 'uncorrected', 'temporal', 'velacc', 'aspect']
             if s in set(d.source)]
    for r, (col, lab) in enumerate(metrics):
        for c, opt in enumerate(opts):
            ax = axes[r][c]
            g = d[d.option == opt].set_index('source').reindex(order)
            v = g[col].values
            ax.bar(range(len(order)), v, color=[SERIES[s] for s in order],
                   edgecolor=SURFACE, lw=1.2)
            if np.isfinite(v).any():
                for i, val in enumerate(v):
                    if np.isfinite(val):
                        ax.annotate(f'{val:.3g}', (i, val), textcoords='offset points',
                                    xytext=(0, 3), ha='center', fontsize=8, color=INK)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order if r == len(metrics) - 1 else [], rotation=20, fontsize=8)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8.5)
            if r == 0:
                ax.set_title(f'{opt} template', fontsize=10, color=INK)
            tidy(ax)
    fig.suptitle('Tier 2 — does correction reduce the SPATIAL reward bias?\n'
                 'lower is better in every panel', fontsize=11.5, color=INK)
    fig.savefig(f'{FIG}/V2_tier2_spatial.png', bbox_inches='tight'); plt.close(fig)


def fig_sampling():
    """What the balanced training set looks like vs natural occupancy."""
    s = pd.read_csv(f'{RES}/segments.csv')
    rears = pd.read_csv('results/spatial_confound/rears.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    ax = axes[0]
    nat = rears.sector_30.value_counts(normalize=True).sort_index()
    bal = s.sector.value_counts(normalize=True).sort_index()
    x = np.arange(12); w = 0.4
    ax.bar(x - w/2, [nat.get(i, 0) for i in x], w, color=MUTED, label='natural (all rears)',
           edgecolor=SURFACE, lw=1)
    ax.bar(x + w/2, [bal.get(i, 0) for i in x], w, color='#2a78d6', label='training sample',
           edgecolor=SURFACE, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f'{i*30-165:+.0f}' for i in x], rotation=60, fontsize=7)
    ax.set_xlabel('arena angle θ (deg)'); ax.set_ylabel('fraction')
    ax.set_title('Sector balance', fontsize=10, color=INK); ax.legend(fontsize=8); tidy(ax)

    ax = axes[1]
    pc = s.posture.value_counts(normalize=True)
    ax.bar(range(len(pc)), pc.values, color='#2a78d6', edgecolor=SURFACE, lw=1)
    ax.set_xticks(range(len(pc))); ax.set_xticklabels(pc.index, rotation=15)
    ax.set_ylabel('fraction'); ax.set_title('Posture enrichment', fontsize=10, color=INK); tidy(ax)

    ax = axes[2]
    tt = s.groupby(['rat', 'third']).size().unstack(fill_value=0)
    bot = np.zeros(len(tt))
    for j, c in enumerate(tt.columns):
        ax.bar(range(len(tt)), tt[c].values, bottom=bot,
               color=['#2a78d6', '#eb6834', '#1baf7a'][j % 3],
               label=f'third {c}', edgecolor=SURFACE, lw=1)
        bot += tt[c].values
    ax.set_xticks(range(len(tt))); ax.set_xticklabels(tt.index)
    ax.set_ylabel('segments'); ax.set_title('Rat × training-time third', fontsize=10, color=INK)
    ax.legend(fontsize=8); tidy(ax)
    fig.suptitle('Balanced sampling: the training set is flattened relative to natural occupancy',
                 fontsize=11.5, color=INK)
    fig.savefig(f'{FIG}/V3_sampling.png', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    for fn in [fig_sampling, fig_tier1, fig_tier2]:
        try:
            fn(); print('ok', fn.__name__)
        except Exception as e:
            print('skip', fn.__name__, repr(e)[:120])
    print('figures in', FIG)
