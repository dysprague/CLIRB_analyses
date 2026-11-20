"""Figure for the three-way feature comparison: equal-weight SLEAP, conf^2-weighted
SLEAP and DANNCE, for ear_mean_z and cum_heading_change.

Reads results/newcal/feature_three_way.csv and feature_three_way_pooled.csv
(written by scripts/newcal_feature_three_way.py).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

OUT = 'results/newcal'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)

C_EQ, C_WT, C_BOTH = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
PAIRC = {'eq_vs_dn': C_EQ, 'wt_vs_dn': C_WT, 'eq_vs_wt': C_BOTH}
PAIRL = {'eq_vs_dn': 'equal-weight SLEAP vs DANNCE',
         'wt_vs_dn': 'conf² SLEAP vs DANNCE',
         'eq_vs_wt': 'equal vs conf² (SLEAP internal)'}
PAIRS = ['eq_vs_dn', 'wt_vs_dn', 'eq_vs_wt']
STRATA = ['near port (<150mm)', 'mid (150-300mm)', 'far (>300mm)']
STRATL = ['near port\n<150 mm', 'mid\n150–300 mm', 'far\n>300 mm']

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK, 'text.color': INK,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'grid.color': '#e8e7e2', 'grid.linewidth': 0.7,
    'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def tidy(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True)


def per_session(ax, d, feat, col, ylab, title, log=False):
    s = d[(d.feature == feat) & (d.stratum == 'all')]
    keys = sorted({(r, ss) for r, ss in zip(s.rat, s.session)})
    x = np.arange(len(keys))
    for p in PAIRS:
        sub = s[s.pair == p].set_index(['rat', 'session'])
        y = [sub.loc[k, col] for k in keys]
        ax.plot(x, y, color=PAIRC[p], lw=1.1, marker='o', ms=4.5, alpha=0.9,
                mec='white', mew=0.6, zorder=3, label=PAIRL[p])
    # rat boundaries
    rats = [k[0] for k in keys]
    for i in range(1, len(keys)):
        if rats[i] != rats[i - 1]:
            ax.axvline(i - 0.5, color=MUTED, lw=0.9, ls='--', zorder=1)
    for r in sorted(set(rats)):
        idx = [i for i, k in enumerate(keys) if k[0] == r]
        ax.annotate(r, xy=(np.mean(idx), 1.005), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=8, color=INK2, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([k[1][5:10].replace('_', '-') for k in keys], fontsize=7, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(keys) - 0.5)
    if log: ax.set_yscale('log')
    ax.set_ylabel(ylab); ax.set_title(title, loc='left', pad=20); tidy(ax)


def strat_bars(ax, p, feat, col, ylab, title, ylim=None, pct=False):
    s = p[(p.feature == feat)]
    x = np.arange(len(STRATA)); w = 0.26
    for j, pr in enumerate(PAIRS):
        y = [s[(s.stratum == st) & (s.pair == pr)][col].iloc[0] for st in STRATA]
        ax.bar(x + (j - 1) * w, y, w, color=PAIRC[pr], edgecolor='white',
               linewidth=0.7, zorder=3, label=PAIRL[pr])
        for xi, yi in zip(x + (j - 1) * w, y):
            ax.annotate(f'{yi:.2f}' if not pct else f'{yi:.3f}', xy=(xi, yi), ha='center',
                        va='bottom', fontsize=6.4, color=PAIRC[pr], rotation=90,
                        xytext=(0, 2.5), textcoords='offset points')
    tot = s[(s.stratum == 'all') & (s.pair == 'eq_vs_dn')]['n_frames'].iloc[0]
    labs = []
    for st, base in zip(STRATA, STRATL):
        sub = s[(s.stratum == st) & (s.pair == 'eq_vs_dn')]
        extra = f'{100 * sub["n_frames"].iloc[0] / tot:.0f}% of frames'
        if feat == 'cum_heading_change':
            extra += f'\nturning {100 * sub["frac_real_turn"].iloc[0]:.0f}% of it'
        labs.append(f'{base}\n{extra}')
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.4)
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(ylab); ax.set_title(title, loc='left', pad=8); tidy(ax)


def main():
    d = pd.read_csv(f'{OUT}/feature_three_way.csv')
    p = pd.read_csv(f'{OUT}/feature_three_way_pooled.csv')

    fig = plt.figure(figsize=(14.6, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.05], hspace=0.62, wspace=0.27)

    ax = fig.add_subplot(gs[0, 0])
    per_session(ax, d, 'ear_mean_z', 'rmse', 'RMSE (mm)',
                'A · ear_mean_z — per-session RMSE')
    ax.legend(loc='upper left', fontsize=6.8, ncol=1)

    ax = fig.add_subplot(gs[0, 1])
    per_session(ax, d, 'ear_mean_z', 'pearson_r', 'Pearson r',
                'B · ear_mean_z — correlation')
    ax.set_ylim(0.93, 1.001)

    ax = fig.add_subplot(gs[0, 2])
    strat_bars(ax, p, 'ear_mean_z', 'rmse', 'RMSE (mm)',
               'C · ear_mean_z by distance to water port\n(pooled, 576k frames)', ylim=(0, 8.6))

    ax = fig.add_subplot(gs[1, 0])
    per_session(ax, d, 'cum_heading_change', 'rmse', 'RMSE (deg)',
                'D · cum_heading_change — per-session RMSE')

    ax = fig.add_subplot(gs[1, 1])
    per_session(ax, d, 'cum_heading_change', 'pearson_r', 'Pearson r',
                'E · cum_heading_change — correlation')
    ax.set_ylim(0.3, 1.02)

    ax = fig.add_subplot(gs[1, 2])
    strat_bars(ax, p, 'cum_heading_change', 'pearson_r', 'Pearson r',
               'F · cum_heading_change by distance to port\n(pooled)', ylim=(0, 1.22), pct=True)
    fig.suptitle('Confidence-weighted triangulation vs equal weighting, measured on the two '
                 'interpretable features (13 sessions, R7/R8/R9, Oct 22–28)',
                 x=0.012, ha='left', fontsize=12.5, y=1.0)
    fig.text(0.012, -0.03,
             'All three sources pass through the online pipeline: DLT → 5-frame causal median → '
             'Procrustes-aligned DANNCE. conf² weighting lowers error against DANNCE in almost every '
             'session and stratum, but the\ndominant effect is position: at the water port, '
             'ear height error is 2× and accumulated-turn correlation falls from 0.94 to 0.60. '
             'The two SLEAP variants agree with each other far more closely\nthan either agrees '
             'with DANNCE, so the residual is DANNCE-vs-SLEAP disagreement, not triangulation noise.',
             fontsize=8, color=INK2, ha='left')
    fig.savefig(f'{FIG}/F20_three_way_features.png', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG}/F20_three_way_features.png')


if __name__ == '__main__':
    main()
