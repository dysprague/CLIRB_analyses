import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

RES = 'results/spatial_confound'
FIG = 'results/figures/spatial_confound'
os.makedirs(FIG, exist_ok=True)

SURFACE = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#8a8880'; GRID = '#e4e3df'
SERIES = {'R4': '#2a78d6', 'R5': '#eb6834', 'R6': '#1baf7a'}
FEAT_COL = {'full23_fresh': MUTED, 'axial': '#2a78d6', 'snout_z': '#eb6834', 'bodyrot': '#1baf7a'}
FEAT_LABEL = {'full23_fresh': 'full 23 kp (fresh-fit baseline)', 'axial': 'axial 9 kp',
             'snout_z': 'snout height', 'bodyrot': 'body rotation angle'}
RATS = ['R4', 'R5', 'R6']
FEATS = ['full23_fresh', 'axial', 'snout_z', 'bodyrot']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 2.0, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax, ygrid=True):
    ax.grid(axis='y' if ygrid else 'x', alpha=0.9, zorder=0); ax.set_axisbelow(True)


d = pd.read_csv(f'{RES}/p_rewarded_by_feature.csv')

# ============================================================= FIG 15
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True, constrained_layout=True)
for ax, rat in zip(axes, RATS):
    g = d[d.rat == rat]
    for feat in FEATS:
        by_sec = g.groupby('sector')[feat].agg(['mean', 'size']).reset_index()
        by_sec = by_sec[by_sec['size'] >= 5]
        ax.plot(by_sec.sector * 30 - 165, by_sec['mean'], 'o-', ms=5, lw=2,
               color=FEAT_COL[feat], label=FEAT_LABEL[feat])
    ax.set_title(rat, fontsize=10.5, color=INK)
    ax.set_xticks([-165, -75, 0, 75, 165]); ax.set_xlabel('arena angle θ (deg)')
    ax.set_ylim(0.4, 1.05)
    tidy(ax)
axes[0].set_ylabel('P(rewarded | DANNCE-detected rear)')
axes[0].legend(fontsize=7.8, loc='lower left')
fig.suptitle('P(rewarded) by sector: body rotation angle is flattest and highest\n'
            '(all four freshly fit the same way, on the same raw-SLEAP data — a fair comparison)',
            fontsize=10.5, color=INK, linespacing=1.35)
fig.savefig(f'{FIG}/F15_p_rewarded_by_sector.png', bbox_inches='tight')
plt.close(fig)

# ============================================================= FIG 16
rows = []
for rat, g in d.groupby('rat'):
    for feat in FEATS:
        by_sec = g.groupby('sector')[feat].mean()
        rows.append(dict(rat=rat, feature=feat, P=g[feat].mean(),
                         spread=by_sec.max() - by_sec.min()))
s = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), constrained_layout=True)
ax = axes[0]
x = np.arange(len(FEATS)); w = 0.25
for i, rat in enumerate(RATS):
    g = s[s.rat == rat].set_index('feature').loc[FEATS]
    ax.bar(x + (i - 1) * w, g.spread, w, color=SERIES[rat], label=rat, edgecolor=SURFACE, lw=1.0)
ax.set_xticks(x); ax.set_xticklabels([FEAT_LABEL[f].replace(' (fresh-fit baseline)', '\n(baseline)')
                                      for f in FEATS], fontsize=8)
ax.set_ylabel('sector spread of P(rewarded)\n(max − min across 12 sectors)')
ax.set_title('Lower is less sector-biased', fontsize=10, color=INK)
ax.legend(fontsize=8.5); tidy(ax)

ax = axes[1]
for i, rat in enumerate(RATS):
    g = s[s.rat == rat].set_index('feature').loc[FEATS]
    ax.bar(x + (i - 1) * w, g.P, w, color=SERIES[rat], label=rat, edgecolor=SURFACE, lw=1.0)
ax.set_xticks(x); ax.set_xticklabels([FEAT_LABEL[f].replace(' (fresh-fit baseline)', '\n(baseline)')
                                      for f in FEATS], fontsize=8)
ax.set_ylabel('pooled P(rewarded)')
ax.set_ylim(0.6, 1.03)
ax.set_title('Higher is better overall recall', fontsize=10, color=INK)
tidy(ax)
fig.suptitle('Body rotation angle: lowest sector-spread AND highest match rate, in all 3 rats',
            fontsize=11, color=INK)
fig.savefig(f'{FIG}/F16_p_rewarded_spread_summary.png', bbox_inches='tight')
plt.close(fig)
print('wrote F15, F16')
