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
RATS = ['R4', 'R5', 'R6']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 2.0, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax):
    ax.grid(axis='y', alpha=0.9, zorder=0); ax.set_axisbelow(True)


d = pd.read_csv(f'{RES}/sector_bias_feature_comparison.csv')

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, constrained_layout=True)
for ax, rat in zip(axes, RATS):
    for feat, col, ls in [('full23', SERIES[rat], '-'), ('axial', MUTED, '--')]:
        g = d[(d.rat == rat) & (d.feature == feat)].sort_values('theta')
        ax.plot(g.theta, g.mu, marker='o', ms=5, lw=2, color=col, ls=ls,
               label='full 23 kp (baseline)' if feat == 'full23' else 'axial 9 kp only')
    ax.axhline(0, color=GRID, lw=1)
    ax.set_title(rat, fontsize=10.5, color=INK)
    ax.set_xticks([-165, -75, 0, 75, 165]); ax.set_xlabel('arena angle θ (deg)')
    tidy(ax)
axes[0].set_ylabel('SLEAP − DANNCE (PC0-like units)')
axes[0].legend(fontsize=8.5, loc='upper left')
fig.suptitle('Dropping the four limbs shrinks the sector-dependent bias by ~35%\n'
            '(axial-only PC0 vs the real production PC0, same units, both re-derived from original SLEAP 2D)',
            fontsize=10.5, color=INK, linespacing=1.35)
fig.savefig(f'{FIG}/F14_axial_sector_bias.png', bbox_inches='tight')
plt.close(fig)
print('wrote F14_axial_sector_bias.png')
