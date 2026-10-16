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
AXIAL = {'Snout', 'EarL', 'EarR', 'SpineF', 'SpineM', 'SpineL', 'TailBase', 'HipL', 'HipR'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax):
    ax.grid(axis='y', alpha=0.9, zorder=0); ax.set_axisbelow(True)


d = pd.read_csv(f'{RES}/stage19_keypoint_mse.csv')
d = d[~d.is_egocentric_origin]  # drop SpineM (egocentric origin, RMSE=0 by construction)

pooled = d[d.rat == 'pooled'].sort_values('rmse_mm')
order = pooled.keypoint.tolist()

fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
y = np.arange(len(order))
bar_colors = ['#c9922e' if kp in AXIAL else '#8a8880' for kp in order]
ax.barh(y, pooled.set_index('keypoint').loc[order, 'rmse_mm'], color=bar_colors, height=0.65)
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('RMSE, SLEAP vs. DANNCE (egocentric mm, pooled all 3 rats, 6.45M frames)')
ax.set_title('Per-keypoint SLEAP-DANNCE tracking disagreement, ranked (lower = better agreement)',
             fontsize=11, color=INK)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#c9922e', label='in the "axial" 9kp mitigation set'),
                   Patch(color='#8a8880', label='limb keypoint (dropped by axial set)')],
          fontsize=8.5, loc='lower right')
tidy(ax)
fig.savefig(f'{FIG}/F20_keypoint_mse_ranking.png', bbox_inches='tight')
plt.close(fig)
print('wrote F20 (pooled ranking)')

fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=True, constrained_layout=True)
for ax, rat in zip(axes, ['R4', 'R5', 'R6']):
    g = d[d.rat == rat].set_index('keypoint').loc[order, 'rmse_mm']
    colors = ['#c9922e' if kp in AXIAL else SERIES[rat] for kp in order]
    ax.barh(y, g, color=colors, height=0.65)
    ax.set_title(rat, fontsize=10.5, color=INK)
    ax.set_xlabel('RMSE (mm)')
    tidy(ax)
axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=9)
axes[0].invert_yaxis()
fig.suptitle('Per-keypoint SLEAP-DANNCE RMSE by rat (pooled order preserved; gold = axial 9kp set)',
            fontsize=11.5, color=INK)
fig.savefig(f'{FIG}/F21_keypoint_mse_by_rat.png', bbox_inches='tight')
plt.close(fig)
print('wrote F21 (per-rat)')
