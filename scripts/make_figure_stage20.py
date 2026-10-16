import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


def tidy(ax, axis='y'):
    ax.grid(axis=axis, alpha=0.9, zorder=0); ax.set_axisbelow(True)


world = pd.read_csv(f'{RES}/stage20_keypoint_mse_world.csv')
ego = pd.read_csv(f'{RES}/stage19_keypoint_mse.csv')
ego = ego[~ego.is_egocentric_origin]  # SpineM excluded there (RMSE=0 by construction)

w_pooled = world[world.rat == 'pooled'].sort_values('rmse_mm')
order = w_pooled.keypoint.tolist()

# ---- F22: pooled world-space ranking (mirrors F20's style)
fig, ax = plt.subplots(figsize=(9.5, 6.8), constrained_layout=True)
y = np.arange(len(order))
colors = ['#c9922e' if kp in AXIAL else '#8a8880' for kp in order]
ax.barh(y, w_pooled.set_index('keypoint').loc[order, 'rmse_mm'], color=colors, height=0.65)
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('RMSE, SLEAP vs. DANNCE (raw 3D world space, Procrustes-aligned per session, mm)')
ax.set_title('Per-keypoint SLEAP-DANNCE disagreement in WORLD space (no egocentric/SpineM confound)',
             fontsize=10.5, color=INK)
ax.legend(handles=[Patch(color='#c9922e', label='in the "axial" 9kp mitigation set'),
                   Patch(color='#8a8880', label='limb keypoint (dropped by axial set)')],
          fontsize=8.5, loc='lower right')
tidy(ax)
fig.savefig(f'{FIG}/F22_keypoint_mse_world_ranking.png', bbox_inches='tight')
plt.close(fig)
print('wrote F22')

# ---- F23: per-rat, world space
fig, axes = plt.subplots(1, 3, figsize=(13, 6.8), sharey=True, constrained_layout=True)
for ax, rat in zip(axes, ['R4', 'R5', 'R6']):
    g = world[world.rat == rat].set_index('keypoint').loc[order, 'rmse_mm']
    colors = ['#c9922e' if kp in AXIAL else SERIES[rat] for kp in order]
    ax.barh(y, g, color=colors, height=0.65)
    ax.set_title(rat, fontsize=10.5, color=INK)
    ax.set_xlabel('RMSE (mm)')
    tidy(ax)
axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=9)
axes[0].invert_yaxis()
fig.suptitle('Per-keypoint SLEAP-DANNCE RMSE by rat, world space (pooled order preserved; gold = axial 9kp)',
            fontsize=11.5, color=INK)
fig.savefig(f'{FIG}/F23_keypoint_mse_world_by_rat.png', bbox_inches='tight')
plt.close(fig)
print('wrote F23')

# ---- F24: egocentric vs. world-space rank, side by side (bump chart)
e_pooled = ego[ego.rat == 'pooled'].sort_values('rmse_mm').reset_index(drop=True)
e_rank = {kp: i + 1 for i, kp in enumerate(e_pooled.keypoint)}
w_rank_full = {kp: i + 1 for i, kp in enumerate(order)}
# SpineM has no egocentric rank (it's the egocentric origin, excluded from stage19
# entirely) -- drop it from this comparison, not from the standalone world-space figures.
kps = [kp for kp in order if kp in e_rank]
w_rank = {kp: i + 1 for i, kp in enumerate(kps)}

fig, ax = plt.subplots(figsize=(7.5, 8), constrained_layout=True)
for kp in kps:
    er, wr = e_rank[kp], w_rank[kp]
    moved = abs(er - wr)
    color = '#c9922e' if kp in AXIAL else '#8a8880'
    lw = 1.0 + 0.35 * moved
    alpha = 0.55 if moved < 4 else 0.95
    ax.plot([0, 1], [er, wr], '-', color=color, lw=lw, alpha=alpha, zorder=2 if moved >= 4 else 1)
    ax.scatter([0, 1], [er, wr], color=color, s=22, zorder=3)
for kp in kps:
    ax.annotate(kp, (0, e_rank[kp]), ha='right', va='center', fontsize=8, color=INK,
               xytext=(-6, 0), textcoords='offset points')
    ax.annotate(kp, (1, w_rank[kp]), ha='left', va='center', fontsize=8, color=INK,
               xytext=(6, 0), textcoords='offset points')
ax.set_xlim(-0.55, 1.55); ax.set_xticks([0, 1])
ax.set_xticklabels(['egocentric\n(SpineM-anchored)\nrank', 'world space\n(Procrustes-aligned)\nrank'],
                   fontsize=9.5)
ax.set_ylim(len(kps) + 1.5, -0.5)
ax.set_yticks([])
ax.set_title('Does the egocentric/SpineM confound change the ranking?\n(thick line = rank changed by ≥4; gold = axial 9kp set)',
             fontsize=10.5, color=INK, linespacing=1.4)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(f'{FIG}/F24_ranking_egocentric_vs_world.png', bbox_inches='tight')
plt.close(fig)
print('wrote F24')

print('\nBiggest rank changes (egocentric -> world):')
diffs = sorted(kps, key=lambda k: -abs(e_rank[k] - w_rank[k]))
for kp in diffs[:8]:
    print(f'  {kp:10s}  ego rank {e_rank[kp]:2d}  ->  world rank {w_rank[kp]:2d}  '
         f'(ego RMSE {e_pooled.set_index("keypoint").loc[kp,"rmse_mm"]:.1f}mm, '
         f'world RMSE {w_pooled.set_index("keypoint").loc[kp,"rmse_mm"]:.1f}mm)')
