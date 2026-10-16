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

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax, axis='y'):
    ax.grid(axis=axis, alpha=0.9, zorder=0); ax.set_axisbelow(True)


gt = pd.read_csv(f'{RES}/stage23_gt_sizes.csv')
s23 = pd.read_csv(f'{RES}/stage23_combo_summary.csv')
s22 = pd.read_csv(f'{RES}/stage22_combo_summary.csv')

# ---- F31: ground-truth size explosion for single features
singles = ['ear_mean_z', 'ear_snout_mean_z', 'snout_z', 'dist_earmean_hipmean', 'dist_spineF_tb',
          'dist_ears_tb', 'dist_snout_tb', 'speed', 'angle_skip_xy', 'angle_lower_xy',
          'angle_upper_xy', 'angle_torsion_xy', 'angle_body_head_xy', 'angle_mid_xy']
med = gt[gt.combo.isin(singles)].groupby('combo').n_gt.median().loc[singles]
colors = ['#2a78d6' if 'z' in c and 'angle' not in c else
         ('#c9922e' if c.startswith('dist') or c == 'speed' else '#c4423f') for c in singles]

fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
y = np.arange(len(singles))
ax.barh(y, med.values, color=colors, height=0.65)
ax.set_yticks(y); ax.set_yticklabels(singles, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('median feature-specific ground-truth events per session')
ax.axvline(gt[gt.combo == 'ear_mean_z'].n_gt.median(), color=MUTED, lw=1, ls=':')
ax.annotate('~seed candidate pool size\n(height features closely\nreproduce it)',
           (gt[gt.combo=='ear_mean_z'].n_gt.median() + 15, 8.3), fontsize=7.8, color=MUTED)
ax.set_title('Feature-specific ground truth balloons for angle features\n'
            '(a loose, high-variance corridor matches far more than the original candidates)',
            fontsize=10.5, color=INK, linespacing=1.4)
tidy(ax, axis='x')
fig.savefig(f'{FIG}/F31_ground_truth_size_by_feature.png', bbox_inches='tight')
plt.close(fig)
print('wrote F31')

# ---- F32: stage22 (shared rearing GT) vs stage23 (feature-specific GT) -- same conclusion?
compare = ['ear_mean_z', 'ear_mean_z+dist_earmean_hipmean', 'ear_mean_z+angle_skip_xy',
          'ear_mean_z+angle_torsion_xy', 'ear_mean_z+angle_body_head_xy',
          'angle_torsion_xy', 'angle_body_head_xy']
p22 = s22[(s22.rat == 'pooled') & (s22.combo.isin(compare))].set_index('combo').loc[compare]
p23 = s23[(s23.rat == 'pooled') & (s23.combo.isin(compare))].set_index('combo').loc[compare]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
x = np.arange(len(compare)); w = 0.32
for ax, metric, title in [(axes[0], 'precision', 'Precision'), (axes[1], 'recall', 'Recall')]:
    ax.bar(x - w / 2, p22[metric], w, color='#8a8880', label='stage22: shared rearing GT')
    ax.bar(x + w / 2, p23[metric], w, color='#2a78d6', label='stage23: feature-specific GT')
    ax.axhline(0.8, color='#1baf7a', lw=1.2, ls='--', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(compare, fontsize=7.5, rotation=28, ha='right')
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=10.5, color=INK)
    tidy(ax)
axes[0].legend(fontsize=8, loc='upper right')
fig.suptitle('Same conclusion under both ground-truth methodologies:\n'
            'angle_torsion_xy/angle_body_head_xy fail either way; angle_skip_xy survives either way',
            fontsize=11, color=INK, linespacing=1.4)
fig.savefig(f'{FIG}/F32_stage22_vs_stage23_comparison.png', bbox_inches='tight')
plt.close(fig)
print('wrote F32')
