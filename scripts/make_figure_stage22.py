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


def tidy(ax):
    ax.grid(alpha=0.9, zorder=0); ax.set_axisbelow(True)


s = pd.read_csv(f'{RES}/stage22_combo_summary.csv')
pooled = s[s.rat == 'pooled'].copy()

fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)

# target zone
ax.axvspan(0.8, 1.02, ymin=(0.8 - 0.55) / (1.02 - 0.55), ymax=1.0, color='#e7f3e9', zorder=0)
ax.axhline(0.8, color='#1baf7a', lw=1.2, ls='--', alpha=0.7)
ax.axvline(0.8, color='#1baf7a', lw=1.2, ls='--', alpha=0.7)
ax.annotate('80% / 80% target', (0.805, 0.565), fontsize=8.5, color='#1baf7a')

colors = {1: '#2a78d6', 2: '#c9922e', 3: '#8a5fc9'}
labels = {1: 'single features', 2: 'pairs', 3: 'ear_mean_z + turn + stretch triples'}
for nf, g in pooled.groupby('n_features'):
    ax.scatter(g.precision, g.recall, s=55, color=colors[nf], alpha=0.75,
              label=f'{labels[nf]} (n={len(g)})', edgecolor='white', linewidth=0.5, zorder=3)

# label the standout single features and the best turn/stretch additions
highlight = ['ear_mean_z', 'snout_z', 'speed', 'ear_mean_z+dist_earmean_hipmean',
            'ear_mean_z+angle_skip_xy', 'ear_mean_z+angle_torsion_xy',
            'ear_mean_z+angle_skip_xy+dist_earmean_hipmean']
for combo in highlight:
    row = pooled[pooled.combo == combo].iloc[0]
    ax.annotate(combo, (row.precision, row.recall), fontsize=7.5, color=INK,
               xytext=(6, 4), textcoords='offset points')
    ax.scatter([row.precision], [row.recall], s=90, facecolor='none',
              edgecolor=INK, linewidth=1.3, zorder=4)

ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
ax.set_xlabel('precision (pooled, all 3 rats)')
ax.set_ylabel('recall (pooled, all 3 rats)')
ax.set_title('Stage 22 — precision/recall of every combo, incl. turn/stretch additions\n'
            'corrector-OFF, new-SLEAP-model sessions only (33 sessions, ~1.1M frames)',
            fontsize=11, color=INK, linespacing=1.4)
ax.legend(fontsize=8.5, loc='lower right')
tidy(ax)
fig.savefig(f'{FIG}/F28_feature_subset_precision_recall.png', bbox_inches='tight')
plt.close(fig)
print('wrote F28')

# ---- F29: per-rat consistency for the best single feature
fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
best = ['ear_mean_z', 'ear_snout_mean_z', 'snout_z']
x = np.arange(len(best)); w = 0.2
for i, rat in enumerate(['R4', 'R5', 'R6']):
    g = s[(s.rat == rat) & (s.combo.isin(best))].set_index('combo').loc[best]
    ax.bar(x + (i - 1) * w, g.precision, w, color=SERIES[rat], alpha=0.55, label=f'{rat} precision')
for i, rat in enumerate(['R4', 'R5', 'R6']):
    g = s[(s.rat == rat) & (s.combo.isin(best))].set_index('combo').loc[best]
    ax.plot(x + (i - 1) * w, g.recall, 'D', color=SERIES[rat], ms=7, markeredgecolor=INK, markeredgewidth=0.6)
ax.axhline(0.8, color='#1baf7a', lw=1.2, ls='--', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(best, fontsize=9)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel('bars = precision, diamonds = recall')
ax.set_title('Top single-feature candidates hold up in every rat individually',
            fontsize=10.5, color=INK)
ax.legend(fontsize=7.5, loc='lower right')
tidy(ax)
fig.savefig(f'{FIG}/F29_best_features_by_rat.png', bbox_inches='tight')
plt.close(fig)
print('wrote F29')

# ---- F30: per-rat check for the best surviving turn/stretch additions to ear_mean_z
fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
combos30 = ['ear_mean_z', 'ear_mean_z+dist_earmean_hipmean', 'ear_mean_z+angle_skip_xy',
           'ear_mean_z+angle_skip_xy+dist_earmean_hipmean']
labels30 = ['ear_mean_z\n(height alone)', '+ dist_earmean\n_hipmean (stretch)',
           '+ angle_skip_xy\n(turn)', '+ both\n(height+turn+stretch)']
x = np.arange(len(combos30)); w = 0.2
for i, rat in enumerate(['R4', 'R5', 'R6']):
    g = s[(s.rat == rat) & (s.combo.isin(combos30))].set_index('combo').loc[combos30]
    ax.bar(x + (i - 1) * w, g.precision, w, color=SERIES[rat], alpha=0.55, label=f'{rat} precision')
for i, rat in enumerate(['R4', 'R5', 'R6']):
    g = s[(s.rat == rat) & (s.combo.isin(combos30))].set_index('combo').loc[combos30]
    ax.plot(x + (i - 1) * w, g.recall, 'D', color=SERIES[rat], ms=7, markeredgecolor=INK, markeredgewidth=0.6)
ax.axhline(0.8, color='#1baf7a', lw=1.2, ls='--', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels30, fontsize=8.5)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel('bars = precision, diamonds = recall')
ax.set_title('Adding turn/stretch to height: R6 recall is the weak point',
            fontsize=10.5, color=INK)
ax.legend(fontsize=7.5, loc='lower left')
tidy(ax)
fig.savefig(f'{FIG}/F30_turn_stretch_by_rat.png', bbox_inches='tight')
plt.close(fig)
print('wrote F30')
