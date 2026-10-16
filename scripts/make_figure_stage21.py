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
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax):
    ax.grid(axis='y', alpha=0.9, zorder=0); ax.set_axisbelow(True)


def grouped_bars(ax, feats, labels, title, ylabel, value_col='rmse'):
    x = np.arange(len(feats)); w = 0.24
    for i, rat in enumerate(RATS):
        g = d[(d.rat == rat) & (d.feature.isin(feats))].set_index('feature').loc[feats, value_col]
        ax.bar(x + (i - 1) * w, g, w, color=SERIES[rat], label=rat, edgecolor=SURFACE, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_ylabel(ylabel)
    tidy(ax)


d = pd.read_csv(f'{RES}/stage21_feature_reliability.csv')

fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), constrained_layout=True)

grouped_bars(axes[0, 0], ['snout_z', 'ear_snout_mean_z', 'ear_mean_z'],
            ['snout Z', 'mean(snout,\nearL,earR) Z', 'mean(earL,\nearR) Z'],
            'Height features', 'RMSE (mm)')
axes[0, 0].legend(fontsize=7.5, loc='upper left')

DIST_FEATS = ['dist_snout_tb', 'dist_ears_tb', 'dist_spineF_tb', 'dist_earmean_hipmean']
DIST_LABELS = ['snout ↔\ntailbase', 'ears ↔\ntailbase', 'spineF ↔\ntailbase', 'ears ↔\nhips']
grouped_bars(axes[0, 1], DIST_FEATS, DIST_LABELS,
            'Body-length / stretch distance features', 'RMSE (mm)')

grouped_bars(axes[0, 2], ['com_velocity'], ['XYZ center-of-mass\nvelocity'],
            'Velocity feature', 'RMSE (mm/s)')

ANGLE_LABELS = ['lower', 'upper', 'mid', 'skip', 'torsion', 'body-\nhead', 'mid-\nhead',
               'torsion-\near']
ANGLE_FEATS = ['angle_lower_xy', 'angle_upper_xy', 'angle_mid_xy', 'angle_skip_xy',
              'angle_torsion_xy', 'angle_body_head_xy', 'angle_mid_head_xy',
              'angle_torsion_ear_xy']
grouped_bars(axes[1, 0], ANGLE_FEATS, ANGLE_LABELS,
            'Angle features — magnitude error', 'RMSE (deg)')
axes[1, 0].tick_params(axis='x', labelsize=7)

grouped_bars(axes[1, 1], ANGLE_FEATS, ANGLE_LABELS,
            'Angle features — directional (sign) agreement', '% frames same sign', value_col='sign_agree_pct')
axes[1, 1].tick_params(axis='x', labelsize=7)
axes[1, 1].set_ylim(0, 100)
axes[1, 1].axhline(50, color=MUTED, lw=1, ls=':')
axes[1, 1].annotate('chance', (4.7, 52), fontsize=7.5, color=MUTED)

axes[1, 2].axis('off')
axes[1, 2].text(0.02, 0.98,
    "ear_mean_z (no snout) is the single\n"
    "best height feature (5.4mm); ears↔hips\n"
    "is the single best stretch feature\n"
    "(8.2mm) — both avoid snout AND\n"
    "tailbase, the two worst-tracked\n"
    "keypoints (§8p–§8r).\n\n"
    "torsion (hip axis vs shoulder axis)\n"
    "has the best sign-agreement of any\n"
    "angle (81.9%) while avoiding snout,\n"
    "tailbase, and short spine segments —\n"
    "the best-motivated turn candidate.\n\n"
    "body-head has the BEST sign-agreement\n"
    "overall (85.6%) but the WORST RMSE\n"
    "(23.0°) — direction is more stable\n"
    "than magnitude when snout is one\n"
    "endpoint of a long lever arm.\n\n"
    "mid-head (spineL→spineM→ears,\n"
    "skipping spineF) is unremarkable\n"
    "(17.0°/77.8%) — mid-of-the-pack.\n"
    "torsion-ear (hip axis vs EAR axis,\n"
    "not shoulder) is the WORST angle\n"
    "by RMSE (24.0°) — ears are close\n"
    "together and move independently\n"
    "of the body, a short + noisy lever.",
    transform=axes[1, 2].transAxes, fontsize=8.3, color=INK2, va='top', linespacing=1.45)

fig.suptitle('Stage 21 — reliability of 16 candidate online-matching features (world space, 6.45M frames)',
            fontsize=12, color=INK)
fig.savefig(f'{FIG}/F27_feature_reliability.png', bbox_inches='tight')
plt.close(fig)
print('wrote F27')
