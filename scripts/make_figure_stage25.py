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


corr = pd.read_csv(f'{RES}/stage25_pc_correlations.csv')
loadings = pd.read_csv(f'{RES}/stage25_pc_loadings.csv')

ANGLE_ORDER = ['angle_mid_xy', 'angle_torsion_ear_xy', 'angle_torsion_xy', 'angle_lower_xy',
              'angle_mid_head_xy', 'angle_skip_xy', 'angle_body_head_xy', 'angle_upper_xy']
ANGLE_SHORT = {'angle_lower_xy': 'lower', 'angle_upper_xy': 'upper', 'angle_mid_xy': 'mid',
              'angle_skip_xy': 'skip', 'angle_torsion_xy': 'torsion',
              'angle_body_head_xy': 'body-\nhead', 'angle_mid_head_xy': 'mid-\nhead',
              'angle_torsion_ear_xy': 'torsion-\near'}

# ---- F36: correlation of every angle feature with PC2/PC3
fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
x = np.arange(len(ANGLE_ORDER)); w = 0.26
med = corr.groupby('feature').agg(r_pc2=('r_pc2', lambda v: np.median(np.abs(v))),
                                  r_pc3=('r_pc3', lambda v: np.median(np.abs(v))),
                                  r2=('r2_pc2_pc3', 'median')).loc[ANGLE_ORDER]
ax.bar(x - w, med.r_pc2, w, color='#2a78d6', label='|r| with PC2')
ax.bar(x, med.r_pc3, w, color='#c9922e', label='|r| with PC3')
ax.bar(x + w, med.r2, w, color='#c4423f', label='joint R² (PC2 + PC3)')
ax.set_xticks(x); ax.set_xticklabels([ANGLE_SHORT[f] for f in ANGLE_ORDER], fontsize=8.5)
ax.set_ylim(0, 0.55)
ax.set_ylabel('median across sessions, SLEAP+DANNCE pooled')
ax.set_title('None of the hand-designed turn angles are a clean proxy for PC2/PC3\n'
            '(best joint R² is only ~0.25 — a real but partial overlap)',
            fontsize=10.5, color=INK, linespacing=1.4)
ax.legend(fontsize=8, loc='upper right')
tidy(ax)
fig.savefig(f'{FIG}/F36_angle_pc23_correlation.png', bbox_inches='tight')
plt.close(fig)
print('wrote F36')

# ---- F37: PC2/PC3 loadings by keypoint, one column per rat
RATS = ['R4', 'R5', 'R6']
rep_template = {r: sorted(f for f in loadings.template_file.unique() if f.startswith(r))[0] for r in RATS}
fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True, sharex='row')
for col, rat in enumerate(RATS):
    tmpl = rep_template[rat]
    for row, pc in enumerate(['PC2', 'PC3']):
        ax = axes[row, col]
        g = loadings[(loadings.template_file == tmpl) & (loadings.pc == pc)].sort_values('magnitude', ascending=True).tail(10)
        y = np.arange(len(g))
        ax.barh(y, g.magnitude, color=SERIES[rat], height=0.65)
        ax.set_yticks(y); ax.set_yticklabels(g.keypoint, fontsize=8)
        ax.set_title(f'{rat} — {pc}' + (f'\n({tmpl})' if row == 0 else ''), fontsize=9.5, color=INK)
        if row == 1:
            ax.set_xlabel('loading magnitude (||x,y,z||)')
        tidy(ax, axis='x')
fig.suptitle('PC2 (top row) and PC3 (bottom row) loadings: top 10 keypoints by magnitude\n'
            'PC2 ≈ one-sided limb + tailbase (asymmetric limb/tail placement); '
            'PC3 ≈ snout pitch + forelimbs (consistent across all 3 rats)',
            fontsize=11.5, color=INK, linespacing=1.4)
fig.savefig(f'{FIG}/F37_pc23_loadings_by_rat.png', bbox_inches='tight')
plt.close(fig)
print('wrote F37')
