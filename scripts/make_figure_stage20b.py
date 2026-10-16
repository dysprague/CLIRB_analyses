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
AXIAL = {'Snout', 'EarL', 'EarR', 'SpineF', 'SpineM', 'SpineL', 'TailBase', 'HipL', 'HipR'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax):
    ax.grid(axis='x', alpha=0.9, zorder=0); ax.set_axisbelow(True)


d = pd.read_csv(f'{RES}/stage20b_stratified_ranking.csv')

# ---- F25: corrector ON vs OFF, grouped horizontal bars, ordered by ON rank
on = d[d.stratum == 'corrector=ON'].set_index('keypoint')
off = d[d.stratum == 'corrector=OFF'].set_index('keypoint')
order = on.sort_values('rank').index.tolist()

fig, ax = plt.subplots(figsize=(9.5, 7), constrained_layout=True)
y = np.arange(len(order)); h = 0.36
ax.barh(y + h / 2, on.loc[order, 'rmse_mm'], h, color='#2a78d6', label='corrector ON (old SLEAP model, n=5.16M frames)')
ax.barh(y - h / 2, off.loc[order, 'rmse_mm'], h, color='#eb6834', label='corrector OFF (new SLEAP model, n=1.30M frames)')
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('RMSE, SLEAP vs. DANNCE (world space, mm)')
ax.set_title('Same ranking, lower absolute error: corrector ON vs. OFF\n'
             '(Spearman rank correlation = 0.83 across all 23 keypoints)',
             fontsize=10.5, color=INK, linespacing=1.4)
ax.legend(fontsize=8.5, loc='lower right')
tidy(ax)
fig.savefig(f'{FIG}/F25_keypoint_mse_corrector_onoff.png', bbox_inches='tight')
plt.close(fig)
print('wrote F25')

# ---- F26: 3-stratum progression bump chart -- main(old+ON) -> bridge(old+OFF) -> post-switch(new+OFF)
strata3 = ['main: old_model + corrector_ON', 'bridge: old_model + corrector_OFF',
          'post-switch: new_model + corrector_OFF']
labels3 = ['old model\ncorrector ON\n(n=5.16M frames)', 'old model\ncorrector OFF\n(n=144k frames, bridge)',
          'new model\ncorrector OFF\n(n=1.15M frames)']
ranks = {s: d[d.stratum == s].set_index('keypoint')['rank'].to_dict() for s in strata3}
kps = list(ranks[strata3[0]].keys())

fig, ax = plt.subplots(figsize=(9, 8.5), constrained_layout=True)
for kp in kps:
    ys = [ranks[s][kp] for s in strata3]
    highlight = kp in ('Snout', 'TailBase')
    color = '#c4423f' if kp == 'Snout' else ('#1baf7a' if kp == 'TailBase' else
                                              ('#c9922e' if kp in AXIAL else '#c7c4bc'))
    lw = 2.6 if highlight else 1.0
    alpha = 1.0 if highlight else 0.55
    zorder = 5 if highlight else 2
    ax.plot([0, 1, 2], ys, '-o', color=color, lw=lw, alpha=alpha, ms=4, zorder=zorder)
    if highlight:
        ax.annotate(kp, (2, ys[-1]), xytext=(8, 0), textcoords='offset points',
                   va='center', fontsize=9.5, weight='bold', color=color)
ax.set_xlim(-0.3, 2.6); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels3, fontsize=9)
ax.set_ylim(24, 0); ax.set_ylabel('rank (1 = best SLEAP-DANNCE agreement)')
ax.set_title('Does the ranking survive the pipeline-version natural experiment?\n'
             'Snout (red) is worst-or-near-worst in every version; TailBase (green) never is',
             fontsize=10.5, color=INK, linespacing=1.4)
tidy(ax)
ax.grid(axis='y', alpha=0.9, zorder=0)
fig.savefig(f'{FIG}/F26_ranking_across_pipeline_versions.png', bbox_inches='tight')
plt.close(fig)
print('wrote F26')
