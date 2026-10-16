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
TEMPLATE_COLOR = {'rearing': '#2a78d6', 'turning': '#c4423f', 'rear_turn': '#1baf7a',
                 'locomotion': '#c9922e'}
TEMPLATE_LABEL = {'rearing': 'rearing', 'turning': 'turning\n(pure)', 'rear_turn': 'rear +\nturn',
                  'locomotion': 'locomotion'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax, axis='y'):
    ax.grid(axis=axis, alpha=0.9, zorder=0); ax.set_axisbelow(True)


d = pd.read_csv(f'{RES}/stage24_results.csv')
s = pd.read_csv(f'{RES}/stage24_summary.csv')
seed = pd.read_csv(f'{RES}/stage24_seed_counts.csv')
TEMPLATES = ['rearing', 'turning', 'rear_turn', 'locomotion']

# ---- F33: ground-truth explosion diagnostic -- seed pool vs n_gt (median across features)
fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
x = np.arange(len(TEMPLATES)); w = 0.35
seed_med = [seed[f'n_seed_{t}'].median() for t in TEMPLATES]
ngt_med = [d[(d.template == t) & (d.n_features == 1)].n_gt.median() for t in TEMPLATES]
ax.bar(x - w / 2, seed_med, w, color='#8a8880', label='seed pool (SLEAP-only, behavior-specific)')
ax.bar(x + w / 2, ngt_med, w, color='#c4423f', label='ground-truth size after rebuild (median over 16 features)')
ax.axhline(1200, color=MUTED, lw=1, ls=':')
ax.annotate('theoretical max (session/refractory = 36000/30)', (-0.4, 1215), fontsize=7.5, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([TEMPLATE_LABEL[t] for t in TEMPLATES])
ax.set_ylim(0, 1330)
ax.set_ylabel('events per session (median)')
ax.set_title('Ground-truth construction only works for postures with a consistent "shape"\n'
            '(turning/locomotion explode toward the refractory ceiling for every feature)',
            fontsize=10.5, color=INK, linespacing=1.4)
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 0.86))
tidy(ax)
fig.savefig(f'{FIG}/F33_ground_truth_explosion_by_template.png', bbox_inches='tight')
plt.close(fig)
print('wrote F33')

# ---- F34: precision/recall across templates for the best candidates
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
combos34 = ['ear_mean_z', 'ear_mean_z+angle_skip_xy']
colors34 = {'ear_mean_z': '#2a78d6', 'ear_mean_z+angle_skip_xy': '#c9922e'}
pooled = s[s.rat == 'pooled']
x = np.arange(len(TEMPLATES)); w = 0.35
for ax, metric, title in [(axes[0], 'precision', 'Precision'), (axes[1], 'recall', 'Recall')]:
    for i, combo in enumerate(combos34):
        vals = [pooled[(pooled.template == t) & (pooled.combo == combo)][metric].values[0]
               for t in TEMPLATES]
        ax.bar(x + (i - 0.5) * w, vals, w, color=colors34[combo], label=combo)
    ax.axhline(0.8, color='#1baf7a', lw=1.2, ls='--', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([TEMPLATE_LABEL[t] for t in TEMPLATES])
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=10.5, color=INK)
    tidy(ax)
axes[0].legend(fontsize=8, loc='lower left')
fig.suptitle('Height alone vs. height+turn, across all 4 templates (pooled)',
            fontsize=11.5, color=INK)
fig.savefig(f'{FIG}/F34_precision_recall_by_template.png', bbox_inches='tight')
plt.close(fig)
print('wrote F34')

# ---- F35: direction-specific (left/right turned rear) result
#
# WARNING (2026-10-16): the script that produced this CSV was never kept -- it
# lived in a since-deleted /tmp scratchpad, so the published F35 PNG cannot be
# regenerated or audited. Worse, its pool sizes do not reconcile with
# stage24_seed_counts.csv (F35 reports median 26 left / 15 right per session,
# total 41, against a rear+turn seed median of 22.5), so F35 was evidently
# computed on a different pool than stage24 wrote. Treat F35 as unverifiable.
#
# scripts/stage26_direction_separation.py reimplements the direction split from
# stage24's own seeds and supersedes it (see SPATIAL_CONFOUND_RESULTS.md §8z).
# The path below is where a regenerated CSV should land; nothing writes it yet.
dir_csv = f'{RES}/stage24_directional_rear_turn.csv'
if os.path.exists(dir_csv):
    dd = pd.read_csv(dir_csv)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    combos35 = ['ear_mean_z', 'ear_mean_z+angle_skip_xy']
    x = np.arange(len(combos35)); w = 0.18
    offsets = {'left_P': -1.5, 'left_R': -0.5, 'right_P': 0.5, 'right_R': 1.5}
    colors35 = {'left_P': '#2a78d6', 'left_R': '#8fb4de', 'right_P': '#c4423f', 'right_R': '#e8817e'}
    labels35 = {'left_P': 'left, precision', 'left_R': 'left, recall',
               'right_P': 'right, precision', 'right_R': 'right, recall'}
    for key, off in offsets.items():
        direction, metric = key.split('_')
        vals = []
        for combo in combos35:
            g = dd[(dd.direction == direction) & (dd.combo == combo)]
            tp_, fp_, fn_ = g.tp.sum(), g.fp.sum(), g.fn.sum()
            v = tp_ / max(tp_ + fp_, 1) if metric == 'P' else tp_ / max(tp_ + fn_, 1)
            vals.append(v)
        ax.bar(x + off * w, vals, w, color=colors35[key], label=labels35[key])
    ax.axhline(0.8, color='#1baf7a', lw=1.2, ls='--', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(combos35, fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_title('Detecting LEFT- vs. RIGHT-turned rears separately',
                fontsize=10.5, color=INK)
    ax.legend(fontsize=7.5, loc='lower center', ncol=2)
    tidy(ax)
    fig.savefig(f'{FIG}/F35_directional_rear_turn.png', bbox_inches='tight')
    plt.close(fig)
    print('wrote F35')
else:
    print('skipping F35: stage24_directional_rear_turn.csv not present. The original\n'
          '  was lost with its scratchpad and is unverifiable (see note above);\n'
          '  stage26_direction_separation.py supersedes it (§8z).')
