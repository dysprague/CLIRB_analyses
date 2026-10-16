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
CAT_COL = {'both': '#1baf7a', 'sleap_only': '#2a78d6', 'dannce_only': '#e34948'}
CAT_LABEL = {'both': 'both (TP)', 'sleap_only': 'SLEAP only (FP)', 'dannce_only': 'DANNCE only (FN)'}
FEAT_LABEL = {'full23': 'full 23kp\n(2-PC baseline)', 'axial': 'axial 9kp\n(1-PC, unfair)',
             'axial2': 'axial 9kp\n(2-PC, fair)', 'snout_z': 'snout\nheight',
             'bodyrot': 'body\nrotation', 'snout_bodyrot': 'snout +\nbodyrot'}
FEATS = ['full23', 'axial', 'axial2', 'snout_z', 'bodyrot', 'snout_bodyrot']
RATS = ['R4', 'R5', 'R6']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 2.0, 'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax, ygrid=True):
    ax.grid(axis='y' if ygrid else 'x', alpha=0.9, zorder=0); ax.set_axisbelow(True)


ev = pd.read_csv(f'{RES}/stage18_events.csv')
summ = pd.read_csv(f'{RES}/stage18_session_summary.csv')

# ============================================================ FIG 17 (Plot 1)
base = ev[ev.feature == 'full23']
base_s = summ[summ.feature == 'full23'].copy()

fig = plt.figure(figsize=(12.5, 11.2), constrained_layout=True)
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.15, 0.9])

# top row: aggregate counts + precision/recall, per rat
ax = fig.add_subplot(gs[0, 0])
x = np.arange(3)
bottom = np.zeros(3)
for cat in ['both', 'sleap_only', 'dannce_only']:
    vals = [base[(base.rat == r) & (base.category == cat)].shape[0] for r in RATS]
    ax.bar(x, vals, bottom=bottom, color=CAT_COL[cat], label=CAT_LABEL[cat], width=0.6)
    bottom += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels(RATS)
ax.set_ylabel('# events (all sessions pooled)')
ax.set_title('Aggregate: TP / FP / FN counts', fontsize=10, color=INK)
ax.legend(fontsize=7.5, loc='upper right'); tidy(ax)

ax = fig.add_subplot(gs[0, 1])
prec = [base_s[base_s.rat == r].apply(lambda row: row['both'], axis=1).sum() /
       max((base_s[base_s.rat == r]['both'].sum() + base_s[base_s.rat == r]['sleap_only'].sum()), 1)
       for r in RATS]
rec = [base_s[base_s.rat == r]['both'].sum() /
      max((base_s[base_s.rat == r]['both'].sum() + base_s[base_s.rat == r]['dannce_only'].sum()), 1)
      for r in RATS]
w = 0.32
ax.bar(x - w / 2, prec, w, color='#2a78d6', label='precision (TP/(TP+FP))')
ax.bar(x + w / 2, rec, w, color='#eb6834', label='recall (TP/(TP+FN))')
ax.set_xticks(x); ax.set_xticklabels(RATS); ax.set_ylim(0, 1.0)
ax.set_title('Aggregate: precision & recall', fontsize=10, color=INK)
ax.legend(fontsize=7.5, loc='lower right'); tidy(ax)

ax = fig.add_subplot(gs[0, 2])
overall_p = base[base.category == 'both'].shape[0] / max(
    base[base.category.isin(['both', 'sleap_only'])].shape[0], 1)
overall_r = base[base.category == 'both'].shape[0] / max(
    base[base.category.isin(['both', 'dannce_only'])].shape[0], 1)
ax.bar(['precision', 'recall'], [overall_p, overall_r], color=[MUTED, INK2], width=0.5)
for i, v in enumerate([overall_p, overall_r]):
    ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(0, 6),
               ha='center', fontsize=10, weight='bold', color=INK)
ax.set_ylim(0, 1.0); ax.set_title('All 3 rats pooled', fontsize=10, color=INK)
tidy(ax)

# bottom row: per-session over time, precision and recall, per rat
ax1 = fig.add_subplot(gs[1, :])
ax2 = ax1.twinx()
for rat in RATS:
    g = base_s[base_s.rat == rat].sort_values('session_idx')
    k = max(3, len(g) // 10)
    ax1.plot(g.session_idx, g.precision.rolling(k, center=True, min_periods=1).mean(),
             '-', color=SERIES[rat], lw=2.2, label=f'{rat} precision')
    ax1.plot(g.session_idx, g.recall.rolling(k, center=True, min_periods=1).mean(),
             '--', color=SERIES[rat], lw=1.6, alpha=0.7)
ax1.set_xlabel('session index (chronological, within rat)')
ax1.set_ylabel('rolling mean (solid=precision, dashed=recall)')
ax1.set_ylim(0, 1.0)
ax1.set_title('Per-session precision (solid) and recall (dashed) over time',
              fontsize=10.5, color=INK)
ax1.legend(fontsize=8, ncol=3, loc='lower center')
ax2.set_yticks([])
tidy(ax1)

# new row: per-session sample size behind each precision/recall point -- total
# REAL rears (hysteresis-detected directly on the DANNCE PC0 trajectory, never
# touching any template match) broken into exclusive categories: did that rear
# get a DANNCE-side match, a SLEAP-side match, both, or neither (full23 only).
rear_summ = pd.read_csv(f'{RES}/stage18_rear_session_summary.csv')
REAR_CATS = ['both', 'dannce_only', 'sleap_only', 'neither']
REAR_CAT_COL = {'both': CAT_COL['both'], 'dannce_only': CAT_COL['dannce_only'],
                'sleap_only': CAT_COL['sleap_only'], 'neither': '#c7c4bc'}
REAR_CAT_LABEL = {'both': 'both', 'dannce_only': 'DANNCE only', 'sleap_only': 'SLEAP only',
                  'neither': 'neither (matched by neither)'}
rear_axes = []
for col, rat in enumerate(RATS):
    ax3 = fig.add_subplot(gs[2, col])
    g = rear_summ[rear_summ.rat == rat].sort_values('session_idx')
    bottom = np.zeros(len(g))
    for cat in REAR_CATS:
        ax3.bar(g.session_idx, g[cat], bottom=bottom, color=REAR_CAT_COL[cat],
                label=REAR_CAT_LABEL[cat], width=0.9, linewidth=0)
        bottom += g[cat].to_numpy()
    ax3.set_xlabel('session index')
    ax3.set_title(f'{rat} — sample size behind each point above (n={int(g[REAR_CATS].to_numpy().sum())} real rears)',
                 fontsize=9, color=INK)
    tidy(ax3)
    rear_axes.append(ax3)
rear_axes[0].set_ylabel('# real (hysteresis-detected) rears,\nby exclusive match outcome')
rear_axes[0].legend(fontsize=7, loc='upper right')

fig.suptitle('Plot 1 — SLEAP vs. DANNCE accuracy, baseline (full 23kp) feature space',
            fontsize=12.5, color=INK)
fig.savefig(f'{FIG}/F17_overlap_overall.png', bbox_inches='tight')
plt.close(fig)
print('wrote F17')

# ============================================================ FIG 18 (Plot 2)
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
for col, rat in enumerate(RATS):
    g = base[base.rat == rat]
    rows_p, rows_r, xs = [], [], []
    for s in range(12):
        gg = g[g.sector == s]
        tp_ = (gg.category == 'both').sum(); fp_ = (gg.category == 'sleap_only').sum()
        fn_ = (gg.category == 'dannce_only').sum()
        if tp_ + fp_ + fn_ < 20:
            continue
        xs.append(s * 30 - 165)
        rows_p.append(tp_ / max(tp_ + fp_, 1)); rows_r.append(tp_ / max(tp_ + fn_, 1))
    ax = axes[0, col]
    ax.plot(xs, rows_p, 'o-', color=SERIES[rat], ms=5, lw=2)
    ax.set_title(f'{rat} — precision by sector', fontsize=9.5, color=INK)
    ax.set_ylim(0.4, 1.0); ax.set_xticks([-165, -75, 15, 105])
    tidy(ax)
    ax = axes[1, col]
    ax.plot(xs, rows_r, 'o-', color=SERIES[rat], ms=5, lw=2)
    ax.set_title(f'{rat} — recall by sector', fontsize=9.5, color=INK)
    ax.set_ylim(0.2, 1.0); ax.set_xticks([-165, -75, 15, 105])
    ax.set_xlabel('arena angle θ (deg)')
    tidy(ax)
axes[0, 0].set_ylabel('precision'); axes[1, 0].set_ylabel('recall')
fig.suptitle('Plot 2 — SLEAP vs. DANNCE precision/recall by sector, baseline feature space',
            fontsize=12, color=INK)
fig.savefig(f'{FIG}/F18_overlap_by_sector.png', bbox_inches='tight')
plt.close(fig)
print('wrote F18')

# ============================================================ FIG 19 (Plot 3)
rows = []
for feat in FEATS:
    for rat in RATS:
        g = ev[(ev.feature == feat) & (ev.rat == rat)]
        tp_ = (g.category == 'both').sum(); fp_ = (g.category == 'sleap_only').sum()
        fn_ = (g.category == 'dannce_only').sum()
        rows.append(dict(feature=feat, rat=rat, precision=tp_ / max(tp_ + fp_, 1),
                         recall=tp_ / max(tp_ + fn_, 1), n=tp_ + fp_ + fn_))
s = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
x = np.arange(len(FEATS)); w = 0.26
for ax, metric, title in [(axes[0], 'precision', 'Precision (of SLEAP matches, how many are real)'),
                          (axes[1], 'recall', 'Recall (of real matches, how many SLEAP catches)')]:
    for i, rat in enumerate(RATS):
        g = s[s.rat == rat].set_index('feature').loc[FEATS]
        ax.bar(x + (i - 1) * w, g[metric], w, color=SERIES[rat], label=rat, edgecolor=SURFACE, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([FEAT_LABEL[f] for f in FEATS], fontsize=8.5)
    ax.axvline(1.5, color=INK, lw=0.7, ls=':', alpha=0.4)
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=10, color=INK)
    tidy(ax)
axes[0].legend(fontsize=8.5, loc='lower right')
fig.suptitle('Plot 3 — 6 feature spaces compared, DANNCE-ground-truthed overlap test\n'
            'axial (1-PC) looked dominant, but matched PC-for-PC (axial2, 2-PC) it is a wash with full23 '
            '(worse precision, better recall) — dimensionality, not the axial keypoints, drove most of the earlier gap',
            fontsize=11, color=INK, linespacing=1.4)
fig.savefig(f'{FIG}/F19_overlap_feature_comparison.png', bbox_inches='tight')
plt.close(fig)
print('wrote F19')
