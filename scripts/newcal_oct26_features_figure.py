"""F14: the two interpretable features on Oct 26, the rear-defined template
matching F1, and the DANNCE scale diagnostic."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from scripts.newcal_eras import ERAS, ERA_COLOR, ERA_LABEL, ERA_SHORT

OUT = 'results/newcal'; FIG = f'{OUT}/figures'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK, 'text.color': INK,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'grid.color': '#e8e7e2', 'grid.linewidth': 0.7,
    'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight',
})


def tidy(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True)


def strip(ax, d, col, log=False):
    rng = np.random.default_rng(0)
    for i, e in enumerate(ERAS):
        y = d[d.era == e][col].dropna().values
        if not len(y):
            continue
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(y)), y, s=24, color=ERA_COLOR[e],
                   alpha=0.78, edgecolor='white', linewidth=0.6, zorder=3)
        ax.plot([i - 0.32, i + 0.32], [np.median(y)] * 2, color=INK, lw=1.8, zorder=4)
    ax.set_xticks(range(len(ERAS)))
    ax.set_xticklabels([ERA_SHORT[e] for e in ERAS], fontsize=8)
    ax.set_xlim(-0.6, len(ERAS) - 0.4)
    if log:
        ax.set_yscale('log')
    tidy(ax)


def main():
    ft = pd.read_csv(f'{OUT}/feature_tracking.csv')
    tm = pd.read_csv(f'{OUT}/feature_template_matching.csv')
    sc_ = pd.read_csv(f'{OUT}/scale_diagnostic.csv')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))

    # A: ear_mean_z RMSE
    ax = axes[0, 0]
    strip(ax, ft[ft.feature == 'ear_mean_z'], 'rmse', log=True)
    ax.set_ylabel('RMSE (mm, log)')
    ax.set_title('A · ear_mean_z — SLEAP vs DANNCE', loc='left')

    # B: cum_heading_change RMSE
    ax = axes[0, 1]
    strip(ax, ft[ft.feature == 'cum_heading_change'], 'rmse', log=True)
    ax.set_ylabel('RMSE (deg, log)')
    ax.set_title('B · cum_heading_change — SLEAP vs DANNCE', loc='left')

    # C: conditional turn-sign agreement
    ax = axes[0, 2]
    strip(ax, ft[ft.feature == 'cum_heading_change'], 'sign_agree_turn_pct')
    ax.axhline(99.5, color=ERA_COLOR['sept_baseline'], lw=1, ls='--')
    ax.annotate('§8ab published 99.5%', xy=(3.4, 99.3), ha='right', va='top',
                fontsize=7.5, color=ERA_COLOR['sept_baseline'])
    ax.set_ylabel('sign agreement (%)')
    ax.set_title('C · Turn sign when actually turning (|value| ≥ 45°)', loc='left')

    # D: template matching F1
    ax = axes[1, 0]
    combos = ['ear_mean_z', 'ear_mean_z+cum_heading_change', 'cum_heading_change']
    short = ['ear_mean_z', 'both', 'cum_heading_change']
    x = np.arange(len(combos)); w = 0.8 / len(ERAS)
    g = tm.groupby(['combo', 'era'])[['tp', 'fp', 'fn']].sum()
    for i, e in enumerate(ERAS):
        f1 = []
        for c in combos:
            try:
                r = g.loc[(c, e)]
                p = r.tp / max(r.tp + r.fp, 1); rc = r.tp / max(r.tp + r.fn, 1)
                f1.append(2 * p * rc / max(p + rc, 1e-9))
            except KeyError:
                f1.append(np.nan)
        ax.bar(x + (i - (len(ERAS) - 1) / 2) * w, f1, width=w * 0.9,
               color=ERA_COLOR[e], label=ERA_SHORT[e])
    ax.axhline(0.8, color=MUTED, lw=1, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_ylabel('F1 (SLEAP vs DANNCE matches)'); ax.set_ylim(0, 1.05)
    ax.set_title('D · Template matching, rear-defined template', loc='left')
    ax.legend(fontsize=7, ncol=2); tidy(ax)

    # E: how selective is each corridor?
    ax = axes[1, 1]
    for i, e in enumerate(ERAS):
        v = []
        for c in combos:
            s = tm[(tm.combo == c) & (tm.era == e)]
            v.append(s.n_dannce.sum() / max(s.n_gt.sum(), 1))
        ax.bar(x + (i - (len(ERAS) - 1) / 2) * w, v, width=w * 0.9, color=ERA_COLOR[e])
    ax.set_yscale('log'); ax.axhline(1, color=MUTED, lw=1, ls='--')
    ax.annotate('1 match per real rear', xy=(2.45, 1.15), ha='right', fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
    ax.set_ylabel('matches ÷ true rears (log)')
    ax.set_title('E · Corridor selectivity — the turn feature fires\n'
                 'tens of times per real rear', loc='left')
    tidy(ax)

    # F: the scale diagnostic
    ax = axes[1, 2]
    era_of = {'sept_baseline': 'Sep', 'oct_b': 'Oct22-23', 'oct_26': 'Oct26'}
    extent = {'Sep': (0.965, 1.081), 'Oct22-23': (0.885, 1.051), 'Oct26': (0.760, 1.054)}
    labs = list(extent)
    xx = np.arange(len(labs))
    arena = [sc_[sc_.era == k].arena_ratio.median()
             for k in ['sept_baseline', 'oct_b', 'oct_26']]
    ax.plot(xx, arena, 'o-', color=INK2, lw=1.8, ms=7, label='arena radius')
    ax.plot(xx, [extent[k][0] for k in labs], 'o-', color='#e34948', lw=2, ms=7,
            label='body, along its own axis')
    ax.plot(xx, [extent[k][1] for k in labs], 'o-', color='#2a78d6', lw=2, ms=7,
            label='body, across its axis')
    ax.axhline(1.0, color=MUTED, lw=1, ls='--')
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel('DANNCE ÷ SLEAP'); ax.set_ylim(0.70, 1.15)
    ax.set_title('F · A scale error would move all three together.\nIt does not.', loc='left')
    ax.legend(fontsize=7, loc='lower left'); tidy(ax)

    fig.suptitle('Oct 26: the two interpretable features, rear-template matching, '
                 'and what the DANNCE size mismatch actually is',
                 x=0.012, ha='left', fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(f'{FIG}/F14_oct26_features_and_scale.png')
    plt.close(fig)
    print(f'wrote {FIG}/F14_oct26_features_and_scale.png')


if __name__ == '__main__':
    main()
