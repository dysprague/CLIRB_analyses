"""F19: confidence-weighted triangulation across eras, and the fine-tune headroom."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from scripts.newcal_eras import ERAS, ERA_COLOR, ERA_SHORT

OUT = 'results/newcal'; FIG = f'{OUT}/figures'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
    'text.color': INK, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'grid.color': '#e8e7e2', 'grid.linewidth': 0.7,
    'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def tidy(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True)


def main():
    d = pd.read_csv(f'{OUT}/weighted_reanalysis.csv')
    h = pd.read_csv(f'{OUT}/finetune_headroom.csv')
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))

    # A: paired per-session change, key metric
    ax = axes[0]
    for i, e in enumerate(ERAS):
        s = d[d.era == e]
        for _, row in s.iterrows():
            ax.plot([i - 0.16, i + 0.16], [row.eq_kp_med, row.wt_kp_med],
                    color=ERA_COLOR[e], lw=1.1, alpha=0.65, marker='o', ms=3.5)
        if len(s):
            ax.plot([i - 0.16, i + 0.16], [s.eq_kp_med.median(), s.wt_kp_med.median()],
                    color=INK, lw=2.4, marker='o', ms=5, zorder=5)
    ax.set_xticks(range(len(ERAS)))
    ax.set_xticklabels([f'{ERA_SHORT[e]}\nequal→wtd' for e in ERAS], fontsize=7)
    ax.set_ylabel('median keypoint error vs DANNCE (mm)')
    ax.set_title('A · Paired, per session', loc='left')
    tidy(ax)

    # B: % change by metric and era
    ax = axes[1]
    mets = [('bone_cv', 'bone CV'), ('jitter', 'jitter'), ('kp_med', 'kp error'),
            ('ear_rmse', 'ear_mean_z'), ('turn_rmse', 'cum_heading'),
            ('fit_residual', 'Procrustes')]
    x = np.arange(len(mets)); w = 0.8 / len(ERAS)
    for i, e in enumerate(ERAS):
        s = d[d.era == e]
        v = [100 * (s[f'wt_{m}'].median() / s[f'eq_{m}'].median() - 1) for m, _ in mets]
        ax.bar(x + (i - (len(ERAS) - 1) / 2) * w, v, width=w * 0.9,
               color=ERA_COLOR[e], label=ERA_SHORT[e])
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in mets], rotation=30,
                                         ha='right', fontsize=7.5)
    ax.set_ylabel('change with weighting (%)')
    ax.set_title('B · Improvement is consistent across eras', loc='left')
    ax.legend(fontsize=7, ncol=2); tidy(ax)

    # C: leave-one-out error vs confidence
    ax = axes[2]
    g = (h.groupby(['era', 'stratum', 'conf_lo'])
          .apply(lambda x: pd.Series({'n': x.n.sum(),
                                      'err': np.average(x.loo_err_med, weights=x.n)}),
                 include_groups=False).reset_index())
    for (era, stratum), col, ls in [(('Sep', 'far from port'), '#2a78d6', '-'),
                                    (('Sep', 'near port'), '#2a78d6', '--'),
                                    (('Oct26', 'far from port'), '#eda100', '-'),
                                    (('Oct26', 'near port'), '#eda100', '--')]:
        s = g[(g.era == era) & (g.stratum == stratum)]
        if len(s):
            ax.plot(s.conf_lo + 0.07, s.err, marker='o', ms=4, lw=1.8,
                    color=col, ls=ls, label=f'{era} {stratum.split()[0]}')
    ax.set_xlabel("held-out camera's own confidence")
    ax.set_ylabel('leave-one-out error (px)')
    ax.set_title('C · Confidence is well calibrated —\nlow confidence really is wrong',
                 loc='left')
    ax.legend(fontsize=7); tidy(ax)

    # D: the 50/50 split
    ax = axes[3]
    ax.bar([0], [19.7], color='#2a78d6', width=0.6)
    ax.bar([1], [29.8], color='#eda100', width=0.6)
    ax.bar([2], [25.0], color='#1baf7a', width=0.6)
    ax.bar([3], [24.6], color='#4a3aa7', width=0.6)
    for i, v in zip(range(4), [19.7, 29.8, 25.0, 24.6]):
        ax.annotate(f'{v:.1f}', xy=(i, v), xytext=(0, 2), textcoords='offset points',
                    ha='center', fontsize=7.5, color=INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Sep\nfar', 'Oct26\nnear', 'if conf mix\nwere Sep\'s',
                        'if accuracy\nwere Sep\'s'], fontsize=7)
    ax.set_ylabel('leave-one-out error (px)')
    ax.set_ylim(0, 34)
    ax.set_title('D · The port penalty splits ~50/50', loc='left')
    tidy(ax)
    ax.annotate('−48 %', xy=(2, 26.5), ha='center', fontsize=8, color='#1baf7a')
    ax.annotate('−51 %', xy=(3, 26.1), ha='center', fontsize=8, color='#4a3aa7')

    fig.suptitle('Confidence-weighted triangulation helps everywhere; the port '
                 'penalty is half fewer confident views, half worse detections',
                 x=0.012, ha='left', fontsize=12, y=1.03)
    fig.text(0.012, -0.08,
             'A–B: both variants from the same sleap_keys_2D over the same '
             'frames. D: counterfactuals re-weight Oct26-near by September\'s '
             'confidence distribution (green) or substitute September\'s '
             'per-confidence accuracy (violet).',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F19_weighted_and_headroom.png')
    plt.close(fig)
    print(f'wrote {FIG}/F19_weighted_and_headroom.png')


if __name__ == '__main__':
    main()
