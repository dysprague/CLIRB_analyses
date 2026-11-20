"""F17: the DANNCE size bias is localised to the water-port sector, and it
appeared with the port mechanism."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from scripts.newcal_eras import ERA_COLOR

OUT = 'results/newcal'; FIG = f'{OUT}/figures'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
ERAS3 = [('sept_baseline', 'September — no port mechanism'),
         ('oct_b', 'Oct 22–23 — mechanism in, task off'),
         ('oct_26', 'Oct 26 — mechanism in, task on')]
BONES = ['SpineF-TailBase', 'HipL-KneeL', 'ShoulderL-ElbowL', 'Snout-SpineF']
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
    d = pd.read_csv(f'{OUT}/sector_scale_bias.csv')
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))

    # A-C: per-sector profile for one representative bone, one panel per era
    for ax, (era, title) in zip(axes[:3], ERAS3):
        s = d[d.era == era]
        g = s.groupby('sector').agg(c=('centre_deg', 'first'),
                                    port=('is_port_sector', 'max'),
                                    occ=('frac', 'median'),
                                    **{b: (b, 'median') for b in BONES}).reset_index()
        pc = g.loc[g.port, 'c']
        if len(pc):
            ax.axvspan(pc.iloc[0] - 30, pc.iloc[0] + 30, color='#eb6834',
                       alpha=0.13, lw=0, zorder=0)
            ax.annotate('water port', xy=(pc.iloc[0], 1.30), ha='center',
                        fontsize=7.5, color='#eb6834')
        for b, col, mk in zip(BONES, ['#2a78d6', '#1baf7a', '#4a3aa7', '#e34948'],
                              ['o', 's', '^', 'D']):
            ax.plot(g.c, g[b], marker=mk, ms=4.5, lw=1.6, color=col, label=b)
        ax.axhline(1.0, color=MUTED, lw=1, ls='--')
        ax.set_xticks(np.arange(-150, 180, 60))
        ax.set_ylim(0.62, 1.35)
        ax.set_xlabel('arena sector (deg)')
        ax.set_ylabel('DANNCE ÷ SLEAP bone length')
        ax.set_title(title, loc='left', fontsize=9)
        tidy(ax)
    axes[0].legend(fontsize=6.6, loc='lower left', ncol=2)

    # D: port sector minus elsewhere, per era
    ax = axes[3]
    x = np.arange(len(BONES)); w = 0.8 / 3
    for i, (era, _) in enumerate(ERAS3):
        s = d[d.era == era]
        delta = [s[s.is_port_sector][b].median() - s[~s.is_port_sector][b].median()
                 for b in BONES]
        ax.bar(x + (i - 1) * w, delta, width=w * 0.9, color=ERA_COLOR[era],
               label=era.replace('sept_baseline', 'Sep').replace('oct_b', 'Oct22-23')
                       .replace('oct_26', 'Oct26'))
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('-', '–') for b in BONES], rotation=30,
                       ha='right', fontsize=7.5)
    ax.set_ylabel('port sector − elsewhere')
    ax.set_title('D · The port-sector penalty, by era', loc='left')
    ax.legend(fontsize=7.5); tidy(ax)

    fig.suptitle('The DANNCE size bias is localised to the water-port sector — '
                 'absent in September, present once the mechanism was added',
                 x=0.012, ha='left', fontsize=12, y=1.03)
    fig.text(0.012, -0.08,
             'Ratios are posture-free: both modalities see the same frames. '
             'September already had 46 % of its occupancy in the port sector and '
             'shows no penalty there, so this is not an occupancy artifact.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F17_sector_scale_bias.png')
    plt.close(fig)
    print(f'wrote {FIG}/F17_sector_scale_bias.png')


if __name__ == '__main__':
    main()
