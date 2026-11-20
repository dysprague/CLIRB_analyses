"""F18: SLEAP's own tracking quality across arena sectors."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from scripts.newcal_eras import ERA_COLOR

OUT = 'results/newcal'; FIG = f'{OUT}/figures'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
ERAS3 = [('sept_baseline', 'September — no port mechanism'),
         ('oct_b', 'Oct 22–23 — mechanism in'),
         ('oct_26', 'Oct 26 — mechanism in, task on')]
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
    d = pd.read_csv(f'{OUT}/sleap_sector_quality.csv')
    specs = [('reproj_px', 'SLEAP 3D vs its own 2D (px)', 'A'),
             ('ncam_conf', 'cameras with confident\ndetections (of 3)', 'B'),
             ('pair_spread_mm', 'two-camera triangulation\ndisagreement (mm)', 'C')]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))

    for ax, (col, ylab, tag) in zip(axes[:3], specs):
        for era, lab in ERAS3:
            s = d[d.era == era]
            g = s.groupby('sector').agg(c=('centre_deg', 'first'),
                                        port=('is_port_sector', 'max'),
                                        v=(col, 'median')).reset_index().dropna()
            ax.plot(g.c, g.v, marker='o', ms=4.5, lw=1.8, color=ERA_COLOR[era],
                    label=lab.split(' — ')[0])
        pc = d[d.is_port_sector].centre_deg.median()
        ax.axvspan(pc - 30, pc + 30, color='#eb6834', alpha=0.12, lw=0, zorder=0)
        ax.annotate('water port', xy=(pc, ax.get_ylim()[1]), xytext=(0, -11),
                    textcoords='offset points', ha='center', fontsize=7.5,
                    color='#eb6834')
        ax.set_xticks(np.arange(-150, 180, 60))
        ax.set_xlabel('arena sector (deg)'); ax.set_ylabel(ylab)
        ax.set_title(f'{tag} · {ylab.splitlines()[0]}', loc='left', fontsize=9)
        tidy(ax)
    axes[0].legend(fontsize=7)

    # D: port-sector penalty, SLEAP self-consistency vs the DANNCE contrast
    ax = axes[3]
    met = [('reproj_px', 'SLEAP reproj'), ('pair_spread_mm', 'SLEAP 2-cam'),
           ('bone_cv', 'SLEAP bone CV'), ('dannce_rmse_mm', 'vs DANNCE')]
    x = np.arange(len(met)); w = 0.8 / 3
    for i, (era, _) in enumerate(ERAS3):
        s = d[d.era == era]
        rel = []
        for c, _ in met:
            p = s[s.is_port_sector][c].median()
            o = s[~s.is_port_sector][c].median()
            rel.append(100 * (p / o - 1) if o and np.isfinite(o) else np.nan)
        ax.bar(x + (i - 1) * w, rel, width=w * 0.9, color=ERA_COLOR[era])
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in met], rotation=30,
                                         ha='right', fontsize=7.5)
    ax.set_ylabel('port sector vs elsewhere (%)')
    ax.set_title('D · The port-sector penalty', loc='left')
    tidy(ax)

    fig.suptitle("SLEAP's own tracking quality by arena sector — uniform in "
                 'September, degraded at the port once the mechanism was added',
                 x=0.012, ha='left', fontsize=12, y=1.03)
    fig.text(0.012, -0.08,
             'A–C use no DANNCE at all. The right-hand bar in D is the '
             'SLEAP-vs-DANNCE disagreement for contrast: it roughly doubles at '
             'the port while SLEAP\'s own self-consistency degrades ~20 %, '
             'which is the split between the two systems.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F18_sleap_sector_quality.png')
    plt.close(fig)
    print(f'wrote {FIG}/F18_sleap_sector_quality.png')


if __name__ == '__main__':
    main()
