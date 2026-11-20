"""Figures for the ear_mean_z rear-detector calibration and sector analysis."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from clirb.spatial.geometry import sector_index
from scripts.newcal_eras import ERAS, ERA_COLOR, ERA_LABEL, ERA_SHORT
from scripts.newcal_rear_detector import detect_rears_fast
from scripts.newcal_rear_sectors import load_traces, era_arenas, CALIBRATED, wilson

OUT = 'results/newcal'
FIG = f'{OUT}/figures'
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


def main():
    sw = pd.read_csv(f'{OUT}/rear_sweep.csv')
    ss = pd.read_csv(f'{OUT}/rear_sleap_sweep.csv')
    sec = pd.read_csv(f'{OUT}/rear_sector_by_era.csv')
    traces = load_traces(); arenas = era_arenas(traces)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))

    # --- A: F1 vs rise, matched thresholds -------------------------------
    ax = axes[0, 0]
    g = sw[(sw.min_frames == 20) & (sw.reset == sw.rise - 40)]
    for e in ERAS:
        q = g[g.era == e].groupby('rise')[['tp', 'n_sleap', 'n_dannce']].sum()
        p = q.tp / q.n_sleap; r = q.tp / q.n_dannce
        ax.plot(q.index, 2 * p * r / (p + r), color=ERA_COLOR[e], lw=2,
                marker='o', ms=3.5, label=ERA_LABEL[e])
    ax.axvline(CALIBRATED['rise'], color=MUTED, lw=1, ls='--')
    ax.annotate(f"calibrated\nrise = {CALIBRATED['rise']} mm",
                xy=(CALIBRATED['rise'] + 4, 0.89), fontsize=7.5, color=INK2)
    ax.axhline(0.8, color=MUTED, lw=0.8, ls=':')
    ax.set_xlabel('rise threshold (mm of mean ear height)')
    ax.set_ylabel('event F1 (SLEAP vs DANNCE)')
    ax.set_ylim(0.55, 1.02)
    ax.set_title('A · Threshold calibration — same detector run on each\n'
                 'modality (reset = rise − 40, min 1 s above rise)', loc='left')
    ax.legend(loc='lower left', fontsize=7.5); tidy(ax)

    # --- B: precision/recall, SLEAP rise swept vs fixed DANNCE -----------
    ax = axes[0, 1]
    for e in ERAS:
        q = ss[ss.era == e].groupby('sleap_rise')[['tp', 'n_sleap', 'n_dannce']].sum()
        ax.plot(100 * q.tp / q.n_dannce, 100 * q.tp / q.n_sleap,
                color=ERA_COLOR[e], lw=1.8, marker='o', ms=3)
        k = CALIBRATED['rise']
        if k in q.index:
            ax.scatter(100 * q.loc[k, 'tp'] / q.loc[k, 'n_dannce'],
                       100 * q.loc[k, 'tp'] / q.loc[k, 'n_sleap'],
                       s=90, facecolor='white', edgecolor=ERA_COLOR[e],
                       linewidth=2, zorder=5)
    ax.axvline(80, color=MUTED, lw=1, ls='--'); ax.axhline(80, color=MUTED, lw=1, ls='--')
    ax.set_xlabel('recall (%)'); ax.set_ylabel('precision (%)')
    ax.set_xlim(40, 102); ax.set_ylim(40, 102)
    ax.set_title('B · SLEAP rise swept against DANNCE held at the\n'
                 'calibrated point (ring = calibrated SLEAP rise)', loc='left')
    tidy(ax)
    ax.annotate('80/80 target', xy=(81, 43), fontsize=7.5, color=INK2)

    # --- C: per-sector recall with Wilson intervals ----------------------
    ax = axes[1, 0]
    W = 60
    for e in ['sept_baseline', 'oct_b']:
        q = (sec[(sec.op == 'calibrated') & (sec.sector_w == W) & (sec.era == e)]
             .groupby('sector').agg(c=('centre_deg', 'first'),
                                    n=('n_dannce', 'sum'), tp=('tp_recall', 'sum')))
        lo, hi = zip(*[wilson(k, n) for k, n in zip(q.tp, q.n)])
        rec = (q.tp / q.n).values * 100
        off = -4 if e == 'sept_baseline' else 4
        ax.errorbar(q.c + off, rec,
                    yerr=[rec - np.array(lo) * 100, np.array(hi) * 100 - rec],
                    fmt='o', ms=5, lw=1.4, capsize=3,
                    color=ERA_COLOR[e], label=f'{ERA_SHORT[e]} ({int(q.n.sum())} rears)')
    ax.set_xticks(np.arange(-150, 181, 60))
    ax.set_xlabel('arena sector — perimeter angle (deg)')
    ax.set_ylabel('recall (%)'); ax.set_ylim(55, 103)
    ax.set_title('C · Per-sector recall at the calibrated point\n'
                 '(bars = Wilson 95% interval)', loc='left')
    ax.legend(loc='lower right', fontsize=7.5); tidy(ax)

    # --- D: per-sector feature error, all frames vs non-rear frames ------
    ax = axes[1, 1]
    for e in ['sept_baseline', 'oct_b']:
        rows = []
        for (rat, s_), t in traces.items():
            if t['era'] != e:
                continue
            arena = arenas[e][0]
            sx = sector_index(arena.theta(t['com']), W)
            err = np.abs(t['zs'] - t['zd'])
            rear = np.zeros(len(err), bool)
            for a, b, _ in detect_rears_fast(t['zd'], **CALIBRATED):
                rear[a:b + 1] = True
            for k in range(int(360 // W)):
                m = sx == k
                rows.append(dict(sector=k, c=-180 + W * (k + 0.5),
                                 all=np.median(err[m]) if m.any() else np.nan,
                                 nonrear=np.median(err[m & ~rear]) if (m & ~rear).any() else np.nan))
        q = pd.DataFrame(rows).groupby(['sector', 'c']).median(numeric_only=True).reset_index()
        ax.plot(q.c, q['all'], color=ERA_COLOR[e], lw=2, marker='o', ms=4,
                label=f'{ERA_SHORT[e]} — all frames')
        ax.plot(q.c, q['nonrear'], color=ERA_COLOR[e], lw=1.4, ls='--',
                marker='s', ms=3.5, alpha=0.75,
                label=f'{ERA_SHORT[e]} — non-rear frames only')
    ax.set_xticks(np.arange(-150, 181, 60))
    ax.set_xlabel('arena sector — perimeter angle (deg)')
    ax.set_ylabel('median |SLEAP − DANNCE| ear height (mm)')
    ax.set_ylim(0, 3.6)
    ax.set_title('D · Does the feature itself vary by sector?\n'
                 '(≈2× spread — but see the scale vs the 150 mm threshold)',
                 loc='left')
    ax.legend(loc='upper right', fontsize=6.8); tidy(ax)

    fig.suptitle('ear_mean_z as a rear detector: threshold calibration and '
                 'arena-sector dependence', x=0.012, ha='left', fontsize=12.5, y=1.0)
    fig.text(0.012, -0.015,
             'Arena refitted per era from pooled DANNCE centre-of-mass — the '
             'fixed ARENA in clirb.spatial.geometry was fitted in the old room. '
             'Rears assigned to sectors by the DANNCE COM at the rear-END frame.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F12_rear_detector_sectors.png')
    plt.close(fig)
    print(f'wrote {FIG}/F12_rear_detector_sectors.png')


if __name__ == '__main__':
    main()
