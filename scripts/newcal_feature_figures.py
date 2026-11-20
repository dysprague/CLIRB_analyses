"""Figures for the interpretable-feature comparison (ear_mean_z, cum_heading_change).

Reads results/newcal/feature_tracking.csv for the summary panels and
recomputes two 60-second example traces directly from the cache for the
qualitative panels.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb import features as cf
from clirb.spatial import dual as du
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample
from scripts.newcal_feature_tracking import REAR_RISE, REAR_RESET, TURN_DEG

OUT = 'results/newcal'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)

from scripts.newcal_eras import (ERAS, ERA_COLOR, ERA_LABEL, ERA_SHORT,
                                 add_group, ordered_groups)

C_SEPT, C_NEW, C_THIRD = ERA_COLOR['sept_baseline'], ERA_COLOR['oct_a'], ERA_COLOR['oct_b']
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


def strip(ax, d, col, order, log=False):
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        sub = d[d.grp == g]
        if not len(sub):
            continue
        y = sub[col].dropna().values
        if not len(y):
            continue
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(y)), y, s=22,
                   color=ERA_COLOR[sub.era.iloc[0]], alpha=0.75,
                   edgecolor='white', linewidth=0.6, zorder=3)
        ax.plot([i - 0.32, i + 0.32], [np.median(y)] * 2, color=INK, lw=1.8, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7.5, rotation=45, ha='right')
    ax.set_xlim(-0.6, len(order) - 0.4)
    if log:
        ax.set_yscale('log')
    tidy(ax)


def trace(rat, session, start, n):
    raw = sc.get_raw_session(rat, session, want_dual=True)
    d3 = du.dannce_aligned(raw)
    m = min(len(raw['triang_3d']), len(d3))
    sl = np.asarray(raw['triang_3d'][:m], np.float32)
    dn = np.asarray(d3[:m], np.float32)
    samp, _ = balanced_sample(sl.mean(axis=1)[:, :2], rng=np.random.default_rng(0))
    fit = robust_proc(dn[samp], sl[samp])
    S = ev.causal_median_5_kp(sl, warmup='online').copy()
    D = ev.causal_median_5_kp(fit['apply'](dn), warmup='online').copy()
    S[:, :, 2] *= -1; D[:, :, 2] *= -1
    fs, fd = cf.compute_features(S), cf.compute_features(D)
    sl_ = slice(start, start + n)
    return {k: (fs[k][sl_], fd[k][sl_]) for k in ('ear_mean_z', 'cum_heading_change')}


def main():
    d = add_group(pd.read_csv(f'{OUT}/feature_tracking.csv'), sep=' ')
    order = ordered_groups(d)
    h = d[d.feature == 'ear_mean_z'].copy()
    t = d[d.feature == 'cum_heading_change'].copy()

    START, NFR = 12000, 1200
    TRACES = [('sept_baseline', 'R5', '2026_09_29_1'),
              ('oct_a', 'R7', '2026_10_21_1'),
              ('oct_b', 'R7', '2026_10_22_1')]
    tr = {e: trace(rat, ses, START, NFR) for e, rat, ses in TRACES}
    tt = np.arange(NFR) / ev.FPS

    fig = plt.figure(figsize=(12.5, 8.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.45], hspace=0.52, wspace=0.3)

    # ---- row 1: ear_mean_z -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    strip(ax, h, 'rmse', order, log=True)
    for i, g in enumerate(order):
        y = h[h.grp == g]['med_abs'].dropna().values
        if len(y):
            ax.scatter(i, np.median(y), s=46, marker='D', facecolor='white',
                       edgecolor=INK2, linewidth=1.1, zorder=5)
    ax.set_ylabel('error (mm, log)')
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='', color=MUTED, markersize=6, label='RMSE (per session)'),
        Line2D([], [], marker='D', ls='', mfc='white', mec=INK2, markersize=6,
               label='median |error| (group)')], loc='upper left', fontsize=7)
    ax.set_title('A · ear_mean_z error', loc='left')

    ax = fig.add_subplot(gs[0, 1])
    for e in ERAS:
        s = h[h.era == e]
        ax.scatter(100 * s.event_recall, 100 * s.event_precision, s=30,
                   color=ERA_COLOR[e], alpha=0.8, edgecolor='white', linewidth=0.6,
                   label=ERA_LABEL[e], zorder=3)
    ax.axvline(80, color=MUTED, lw=1, ls='--'); ax.axhline(80, color=MUTED, lw=1, ls='--')
    ax.set_xlim(60, 102); ax.set_ylim(60, 102)
    ax.set_xlabel('recall (%)'); ax.set_ylabel('precision (%)')
    ax.set_title('B · Rearing events detected from\near_mean_z alone, ±300 ms', loc='left')
    ax.legend(loc='lower left', fontsize=6.5); tidy(ax)
    ax.annotate('80/80 target', xy=(81, 61), fontsize=7.5, color=INK2)

    sub = gs[0, 2].subgridspec(3, 1, hspace=0.22)
    for row, (e, rat_, ses_) in enumerate(TRACES):
        lab = f'{rat_} {ses_}'
        ax = fig.add_subplot(sub[row])
        a, b = tr[e]['ear_mean_z']
        ax.plot(tt, b, color=MUTED, lw=1.5, zorder=2, label='DANNCE')
        ax.plot(tt, a, color=ERA_COLOR[e], lw=0.9, zorder=3, label='SLEAP')
        ax.axhline(REAR_RISE, color=INK2, lw=0.8, ls=':')
        ax.set_ylim(20, 260)
        ax.set_ylabel('ear ht (mm)', fontsize=7.5)
        ax.annotate(lab, xy=(0.99, 0.93), xycoords='axes fraction', ha='right',
                    va='top', fontsize=7, color=ERA_COLOR[e])
        if row == 0:
            ax.set_title('C · 60 s of ear height, one session per condition\n'
                         '(dotted = rear threshold)', loc='left')
            ax.legend(loc='upper left', fontsize=6.5, ncol=2)
        if row < len(TRACES) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('time (s)')
        tidy(ax)

    # ---- row 2: cum_heading_change ----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    strip(ax, t, 'rmse', order, log=True)
    for i, g in enumerate(order):
        y = t[t.grp == g]['med_abs'].dropna().values
        if len(y):
            ax.scatter(i, np.median(y), s=46, marker='D', facecolor='white',
                       edgecolor=INK2, linewidth=1.1, zorder=5)
    ax.set_ylabel('error (deg, log)')
    ax.set_title('D · cum_heading_change error', loc='left')

    ax = fig.add_subplot(gs[1, 1])
    strip(ax, t, 'sign_agree_turn_pct', order)
    ax.axhline(99.5, color=C_SEPT, lw=1, ls='--')
    ax.annotate('§8ab published 99.5%', xy=(len(order) - 0.5, 99.4), ha='right',
                va='top', fontsize=7.5, color=C_SEPT)
    ax.set_ylabel('sign agreement (%)')
    ax.set_title('E · Turn-sign agreement when the rat\nis actually turning (|value| ≥ 45°)',
                 loc='left')

    subf = gs[1, 2].subgridspec(3, 1, hspace=0.22)
    for row, (e, rat_, ses_) in enumerate(TRACES):
        lab = f'{rat_} {ses_}'
        ax = fig.add_subplot(subf[row])
        a, b = tr[e]['cum_heading_change']
        ax.plot(tt, b, color=MUTED, lw=1.5, zorder=2)
        ax.plot(tt, a, color=ERA_COLOR[e], lw=0.9, zorder=3)
        for thr in (TURN_DEG, -TURN_DEG):
            ax.axhline(thr, color=INK2, lw=0.8, ls=':')
        ax.set_ylim(-210, 210)
        ax.set_ylabel('turn (deg)', fontsize=7.5)
        ax.annotate(lab, xy=(0.99, 0.95), xycoords='axes fraction', ha='right',
                    va='top', fontsize=7, color=ERA_COLOR[e])
        if row == 0:
            ax.set_title('F · Same sessions, accumulated turn\n'
                         '(dotted = ±45°)', loc='left')
        if row < len(TRACES) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('time (s)')
        tidy(ax)

    fig.suptitle('Interpretable features across the three setups — height held '
                 'throughout; accumulated turn broke on Oct 14–21 and is largely '
                 'restored on Oct 22', x=0.012, ha='left', fontsize=12.5, y=0.99)
    fig.text(0.012, -0.005,
             'One point per session. Grey trace in C and F is DANNCE (ground '
             'truth); coloured trace over it is SLEAP. The four September '
             'outliers near 50% in panel E are the 2026_09_08 acquisition '
             'failures.',
             fontsize=8, color=INK2, ha='left')
    fig.savefig(f'{FIG}/F11_feature_tracking.png', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG}/F11_feature_tracking.png')


if __name__ == '__main__':
    main()
