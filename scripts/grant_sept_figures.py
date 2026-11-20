"""Grant figures: SLEAP vs DANNCE agreement, September baseline (R4/R5/R6).

Reads results/grant_sept/{keypoint_error,pc_error,session_summary}.csv and
frame_rmse.npz written by scripts/grant_sept_sleap_dannce.py.

  F1  aggregated across all animals and sessions  -- the grant figure
  F3  the same quantities with animals and sessions separated

Palette is the dataviz reference instance, unmodified: categorical slots 1-3
(blue / orange / aqua) for the three animals -- the documented all-pairs-safe
subset -- and the blue sequential ramp for the ORDINAL PC axis (PC1..PC5 is an
ordered series, not five unrelated identities), starting no lighter than
step 250 as the ordinal floor requires.
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

OUT = 'results/grant_sept'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)

RAT_COLOR = {'R4': '#2a78d6', 'R5': '#eb6834', 'R6': '#1baf7a'}
PC_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']   # ordinal
ACCENT = '#2a78d6'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#b9b8b2', '#e8e7e2'
N_PC = 5

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK, 'text.color': INK,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def tidy(ax, axis='y'):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    (ax.yaxis if axis == 'y' else ax.xaxis).grid(True)


def wrmse(t, col='mse_mm2'):
    return float(np.sqrt(np.average(t[col], weights=t.n_frames)))


def load():
    """Only qc_pass sessions reach the figures. The four excluded ones are the
    2026_09_08 acquisition dropouts: normal median error but a handful of frames
    off by metres, which the gate (per-frame RMSE p90 <= 100 mm) separates by
    more than 10x."""
    kp = pd.read_csv(f'{OUT}/keypoint_error.csv')
    pc = pd.read_csv(f'{OUT}/pc_error.csv')
    se = pd.read_csv(f'{OUT}/session_summary.csv')
    fr = np.load(f'{OUT}/frame_rmse.npz')
    keep = {(r, s) for r, s in zip(se[se.qc_pass].rat, se[se.qc_pass].session)}
    frd = {k: fr[k] for k in fr.files if tuple(k.split('|')) in keep}
    n_drop = int((~se.qc_pass).sum())
    return kp[kp.qc_pass], pc[pc.qc_pass], se[se.qc_pass], frd, n_drop


# --------------------------------------------------------------------------- #
# F1 -- aggregated
# --------------------------------------------------------------------------- #
def figure1(kp, pc, se, fr, n_drop):
    fig = plt.figure(figsize=(13.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.92, 1.0], wspace=0.32)

    # A: per-keypoint RMSE, pooled
    ax = fig.add_subplot(gs[0])
    g = (kp[kp.keypoint != 'SpineM'].groupby('keypoint')
         .apply(wrmse, include_groups=False).sort_values())
    y = np.arange(len(g))
    ax.barh(y, g.values, 0.72, color=ACCENT, edgecolor='white', linewidth=0.6,
            zorder=3)
    for i, v in enumerate(g.values):
        ax.annotate(f'{v:.1f}', xy=(v, i), xytext=(3, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=6.6, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=7.4)
    ax.set_ylim(-1.15, len(g) - 0.3)
    ax.set_xlim(0, g.max() * 1.17)
    ax.set_xlabel('RMSE vs DANNCE (mm)')
    ax.set_title('A · Per-keypoint agreement', loc='left')
    tidy(ax, 'x')
    pooled = wrmse(kp[kp.keypoint != 'SpineM'])
    ax.axvline(pooled, color=MUTED, lw=1.0, ls='--', zorder=1)
    ax.annotate(f'all 22 keypoints: {pooled:.1f} mm', xy=(pooled, -0.55),
                xytext=(5, 0), textcoords='offset points', fontsize=7.6,
                color=INK2, va='center', ha='left')

    # B: PC1-5 RMSE against the signal it has to resolve (one axis, two marks)
    ax = fig.add_subplot(gs[1])
    gp = pc.groupby('pc').apply(
        lambda t: pd.Series({
            'rmse': np.sqrt(np.average(t.mse, weights=t.n_frames)),
            'sd': np.average(t.dannce_sd, weights=t.n_frames),
            'r': np.average(t.pearson_r, weights=t.n_frames)}),
        include_groups=False)
    x = np.arange(N_PC)
    ax.bar(x, gp.sd.values, 0.70, color='#dfe8f2', edgecolor='white',
           linewidth=0.8, zorder=2, label='DANNCE spread (1 SD)')
    ax.bar(x, gp.rmse.values, 0.70, color=PC_RAMP, edgecolor='white',
           linewidth=0.8, zorder=3)
    for i, (v, sd, r) in enumerate(zip(gp.rmse, gp.sd, gp.r)):
        ax.annotate(f'{100*v/sd:.0f}%\nr={r:.2f}', xy=(i, sd), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7.2, color=INK2)
        ax.annotate(f'{v:.0f}', xy=(i, v), xytext=(0, -3), textcoords='offset points',
                    ha='center', va='top', fontsize=7.2, color='white', weight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'PC{k+1}' for k in range(N_PC)])
    ax.set_ylim(0, gp.sd.max() * 1.52)
    ax.set_ylabel('PC units')
    ax.set_title('B · Posture-PC agreement', loc='left')
    ax.legend(handles=[
        Line2D([], [], marker='s', ls='', color='#dfe8f2', markersize=8,
               label='DANNCE spread (1 SD)'),
        Line2D([], [], marker='s', ls='', color=PC_RAMP[2], markersize=8,
               label='SLEAP-vs-DANNCE RMSE')], loc='upper right', fontsize=7.4,
        borderpad=0.2, handletextpad=0.5)
    tidy(ax)

    # C: distribution of per-frame error
    ax = fig.add_subplot(gs[2])
    allf = np.concatenate(list(fr.values()))
    xs = np.sort(allf)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    step = max(1, len(xs) // 4000)
    ax.plot(xs[::step], 100 * cdf[::step], color=ACCENT, lw=2.0, zorder=3)
    med, p90 = np.median(allf), np.percentile(allf, 90)
    for v, lab, yy in [(med, 'median', 50), (p90, '90th pct', 90)]:
        ax.plot([0, v], [yy, yy], color=MUTED, lw=1.0, ls='--', zorder=2)
        ax.plot([v, v], [0, yy], color=MUTED, lw=1.0, ls='--', zorder=2)
        ax.annotate(f'{lab}\n{v:.1f} mm', xy=(v, yy), xytext=(6, -4),
                    textcoords='offset points', fontsize=7.4, color=INK2, va='top')
    ax.set_xlim(0, np.percentile(allf, 99.5))
    ax.set_ylim(0, 102)
    ax.set_xlabel('per-frame RMSE over 22 keypoints (mm)')
    ax.set_ylabel('% of frames at or below')
    ax.set_title('C · Frame-by-frame agreement', loc='left')
    tidy(ax)

    nse, nfr = len(se), int(se.n_frames.sum())
    fig.suptitle('SLEAP (3 camera, online) reproduces sDANNCE (6 camera, offline) '
                 'pose across the September baseline',
                 x=0.012, ha='left', fontsize=13, y=1.14)
    fig.text(0.012, 1.075,
             f'{nse} sessions, 3 animals (R4/R5/R6), {nfr:,} frames. Both modalities '
             'pass the identical online pipeline: egocentric normalisation '
             '(SpineM-centred, heading-aligned) then a 5-frame causal median — '
             'the filter the closed-loop rule itself applies.\nSpineM is zero by '
             'construction and is excluded. PCs are the first five of one common '
             'basis so a given PC is the same postural axis in every session.\n'
             f'{n_drop} further sessions (all recorded 2026_09_08) are excluded for '
             'acquisition dropouts — normal median error but isolated frames off by metres.',
             fontsize=8.2, color=INK2, ha='left', va='top')
    fig.savefig(f'{FIG}/F1_grant_aggregate.png', bbox_inches='tight')
    fig.savefig(f'{FIG}/F1_grant_aggregate.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG}/F1_grant_aggregate.png/.pdf')


# --------------------------------------------------------------------------- #
# F3 -- animals and sessions separated
# --------------------------------------------------------------------------- #
def figure3(kp, pc, se, fr, n_drop):
    rats = [r for r in ['R4', 'R5', 'R6'] if r in set(se.rat)]
    fig = plt.figure(figsize=(14.4, 9.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0],
                          height_ratios=[1, 0.92], hspace=0.42, wspace=0.28)

    # A: per-keypoint, one series per animal
    ax = fig.add_subplot(gs[0, 0])
    order = (kp[kp.keypoint != 'SpineM'].groupby('keypoint')
             .apply(wrmse, include_groups=False).sort_values().index)
    y = np.arange(len(order))
    per = {r: kp[(kp.rat == r)].groupby('keypoint').apply(wrmse, include_groups=False)
           for r in rats}
    for i, k in enumerate(order):
        vals = [per[r][k] for r in rats]
        ax.plot([min(vals), max(vals)], [i, i], color=MUTED, lw=1.0, zorder=2)
    for r in rats:
        ax.scatter([per[r][k] for k in order], y, s=30, color=RAT_COLOR[r],
                   alpha=0.92, edgecolor='white', linewidth=0.6, zorder=3, label=r)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7.2)
    ax.set_ylim(-0.7, len(order) - 0.3); ax.set_xlim(left=0)
    ax.set_xlabel('RMSE vs DANNCE (mm)')
    ax.set_title('A · Per keypoint, by animal', loc='left')
    ax.legend(loc='lower right', fontsize=8, title='animal', title_fontsize=8)
    tidy(ax, 'x')

    # B: per-session keypoint RMSE
    ax = fig.add_subplot(gs[0, 1:])
    se = se.sort_values(['rat', 'session']).reset_index(drop=True)
    xoff = 0; ticks, labels = [], []
    for r in rats:
        sub = se[se.rat == r]
        xs = np.arange(len(sub)) + xoff
        ax.plot(xs, sub.kp_rmse_mm.values, color=RAT_COLOR[r], lw=1.6,
                marker='o', ms=5.5, mec='white', mew=0.6, zorder=3, label=r)
        ticks.extend(xs); labels.extend(s[5:10].replace('_', '-') for s in sub.session)
        if xoff:
            ax.axvline(xoff - 0.5, color=MUTED, lw=0.9, ls='--', zorder=1)
        ax.annotate(r, xy=(xs.mean(), 1.01), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=9, color=RAT_COLOR[r],
                    weight='bold')
        xoff += len(sub) + 1
    pooled = wrmse(kp[kp.keypoint != 'SpineM'])
    ax.axhline(pooled, color=INK2, lw=1.1, ls='--', zorder=2)
    ax.annotate(f'pooled {pooled:.1f} mm', xy=(0.995, pooled), xycoords=('axes fraction', 'data'),
                ha='right', va='bottom', fontsize=7.6, color=INK2)
    ax.annotate(f'every session falls in {se.kp_rmse_mm.min():.1f}–'
                f'{se.kp_rmse_mm.max():.1f} mm', xy=(0.012, 0.055),
                xycoords='axes fraction', fontsize=8, color=INK2)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6.4, rotation=60, ha='right')
    ax.set_ylabel('session RMSE (mm)'); ax.set_ylim(bottom=0)
    ax.set_title('B · Per session, in date order', loc='left', pad=18)
    tidy(ax)

    # C: PC1-5 by animal
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(N_PC); w = 0.26
    for j, r in enumerate(rats):
        v = [np.sqrt(np.average(pc[(pc.rat == r) & (pc.pc == k + 1)].mse,
                                weights=pc[(pc.rat == r) & (pc.pc == k + 1)].n_frames))
             for k in range(N_PC)]
        ax.bar(x + (j - 1) * w, v, w, color=RAT_COLOR[r], edgecolor='white',
               linewidth=0.8, zorder=3, label=r)
    ax.set_xticks(x); ax.set_xticklabels([f'PC{k+1}' for k in range(N_PC)])
    ax.set_ylabel('RMSE (PC units)')
    ax.set_title('C · Posture PCs, by animal', loc='left')
    ax.legend(loc='upper right', fontsize=8); tidy(ax)

    # D: per-session PC RMSE, PC1-5 as an ordinal ramp
    ax = fig.add_subplot(gs[1, 1:])
    xoff = 0; ticks, labels = [], []
    for r in rats:
        sub = se[se.rat == r]
        xs = np.arange(len(sub)) + xoff
        for k in range(N_PC):
            v = [pc[(pc.rat == r) & (pc.session == s) & (pc.pc == k + 1)].rmse.iloc[0]
                 for s in sub.session]
            ax.plot(xs, v, color=PC_RAMP[k], lw=1.8, marker='o', ms=3.8,
                    mec='white', mew=0.4, zorder=3,
                    label=f'PC{k+1}' if r == rats[0] else None)
        ticks.extend(xs); labels.extend(s[5:10].replace('_', '-') for s in sub.session)
        if xoff:
            ax.axvline(xoff - 0.5, color=MUTED, lw=0.9, ls='--', zorder=1)
        ax.annotate(r, xy=(xs.mean(), 1.01), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=9, color=RAT_COLOR[r],
                    weight='bold')
        xoff += len(sub) + 1
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6.4, rotation=60, ha='right')
    ax.set_ylabel('RMSE (PC units)'); ax.set_ylim(bottom=0)
    ax.set_title('D · Posture PCs per session', loc='left', pad=34)
    ax.legend(loc='lower right', fontsize=7.6, ncol=5, columnspacing=1.0,
              handletextpad=0.4, bbox_to_anchor=(1.0, 1.075))
    tidy(ax)

    fig.suptitle('SLEAP-vs-DANNCE agreement, animals and sessions separated '
                 '(September baseline)', x=0.012, ha='left', fontsize=13, y=0.985)
    fig.savefig(f'{FIG}/F3_grant_detail.png', bbox_inches='tight')
    fig.savefig(f'{FIG}/F3_grant_detail.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG}/F3_grant_detail.png/.pdf')


def main():
    kp, pc, se, fr, n_drop = load()
    figure1(kp, pc, se, fr, n_drop)
    figure3(kp, pc, se, fr, n_drop)


if __name__ == '__main__':
    main()
