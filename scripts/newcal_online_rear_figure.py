"""Figure: does the rig's online rear detection match an offline rerun?"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from scripts.newcal_rear_detector import detect_rears_fast
from scripts.newcal_online_vs_offline_rears import dannce_ear_z

OUT = 'results/newcal'; FIG = f'{OUT}/figures'
C_SLEAP, C_DANNCE, C_ONLINE = '#1baf7a', '#eb6834', '#2a78d6'
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
RATS = ['R7', 'R8', 'R9']
SESSION = '2026_10_26_1'


def tidy(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True)


def load(rat):
    z = np.load(f'{OUT}/online_replay_{rat}_{SESSION}.npz', allow_pickle=True)
    par = json.loads(str(z['par']))
    head, task, real = z['head'], z['task'], z['real']
    trig = z['acq_of_log'][z['logged_trig']]
    dz, _ = dannce_ear_z(rat, SESSION, len(head))
    md = int(par['hold_time'] * par['fps'])
    sl = detect_rears_fast(head, par['head_min'], par['reset_threshold'], md)
    dn = detect_rears_fast(np.where(np.isfinite(dz), dz, -1e3),
                           par['head_min'], par['reset_threshold'], md)
    return par, head, dz, task, real, trig, sl, dn


def main():
    D = {r: load(r) for r in RATS}
    summ = pd.read_csv(f'{OUT}/online_vs_offline_summary.csv')
    rep = pd.read_csv(f'{OUT}/online_replay_summary.csv')

    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.25, 1.1], hspace=0.62, wspace=0.3)

    # --- A/B: traces for R8 and R7 inside the task window ---------------
    for row, rat in enumerate(['R8', 'R7']):
        par, head, dz, task, real, trig, sl, dn = D[rat]
        fps = par['fps']
        t0, t1 = 900 * fps, 1020 * fps            # 2 minutes inside the task
        ax = fig.add_subplot(gs[row, :])
        tt = np.arange(t0, t1) / fps
        ax.plot(tt, dz[t0:t1], color=C_DANNCE, lw=1.6, label='DANNCE ear_mean_z')
        ax.plot(tt, head[t0:t1], color=C_SLEAP, lw=1.0, label='SLEAP ear_mean_z (what the rig saw)')
        ax.axhline(par['head_min'], color=INK2, lw=0.9, ls=':')
        ax.axhline(par['reset_threshold'], color=MUTED, lw=0.9, ls=':')
        for a, b, _ in dn:
            if b >= t0 and a <= t1:
                ax.axvspan(max(a, t0) / fps, min(b, t1) / fps, color=C_DANNCE, alpha=0.10, lw=0)
        for a, b, _ in sl:
            if b >= t0 and a <= t1:
                ax.plot([max(a, t0) / fps, min(b, t1) / fps], [252, 252],
                        color=C_SLEAP, lw=3, solid_capstyle='butt')
        tr = trig[(trig >= t0) & (trig < t1)]
        ax.scatter(tr / fps, np.full(len(tr), 265), marker='v', s=46,
                   color=C_ONLINE, zorder=5)
        ax.set_ylim(0, 282); ax.set_ylabel('ear height (mm)')
        ax.set_title(f'{"AB"[row]} · {rat} — two minutes inside the task window '
                     f'(shading = offline DANNCE rears, green bar = offline '
                     f'SLEAP rears, triangle = online trigger)', loc='left')
        if row == 1:
            ax.set_xlabel('time in session (s)')
        tidy(ax)

    # --- C: event counts -------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    x = np.arange(len(RATS)); w = 0.26
    for i, (col, c, lab) in enumerate([('n_online_task', C_ONLINE, 'online triggers'),
                                       ('n_sleap_ep_task', C_SLEAP, 'offline SLEAP rears'),
                                       ('n_dannce_ep_task', C_DANNCE, 'offline DANNCE rears')]):
        v = [summ[summ.rat == r][col].iloc[0] for r in RATS]
        ax.bar(x + (i - 1) * w, v, width=w * 0.92, color=c, label=lab)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.annotate(f'{int(vi)}', xy=(xi, vi), xytext=(0, 2),
                        textcoords='offset points', ha='center', fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(RATS)
    ax.set_ylabel('events in the task window')
    ax.set_title('C · Task window (600–1800 s)', loc='left')
    ax.legend(fontsize=6.8); tidy(ax)

    # --- D: replay fidelity ---------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    v = [rep[rep.rat == r]['head_med_abs_err_mm'].iloc[0] for r in RATS]
    p99 = [rep[rep.rat == r]['head_p99_abs_err_mm'].iloc[0] for r in RATS]
    ax.bar(x - 0.17, v, width=0.32, color=C_SLEAP, label='median')
    ax.bar(x + 0.17, p99, width=0.32, color=MUTED, label='p99')
    ax.axhline(0.0005, color=INK2, lw=0.9, ls='--')
    ax.annotate("log's own 3-decimal rounding", xy=(2.45, 0.0006), ha='right',
                fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(RATS)
    ax.set_ylabel('|replay − logged| head height (mm)')
    ax.set_ylim(0, 0.0012)
    ax.set_title('D · Replay fidelity of the saved pose', loc='left')
    ax.legend(fontsize=7); tidy(ax)

    # --- E: trigger latency ----------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    for rat in ['R7', 'R8']:
        par, head, dz, task, real, trig, sl, dn = D[rat]
        fps = par['fps']
        tw = np.where(task)[0]
        tr = trig[(trig >= tw.min()) & (trig <= tw.max())]
        lat = []
        for f in tr:
            j = np.where((dn[:, 0] - 6 <= f) & (f <= dn[:, 1] + 6))[0]
            if len(j):
                lat.append((f - dn[j[0], 0]) / fps)
        ax.scatter(np.full(len(lat), RATS.index(rat)) +
                   np.random.default_rng(0).uniform(-0.12, 0.12, len(lat)), lat,
                   s=22, color=C_ONLINE, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.axhline(0.5, color=INK2, lw=1, ls='--')
    ax.annotate('hold_time = 0.5 s', xy=(2.45, 0.52), ha='right', fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(RATS); ax.set_ylim(0, 1.2)
    ax.set_ylabel('trigger delay after rear onset (s)')
    ax.set_title('E · Online trigger latency', loc='left')
    tidy(ax)

    fig.suptitle('Online rear detection on 2026-10-26 vs an offline rerun — '
                 'R7/R8/R9, HeadHold rule on ear_mean_z',
                 x=0.012, ha='left', fontsize=12.5, y=1.0)
    fig.legend(handles=[
        Line2D([], [], color=C_DANNCE, lw=1.8, label='DANNCE ear_mean_z'),
        Line2D([], [], color=C_SLEAP, lw=1.4, label='SLEAP ear_mean_z (what the rig saw)'),
        Line2D([], [], color=C_ONLINE, marker='v', ls='', label='online trigger'),
        Line2D([], [], color=INK2, lw=1.0, ls=':', label='head_min 140 mm / reset 100 mm')],
        loc='lower center', bbox_to_anchor=(0.5, -0.045), ncol=4, fontsize=8)
    fig.text(0.012, -0.075,
             'R9 made no rear at all inside its task window (peak ear height '
             '92 mm against a 140 mm threshold), which is why all three event '
             'counts are zero for it.', fontsize=8, color=INK2, ha='left')
    fig.savefig(f'{FIG}/F13_online_vs_offline_rears.png', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG}/F13_online_vs_offline_rears.png')


if __name__ == '__main__':
    main()
