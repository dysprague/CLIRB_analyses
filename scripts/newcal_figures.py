"""Summary figures for the post-recalibration SLEAP-vs-DANNCE evaluation.

Reads results/newcal/{session_metrics,session_keypoint_*_rmse}.csv and the
per-frame traces written by scripts/newcal_sleap_dannce_eval.py; writes PNGs to
results/newcal/figures/.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from config import NODE_IDX

OUT = 'results/newcal'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]

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


def tidy(ax, ygrid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ygrid:
        ax.set_axisbelow(True); ax.yaxis.grid(True)


def strip_by_group(ax, d, col, groups, log=False):
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        sub = d[d.grp == g]
        if not len(sub):
            continue
        era = sub.era.iloc[0]
        y = sub[col].values
        x = i + rng.uniform(-0.17, 0.17, size=len(y))
        ax.scatter(x, y, s=22, color=ERA_COLOR[era], alpha=0.75,
                   edgecolor='white', linewidth=0.6, zorder=3)
        med = np.nanmedian(y)
        ax.plot([i - 0.32, i + 0.32], [med, med], color=INK, lw=1.8, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace('\n', ' ') for g in groups], fontsize=7,
                       rotation=45, ha='right')
    ax.set_xlim(-0.6, len(groups) - 0.4)
    if log:
        ax.set_yscale('log')
    tidy(ax)


def bars_by_era(ax, x, series, label=True, scale=1.0):
    """One bar per era at each x position (3 eras -> 3 bars)."""
    n = len(ERAS); w = 0.8 / n
    for i, e in enumerate(ERAS):
        off = (i - (n - 1) / 2) * w
        ax.bar(x + off, np.asarray(series[e]) * scale, width=w * 0.92,
               color=ERA_COLOR[e], label=ERA_LABEL[e] if label else None)


def era_legend(ax, loc='upper left', eras=ERAS):
    h = [Line2D([], [], marker='o', ls='', color=ERA_COLOR[e], markersize=6,
                label=ERA_LABEL[e]) for e in eras]
    ax.legend(handles=h, loc=loc, fontsize=7.5)


# --------------------------------------------------------------------------- #
def fig1_headline(d):
    specs = [('world_frame_medkp_p50', 'Median keypoint error (mm, log)',
              'Core-body agreement — median keypoint, median frame', True),
             ('world_frame_rmse_p50', 'Median per-frame RMSE (mm, log)',
              'Whole-pose agreement — RMS over 23 keypoints, median frame', True),
             ('world_gross_frac', 'Fraction of frames > 40 mm RMSE',
              'Gross-disagreement frame rate', False),
             ('ego_rmse_mm', 'Egocentric RMSE (mm, log)',
              'Egocentric keypoint space (what the rule sees)', True)]
    groups = ordered_groups(d)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, (col, ylab, title, log) in zip(axes.ravel(), specs):
        strip_by_group(ax, d, col, groups, log=log)
        ax.set_ylabel(ylab); ax.set_title(title, loc='left')
    era_legend(axes[0, 0])
    fig.suptitle('SLEAP vs DANNCE keypoint agreement across three setups: '
                 'September, the Oct 14 change, and the Oct 22 change',
                 x=0.012, ha='left', fontsize=12, y=1.0)
    fig.text(0.012, -0.015,
             'One point = one session; black bar = group median. DANNCE mapped '
             'into SLEAP world by one session-level Procrustes fit.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F1_headline_keypoint_agreement.png')
    plt.close(fig)


def fig2_frame_distribution():
    files = sorted(glob.glob(f'{OUT}/frame_traces/*.npz'))
    pooled = {e: [] for e in ERAS}
    per_rat = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        era = str(z['era']); rat = str(z['rat'])
        v = z['frame_rmse']
        v = v[np.isfinite(v)]
        pooled[era].append(v[::5])
        per_rat.setdefault((rat, era), []).append(v[::5])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    ax = axes[0]
    for era in ERAS:
        if not pooled[era]:
            continue
        v = np.concatenate(pooled[era])
        xs = np.sort(v)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=ERA_COLOR[era], lw=2, label=ERA_LABEL[era])
    ax.set_xscale('log'); ax.set_xlim(3, 3000)
    ax.set_xlabel('Per-frame SLEAP–DANNCE RMSE (mm, log)')
    ax.set_ylabel('Fraction of frames ≤ x')
    ax.axvline(40, color=MUTED, lw=1, ls='--')
    ax.text(42, 0.08, '40 mm\ngross threshold', fontsize=7.5, color=INK2)
    ax.set_title('Pooled per-frame error, all sessions', loc='left')
    ax.legend(loc='upper left', fontsize=8); tidy(ax)

    ax = axes[1]
    keys = sorted(per_rat.keys(), key=lambda k: (ERAS.index(k[1]), k[0]))
    ys = np.linspace(0.30, 0.78, len(keys))        # stagger the direct labels
    for i, k in enumerate(keys):
        v = np.concatenate(per_rat[k])
        xs = np.sort(v); cy = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, cy, color=ERA_COLOR[k[1]], lw=1.6,
                alpha=0.55 if k[1] == 'sept_baseline' else 0.95)
        yl = ys[i]
        xl = float(np.interp(yl, cy, xs))
        ax.annotate(f'{k[0]} {ERA_SHORT[k[1]]}',
                    xy=(xl, yl), xytext=(xl * 1.12, yl), fontsize=7,
                    color=ERA_COLOR[k[1]], ha='left', va='center')
    ax.set_xscale('log'); ax.set_xlim(3, 3000)
    ax.axvline(40, color=MUTED, lw=1, ls='--')
    ax.set_xlabel('Per-frame SLEAP–DANNCE RMSE (mm, log)')
    ax.set_ylabel('Fraction of frames ≤ x')
    ax.set_title('Same, split by animal and era', loc='left'); tidy(ax)
    fig.tight_layout()
    fig.savefig(f'{FIG}/F2_per_frame_error_distribution.png')
    plt.close(fig)


def fig3_per_keypoint(dkp, d):
    """Per-keypoint agreement.  No session is dropped -- a handful of sessions
    in BOTH eras carry extreme SLEAP outliers, and dropping them would hide the
    very phenomenon under test.  Instead the across-session summary is a median
    (robust to one bad session) and is paired with an explicit fly-away rate."""
    rmse, fly = {}, {}
    for era in ERAS:
        sub = dkp[dkp.era == era]
        rmse[era] = np.median(sub[[f'world_rmse_{k}' for k in KP_NAMES]].values, axis=0)
        fly[era] = np.median(sub[[f'flyaway_{k}' for k in KP_NAMES]].values, axis=0)
    order = np.argsort(rmse['sept_baseline'])
    x = np.arange(len(KP_NAMES))
    labels = [KP_NAMES[i] for i in order]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    ax = axes[0]
    bars_by_era(ax, x, {e: rmse[e][order] for e in ERAS})
    ax.set_yscale('log')
    ax.set_ylabel('World-space RMSE (mm, log)')
    ax.set_title('Per-keypoint SLEAP–DANNCE disagreement '
                 '(median across sessions)', loc='left')
    ax.legend(loc='upper left', fontsize=8); tidy(ax)

    ax = axes[1]
    bars_by_era(ax, x, {e: fly[e][order] for e in ERAS}, label=False, scale=100)
    ax.set_ylabel('% of frames > 100 mm off')
    ax.set_title('Per-keypoint "fly-away" rate (median across sessions)', loc='left')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    tidy(ax)
    fig.text(0.012, -0.02, 'Keypoints ordered by their September RMSE '
             '(best agreement on the left).', fontsize=8, color=INK2, ha='left')
    fig.tight_layout(); fig.savefig(f'{FIG}/F3_per_keypoint_rmse.png')
    plt.close(fig)


def fig4_pc_space(d):
    groups = ordered_groups(d)
    specs = [('com_pc0_rmse', 'PC0 RMSE (log)', 'PC0 error, common basis', True),
             ('com_pc1_rmse', 'PC1 RMSE (log)', 'PC1 error, common basis', True),
             ('com_pc0_r', 'Pearson r', 'PC0: SLEAP vs DANNCE correlation', False),
             ('com_pc1_r', 'Pearson r', 'PC1: SLEAP vs DANNCE correlation', False),
             ('com_pc0_nrmse', 'RMSE ÷ SD(DANNCE PC0)',
              'PC0 error relative to behavioural spread', True),
             ('com_pc1_nrmse', 'RMSE ÷ SD(DANNCE PC1)',
              'PC1 error relative to behavioural spread', True)]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, (col, ylab, title, log) in zip(axes.ravel(), specs):
        strip_by_group(ax, d, col, groups, log=log)
        ax.set_ylabel(ylab); ax.set_title(title, loc='left')
        if col.endswith('_r'):
            ax.axhline(1.0, color=MUTED, lw=0.8, ls='--')
        if col.endswith('nrmse'):
            ax.axhline(1.0, color=MUTED, lw=0.8, ls='--')
    era_legend(axes[0, 0], loc='upper left')
    fig.suptitle('Template-matching PC space — all sessions projected through '
                 'ONE common basis (R5_template_8)',
                 x=0.012, ha='left', fontsize=12, y=1.0)
    fig.text(0.012, -0.01,
             'nRMSE = 1 (dashed) means the SLEAP–DANNCE disagreement is as large '
             'as the entire behavioural range of that PC.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout(); fig.savefig(f'{FIG}/F4_pc_space_agreement.png')
    plt.close(fig)


def fig5_diagnostics(d):
    groups = ordered_groups(d)
    specs = [('bone_frame_cv_sleap_p50', 'SLEAP bone-length CV (median frame)',
              'SLEAP skeleton rigidity (truth-free)', False),
             ('bone_frame_cv_dannce_p50', 'DANNCE bone-length CV (median frame)',
              'DANNCE skeleton rigidity (truth-free)', False),
             ('sleap_conf_mean', 'Mean 2D detection confidence',
              'SLEAP 2D confidence', False),
             ('sleap_reproj_px_p50', 'Median reprojection residual (px)',
              'SLEAP 3D vs its own 2D detections', False),
             ('dannce_reproj_px_p50', 'Median distance (px)',
              'DANNCE reprojected vs SLEAP’s confident 2D detections', False),
             ('sleap_offimage_frac', 'Fraction of frames',
              'Highest keypoint projects outside ≥1 camera image', False)]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax, (col, ylab, title, log) in zip(axes.ravel(), specs):
        if col not in d.columns:
            ax.axis('off'); continue
        strip_by_group(ax, d, col, groups, log=log)
        ax.set_ylabel(ylab); ax.set_title(title, loc='left')
    era_legend(axes[0, 0])
    fig.suptitle('Which modality degraded? Truth-free, per-modality quality '
                 'measures that need no cross-system comparison',
                 x=0.012, ha='left', fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(f'{FIG}/F5_modality_diagnostics.png')
    plt.close(fig)


def fig6_r5_control(d):
    """R5 is the controlled comparison: same animal, same template, same SLEAP
    and DANNCE models — only the arena and calibration changed."""
    r5 = d[d.rat == 'R5'].copy()
    if not len(r5) or 'oct_a' not in set(r5.era):
        return
    cols = [('world_frame_rmse_p50', 'Median per-frame\nRMSE (mm)'),
            ('world_frame_medkp_p50', 'Median keypoint\nerror (mm)'),
            ('world_gross_frac', 'Frames > 40 mm\n(fraction)'),
            ('com_pc0_rmse', 'PC0 RMSE'),
            ('com_pc1_rmse', 'PC1 RMSE'),
            ('bone_frame_cv_sleap_p50', 'SLEAP bone CV'),
            ('sleap_conf_mean', 'SLEAP 2D conf.')]
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 3.3))
    for ax, (c, lab) in zip(axes, cols):
        if c not in r5.columns:
            ax.axis('off'); continue
        sept = r5[r5.era == 'sept_baseline'][c].dropna().values
        new = r5[r5.era == 'oct_a'][c].dropna().values
        rng = np.random.default_rng(1)
        ax.scatter(rng.uniform(-0.15, 0.15, len(sept)), sept, s=26,
                   color=C_SEPT, edgecolor='white', linewidth=0.6, zorder=3)
        ax.scatter(np.ones(len(new)) + rng.uniform(-0.05, 0.05, len(new)), new,
                   s=52, color=C_NEW, edgecolor='white', linewidth=0.8, zorder=4)
        if len(sept):
            ax.plot([-0.3, 0.3], [np.median(sept)] * 2, color=INK, lw=1.8, zorder=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Sep', 'Oct 14'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(lab, loc='left', fontsize=8.5)
        tidy(ax)
    fig.suptitle('R5 only — the controlled before/after (same animal, same '
                 'template R5_template_8, same SLEAP + sDANNCE models)',
                 x=0.012, ha='left', fontsize=11.5, y=1.06)
    fig.tight_layout(); fig.savefig(f'{FIG}/F6_R5_controlled_comparison.png')
    plt.close(fig)


def fig7_timeseries():
    """Per-frame error over time for one September and the new sessions —
    is the degradation continuous or episodic?"""
    files = sorted(glob.glob(f'{OUT}/frame_traces/*.npz'))
    picks = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        era, rat, ses = str(z['era']), str(z['rat']), str(z['session'])
        if era in ('oct_a', 'oct_b'):
            picks.append((f, rat, ses, era))
    base = [f for f in files if 'R5_2026_09_29_1' in f] or \
           [f for f in files if str(np.load(f, allow_pickle=True)['era']) == 'sept_baseline']
    if base:
        z = np.load(base[0], allow_pickle=True)
        picks.insert(0, (base[0], str(z['rat']), str(z['session']), 'sept_baseline'))
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 1.6 * len(picks)),
                             sharex=True, squeeze=False)
    for ax, (f, rat, ses, era) in zip(axes.ravel(), picks):
        z = np.load(f, allow_pickle=True)
        v = z['frame_rmse']
        t = np.arange(len(v)) / 20.0 / 60.0
        ax.plot(t, v, color=ERA_COLOR[era], lw=0.4)
        ax.axhline(40, color=MUTED, lw=0.8, ls='--')
        ax.set_yscale('log'); ax.set_ylim(3, 5000)
        ax.set_ylabel(f'{rat}\n{ses}', fontsize=7.5, rotation=0,
                      ha='right', va='center')
        tidy(ax)
    axes.ravel()[-1].set_xlabel('Time in session (min)')
    fig.suptitle('Per-frame SLEAP–DANNCE RMSE over the session (log scale; '
                 'dashed = 40 mm)', x=0.012, ha='left', fontsize=11.5, y=1.0)
    fig.tight_layout(); fig.savefig(f'{FIG}/F7_error_timeseries.png')
    plt.close(fig)



def fig8_mechanism():
    """Bad calibration or bad detector? Plus the SpineM confidence paradox."""
    tc = f'{OUT}/triangulation_consistency.csv'
    kd = f'{OUT}/keypoint_2d_diagnostic.csv'
    if not (os.path.exists(tc) and os.path.exists(kd)):
        return
    t = pd.read_csv(tc)
    piv = (t.groupby(['era', 'keypoint']).pair_spread_mm.median()
            .unstack(0).reindex(columns=ERAS))
    piv = piv.loc[[k for k in KP_NAMES if k in piv.index]]
    order = piv[ERAS[-1]].fillna(piv['oct_a']).sort_values(ascending=False).index
    piv = piv.loc[order]
    x = np.arange(len(piv))

    k = pd.read_csv(kd)
    agg = k.groupby(['era', 'keypoint']).median(numeric_only=True)

    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.65, wspace=0.26)

    ax = fig.add_subplot(gs[0, :])
    bars_by_era(ax, x, {e: piv[e].values for e in ERAS})
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Pairwise disagreement (mm)')
    ax.set_title('Bad calibration or bad detector? Disagreement between the three '
                 'two-camera triangulations of the same keypoint\n'
                 '(frames where all three cameras report confidence ≥ 0.5 — SLEAP '
                 'data only, no DANNCE)', loc='left')
    ax.annotate('every other keypoint: unchanged', xy=(11, 20), fontsize=9,
                color=INK2, va='center')
    ax.legend(loc='upper right', fontsize=8); tidy(ax)

    ax = fig.add_subplot(gs[1, 0])
    kps = ['SpineM', 'SpineL', 'TailBase', 'Snout']
    xx = np.arange(len(kps))
    bars_by_era(ax, xx, {e: [agg.loc[(e, kp), 'conf_lo_frac'] for kp in kps]
                         for e in ERAS}, scale=100)
    ax.set_xticks(xx); ax.set_xticklabels(kps, fontsize=8)
    ax.set_ylabel('% of camera-views below conf 0.5')
    ax.set_title('SLEAP got MORE confident about SpineM', loc='left')
    ax.legend(fontsize=6.5); tidy(ax)

    ax = fig.add_subplot(gs[1, 1])
    bars_by_era(ax, xx, {e: [agg.loc[(e, kp), 'reproj_px'] for kp in kps]
                         for e in ERAS}, label=False)
    ax.set_xticks(xx); ax.set_xticklabels(kps, fontsize=8)
    ax.set_ylabel('Median residual (px)')
    ax.set_title('…while its 3D no longer lands on its own 2D', loc='left')
    tidy(ax)

    fig.text(0.012, 0.015,
             'A wrong camera calibration would inflate every keypoint in the top '
             'panel — the geometry is shared. Only SpineM moves, so the geometry '
             'is sound and the 2D detector is not.',
             fontsize=8, color=INK2, ha='left')
    fig.savefig(f'{FIG}/F8_calibration_vs_detector.png', bbox_inches='tight')
    plt.close(fig)



def fig9_keypoint_breakdown():
    """Per-keypoint stability, in both spaces.

    World space isolates each keypoint's own error.  Egocentric space is what
    the online rule consumes, and it subtracts SpineM from every keypoint --
    so the contrast between the two panels IS the mechanism.
    """
    f = f'{OUT}/keypoint_breakdown.csv'
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    g = d.groupby(['era', 'keypoint']).median(numeric_only=True)
    kp = [k for k in KP_NAMES]
    wmed = {e: np.array([g.loc[(e, k), 'world_med_mm'] for k in kp]) for e in ERAS}
    wp90 = {e: np.array([g.loc[(e, k), 'world_p90_mm'] for k in kp]) for e in ERAS}
    emed = {e: np.array([g.loc[(e, k), 'ego_med_mm'] for k in kp]) for e in ERAS}
    ratio = wmed[ERAS[-1]] / wmed['sept_baseline']
    order = np.argsort(-ratio)
    lab = [kp[i] for i in order]
    x = np.arange(len(kp))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True)

    ax = axes[0]
    bars_by_era(ax, x, {e: wmed[e][order] for e in ERAS})
    for i, j in enumerate(order):
        if ratio[j] >= 1.4:
            ax.annotate(f'{ratio[j]:.1f}×', xy=(i + 0.27, wmed[ERAS[-1]][j]),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=7, color=C_THIRD)
    ax.set_yscale('log')
    ax.set_ylabel('Median error (mm, log)')
    ax.set_title('A · Raw 3D world space — each keypoint on its own. Under '
                 'Oct 14–21 SpineM alone blows up (7.6×); after the Oct 22 '
                 'change every keypoint is back within 1.1–1.6×.', loc='left')
    ax.legend(loc='upper right', fontsize=8); tidy(ax)

    ax = axes[1]
    bars_by_era(ax, x, {e: wp90[e][order] for e in ERAS}, label=False)
    ax.set_yscale('log')
    ax.set_ylabel('90th-percentile error (mm, log)')
    ax.set_title('B · Same, at the 90th percentile — the Oct 14–21 heavy tails '
                 'on the trunk and left forelimb are gone on Oct 22', loc='left')
    tidy(ax)

    ax = axes[2]
    bars_by_era(ax, x, {e: emed[e][order] for e in ERAS}, label=False)
    er = emed[ERAS[-1]][order] / emed['sept_baseline'][order]
    for i in range(len(kp)):
        if np.isfinite(er[i]) and er[i] >= 1.4:
            ax.annotate(f'{er[i]:.1f}×', xy=(i + 0.27, emed[ERAS[-1]][order][i]),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=6.5, color=C_THIRD)
    ax.set_yscale('log')
    ax.set_ylabel('Median error (mm, log)')
    ax.set_title('C · Egocentric space, what the online rule consumes — SpineM '
                 'dragged every keypoint to 3.8–6.9× under Oct 14–21; on Oct 22 '
                 'they are 1.1–1.6×', loc='left')
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=60, ha='right', fontsize=8)
    tidy(ax)

    fig.suptitle('Per-keypoint stability across the three setups '
                 '(ordered by Oct-22 degradation vs September; '
                 'orange ×-labels omitted for clarity, green = Oct 22 ÷ September)',
                 x=0.012, ha='left', fontsize=12, y=1.0)
    fig.text(0.012, -0.012,
             'SpineM has no bar in panel C: it is the origin of the egocentric '
             'transform, so its error there is 0 by construction — and is instead '
             'redistributed into all 22 others. That redistribution is what '
             'panel C shows being switched on by the Oct 14 change and off '
             'again by the Oct 22 change.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F9_keypoint_stability.png')
    plt.close(fig)


def fig10_keypoint_consistency():
    """Is the per-keypoint ranking the same in all four new sessions?"""
    f = f'{OUT}/keypoint_breakdown.csv'
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    sep = d[d.era == 'sept_baseline'].groupby('keypoint').world_med_mm.median()
    o = d[d.era != 'sept_baseline'].copy()
    o['ratio'] = o.world_med_mm / o.keypoint.map(sep)
    o['lbl'] = o.rat + ' ' + o.era.map(ERA_SHORT)
    piv = o.pivot_table(index='keypoint', columns='lbl', values='ratio')
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    x = np.arange(len(piv))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    rng = np.random.default_rng(0)
    for col in piv.columns:
        era = 'oct_b' if col.endswith(ERA_SHORT['oct_b']) else 'oct_a'
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(x)), piv[col].values, s=26,
                   color=ERA_COLOR[era], alpha=0.8, edgecolor='white',
                   linewidth=0.6, zorder=3)
    for i in range(len(piv)):
        m = piv.iloc[i].mean()
        ax.plot([i - 0.3, i + 0.3], [m, m], color=INK, lw=1.8, zorder=4)
    ax.axhline(1.0, color=C_SEPT, lw=1.2, ls='--')
    ax.annotate('September level', xy=(len(piv) - 1, 1.0), xytext=(-4, 6),
                textcoords='offset points', ha='right', fontsize=8, color=C_SEPT)
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Median error ÷ September median (log)')
    ax.set_title('Each post-recalibration session separately — one point per '
                 'rat x day; black bar = mean across them', loc='left')
    era_legend(ax, loc='upper right', eras=['oct_a', 'oct_b'])
    tidy(ax)
    fig.text(0.012, -0.03,
             'R5 (Oct 14) is the same animal as in September; R7/R8/R9 are new '
             'animals. Orange = Oct 14–21, aqua = Oct 22.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F10_keypoint_per_session.png')
    plt.close(fig)


def main():
    d = add_group(pd.read_csv(f'{OUT}/session_metrics.csv'))
    dkp = pd.read_csv(f'{OUT}/session_keypoint_world_rmse.csv')
    fig1_headline(d)
    fig2_frame_distribution()
    fig3_per_keypoint(dkp, d)
    fig4_pc_space(d)
    fig5_diagnostics(d)
    fig6_r5_control(d)
    fig7_timeseries()
    fig8_mechanism()
    fig9_keypoint_breakdown()
    fig10_keypoint_consistency()
    print(f'figures written to {FIG}')


if __name__ == '__main__':
    main()
