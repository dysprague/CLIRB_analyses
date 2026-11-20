"""Figure for the R9 2026_10_28_1 adaptive-DANNCE comparison.

Reads results/newcal/adaptive_dannce_r9_{overall,keypoints,features,cameras,
bones}.csv written by scripts/newcal_adaptive_dannce_r9.py.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

OUT = 'results/newcal'
FIG = f'{OUT}/figures'
os.makedirs(FIG, exist_ok=True)

C = {'base': '#2a78d6', 'c2d6': '#eda100', 'c2d6_c5d4': '#eb6834'}
L = {'base': 'baseline 6-cam',
     'c2d6': 'pass_c2d6  (Cam2←Cam6)',
     'c2d6_c5d4': 'pass_c2d6_c5d4  (+ Cam5←Cam4)'}
ORDER = ['base', 'c2d6', 'c2d6_c5d4']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
STRATA = ['all', 'near port (<150mm)', 'mid (150-300mm)', 'far (>300mm)']
SLAB = ['all', 'near port\n<150 mm', 'mid\n150–300 mm', 'far\n>300 mm']

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK, 'text.color': INK,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'grid.color': '#e8e7e2', 'grid.linewidth': 0.7,
    'legend.frameon': False, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def tidy(ax, yg=True):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    (ax.yaxis if yg else ax.xaxis).grid(True)


def bars(ax, df, value, cats, labels, title, ylab, fmt='{:.2f}', ylim=None,
         catcol='stratum', headroom=1.22):
    x = np.arange(len(cats)); w = 0.26
    hi = 0
    for j, v in enumerate(ORDER):
        y = [df[(df[catcol] == c) & (df.variant == v)][value].iloc[0] for c in cats]
        hi = max(hi, max(y))
        ax.bar(x + (j - 1) * w, y, w, color=C[v], edgecolor='white', linewidth=0.7,
               zorder=3, label=L[v])
        for xi, yi in zip(x + (j - 1) * w, y):
            ax.annotate(fmt.format(yi), xy=(xi, yi), ha='center', va='bottom',
                        fontsize=6.3, color=C[v], rotation=90,
                        xytext=(0, 2.5), textcoords='offset points')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylim(*(ylim if ylim else (0, hi * headroom)))
    ax.set_ylabel(ylab); ax.set_title(title, loc='left'); tidy(ax)


def main():
    ov = pd.read_csv(f'{OUT}/adaptive_dannce_r9_overall.csv')
    kp = pd.read_csv(f'{OUT}/adaptive_dannce_r9_keypoints.csv')
    ft = pd.read_csv(f'{OUT}/adaptive_dannce_r9_features.csv')
    cam = pd.read_csv(f'{OUT}/adaptive_dannce_r9_cameras.csv')
    bone = pd.read_csv(f'{OUT}/adaptive_dannce_r9_bones.csv')
    ft = ft[ft.reference == 'sleap_conf2']

    fig = plt.figure(figsize=(15.4, 13.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.52, wspace=0.26)

    # A -- world RMSE by stratum
    ax = fig.add_subplot(gs[0, 0])
    bars(ax, ov, 'world_rmse_mm', STRATA, SLAB,
         'A · Full keypoint-space RMSE vs conf² SLEAP', 'RMSE (mm)')
    ax.legend(loc='upper right', fontsize=6.6)

    # B -- gross-failure frames
    ax = fig.add_subplot(gs[0, 1])
    bars(ax, ov, 'gross_frac', STRATA, SLAB,
         'B · Frames where the two rigs grossly disagree\n(per-frame RMSE > 40 mm)',
         'fraction of frames', fmt='{:.3f}')

    # C -- per-camera DANNCE 2D confidence
    ax = fig.add_subplot(gs[0, 2])
    xx = np.arange(len(cam))
    col = ['#d14b4b' if c in ('Camera2', 'Camera5') else '#9aa0a6' for c in cam.camera]
    ax.bar(xx, cam.mean_pmax.values, 0.58, color=col, edgecolor='white',
           linewidth=0.7, zorder=3)
    for i, (m, f) in enumerate(zip(cam.mean_pmax.values, cam.frac_below_thresh.values)):
        ax.annotate(f'{m:.2f}\n{100*f:.0f}% low', xy=(i, m), ha='center', va='bottom',
                    fontsize=6.6, color=INK2, xytext=(0, 2), textcoords='offset points')
    ax.set_xticks(xx); ax.set_xticklabels(cam.camera, fontsize=7.4, rotation=20, ha='right')
    ax.set_ylabel('mean per-camera confidence')
    ax.set_ylim(0, 1.18)
    ax.set_title('C · Why those two slots\n(red = the cameras replaced)', loc='left')
    tidy(ax)

    # D -- ear_mean_z
    ax = fig.add_subplot(gs[1, 0])
    bars(ax, ft[ft.feature == 'ear_mean_z'], 'rmse', STRATA, SLAB,
         'D · ear_mean_z RMSE vs conf² SLEAP', 'RMSE (mm)')

    # E -- cum_heading_change
    ax = fig.add_subplot(gs[1, 1])
    bars(ax, ft[ft.feature == 'cum_heading_change'], 'rmse', STRATA, SLAB,
         'E · cum_heading_change RMSE', 'RMSE (deg)', fmt='{:.1f}')

    # F -- cum_heading correlation
    ax = fig.add_subplot(gs[1, 2])
    bars(ax, ft[ft.feature == 'cum_heading_change'], 'pearson_r', STRATA, SLAB,
         'F · cum_heading_change correlation', 'Pearson r', fmt='{:.3f}',
         ylim=(0, 1.22))

    # G -- per keypoint, world + ego
    sub = gs[2, :].subgridspec(1, 3, width_ratios=[1, 1, 0.82], wspace=0.22)
    for col_i, (val, lab) in enumerate([('world_rmse_mm', 'world'),
                                        ('ego_rmse_mm', 'egocentric')]):
        ax = fig.add_subplot(sub[col_i])
        t = kp.pivot_table(index='keypoint', columns='variant',
                           values=val).reindex(columns=ORDER)
        t = t[t['base'] > 0].sort_values('base')
        y = np.arange(len(t))
        for i in y:
            ax.plot([t['base'].values[i], t['c2d6_c5d4'].values[i]], [i, i],
                    color=MUTED, lw=1.0, zorder=2)
        for v in ORDER:
            ax.scatter(t[v].values, y, s=30, color=C[v], alpha=0.92,
                       edgecolor='white', linewidth=0.5, zorder=3, label=L[v])
        ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=7)
        ax.set_ylim(-0.7, len(t) - 0.3); ax.set_xlim(left=0)
        ax.set_xlabel(f'{lab} RMSE vs conf² SLEAP (mm)')
        ax.set_title(f'G{col_i+1} · Per keypoint, {lab} space'
                     + ('\n(SpineM is 0 by construction, omitted)' if col_i else ''),
                     loc='left')
        tidy(ax, yg=False)
        if col_i == 0:
            ax.legend(loc='lower right', fontsize=6.6)

    # H -- body proportions, the anisotropic shrink a Procrustes scale cannot see
    ax = fig.add_subplot(sub[2])
    SPANS = ['[span] body axis SpineF-TailBase', '[span] body axis SpineF-SpineL',
             '[span] shoulder width', '[span] hip width HipL-HipR',
             '[span] head EarL-EarR', '[span] snout-SpineF']
    SPANL = ['body axis\nSpineF–TailBase', 'body axis\nSpineF–SpineL', 'shoulder\nwidth',
             'hip\nwidth', 'head width\nEarL–EarR', 'snout–SpineF']
    yy = np.arange(len(SPANS))[::-1]
    bsp = bone.set_index(['variant', 'bone'])
    for v in ORDER:
        ax.scatter([bsp.loc[(v, b), 'ratio'] for b in SPANS], yy, s=32, color=C[v],
                   alpha=0.92, edgecolor='white', linewidth=0.5, zorder=3, label=L[v])
    for i, b in enumerate(SPANS):
        ax.plot([bsp.loc[('base', b), 'ratio'], bsp.loc[('c2d6_c5d4', b), 'ratio']],
                [yy[i]] * 2, color=MUTED, lw=1.0, zorder=2)
    ax.axvline(1.0, color=INK2, lw=1.1, ls='--', zorder=4)
    ax.set_yticks(yy[::-1]); ax.set_yticklabels(SPANL[::-1], fontsize=7)
    ax.set_ylim(-0.7, len(SPANS) - 0.3)
    ax.set_xlabel('DANNCE length / SLEAP length')
    ax.set_title('H · Body proportions\n(dashed = same size as SLEAP)', loc='left')
    tidy(ax, yg=False)

    s_ = ov[ov.stratum == 'all'].set_index('variant').reindex(ORDER)
    sp = bone[bone.bone == '[span] body axis SpineF-TailBase'].set_index('variant').reindex(ORDER)
    fig.suptitle('R9 2026_10_28_1 — adaptive sDANNCE passes measured against the '
                 'conf²-weighted, median-filtered SLEAP reconstruction',
                 x=0.012, ha='left', fontsize=12.5, y=0.995)
    scales = ', '.join(f'{v} {sc:.4f}' for v, sc in zip(ORDER, s_.procrustes_scale))
    axis_ratio = ', '.join(f'{v} {r:.2f}' for v, r in zip(ORDER, sp.ratio))
    fig.text(0.012, 0.966,
             'Camera2 is obstructed for most of this session (mean per-camera confidence 0.25, '
             'worst view in 72% of frames) and the rat is within 150 mm of the water port in 91% '
             'of it. Each pass feeds an obstructed\ncamera SLOT with a clean camera\'s video plus '
             'the matching calibration swap, so sDANNCE still sees six views. The global Procrustes '
             'scale barely moves (' + scales + '), so what the swap\nfixes is frame-level failure, '
             'not a uniform size error — the SpineF–TailBase body axis goes from ' + axis_ratio
             + ' of SLEAP\'s, and the mean bone-length CV from 0.29 to 0.11.',
             fontsize=8, color=INK2, ha='left', va='top')
    fig.savefig(f'{FIG}/F21_adaptive_dannce_r9.png', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG}/F21_adaptive_dannce_r9.png')


if __name__ == '__main__':
    main()
