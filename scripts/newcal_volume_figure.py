"""F15: COM bounding box vs keypoint space as the source of DANNCE's shrinkage."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from config import NODE_IDX as NI

OUT = 'results/newcal'; FIG = f'{OUT}/figures'
CACHE = f'{OUT}/dannce_volume_cache'
C23, C26 = '#1baf7a', '#eda100'       # Oct 22-23 aqua, Oct 26 yellow
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
KP = [n for n, _ in sorted(NI.items(), key=lambda kv: kv[1])]
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


def load(rat, ses):
    z = np.load(f'{CACHE}/{rat}_{ses}.npz')
    return z['com'], z['kp']


def main():
    d = pd.read_csv(f'{OUT}/dannce_volume_check.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # A: offset distribution along the body-long world axes, with the box edge
    ax = axes[0]
    for ses, col, lab in [('2026_10_23_1', C23, 'Oct 23'), ('2026_10_26_1', C26, 'Oct 26')]:
        v = []
        for rat in ['R7', 'R8', 'R9']:
            com, kp = load(rat, ses)
            o = np.abs(kp - com[:, None, :])[::20].reshape(-1, 3)
            v.append(o[np.isfinite(o).all(axis=1)].max(axis=1))
        v = np.concatenate(v)
        xs = np.sort(v); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, 100 * (1 - ys), color=col, lw=2, label=lab)
    # per-AXIS max only -- the box is axis-aligned, so final_r_max (a radial
    # corner distance) is not its half-size
    edge = d[[f'final_{a}_max' for a in 'xyz']].values.max()
    ax.axvline(edge, color=INK2, lw=1.2, ls='--')
    ax.annotate(f'box half-size\n≈ {edge:.0f} mm', xy=(edge - 4, 2), ha='right',
                fontsize=7.5, color=INK2)
    ax.set_yscale('log'); ax.set_ylim(1e-4, 110)
    ax.set_xlabel('max |keypoint − COM| over the 3 world axes (mm)')
    ax.set_ylabel('% of keypoint-frames above x (log)')
    ax.set_title('A · The box is the same and nothing touches it', loc='left')
    ax.legend(fontsize=8); tidy(ax)

    # B: body size measured inside the volume
    ax = axes[1]
    rats = ['R7', 'R8', 'R9']; x = np.arange(len(rats))
    for i, (ses, col, lab) in enumerate([('2026_10_23_1', C23, 'Oct 23'),
                                         ('2026_10_26_1', C26, 'Oct 26')]):
        v = [d[(d.rat == r) & (d.session == ses)].final_spineF_tb.iloc[0] for r in rats]
        ax.bar(x + (i - 0.5) * 0.36, v, width=0.33, color=col, label=lab)
        for xi, vi in zip(x + (i - 0.5) * 0.36, v):
            ax.annotate(f'{vi:.0f}', xy=(xi, vi), xytext=(0, 2),
                        textcoords='offset points', ha='center', fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(rats)
    ax.set_ylabel('SpineF–TailBase (mm)')
    ax.set_title('B · …but the pose inside it shrank', loc='left')
    ax.legend(fontsize=8); tidy(ax)

    # C: per-keypoint shrinkage vs distance from COM
    ax = axes[2]
    rows = []
    for rat in rats:
        c23, k23 = load(rat, '2026_10_23_1')
        c26, k26 = load(rat, '2026_10_26_1')
        a = np.nanmedian(np.linalg.norm(k23 - c23[:, None, :], axis=-1), axis=0)
        b = np.nanmedian(np.linalg.norm(k26 - c26[:, None, :], axis=-1), axis=0)
        for k, n in enumerate(KP):
            rows.append(dict(keypoint=n, d=a[k], ratio=b[k] / a[k]))
    g = pd.DataFrame(rows).groupby('keypoint').median(numeric_only=True)
    ax.scatter(g.d, g.ratio, s=34, color=C26, edgecolor='white', linewidth=0.6, zorder=3)
    for n in ['Snout', 'TailBase', 'KneeL', 'SpineM', 'EarL', 'HipL']:
        if n in g.index:
            ax.annotate(n, xy=(g.loc[n, 'd'], g.loc[n, 'ratio']), xytext=(4, 3),
                        textcoords='offset points', fontsize=7, color=INK2)
    ax.axhline(1.0, color=MUTED, lw=1, ls='--')
    ax.set_xlabel('distance from COM on Oct 23 (mm)')
    ax.set_ylabel('Oct 26 ÷ Oct 23 distance from COM')
    ax.set_title('C · Shrinkage is not edge-driven — the most\n'
                 'distal keypoint (Snout) does not shrink at all', loc='left')
    tidy(ax)

    fig.suptitle('Where DANNCE\'s shrunken skeleton comes from: '
                 'the reconstruction volume is unchanged, the predictions inside it are not',
                 x=0.012, ha='left', fontsize=12, y=1.02)
    fig.text(0.012, -0.03,
             'All quantities in DANNCE\'s own raw output coordinates, from '
             'com3d_used.mat and save_data_AVG.mat — no SLEAP, no Procrustes.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F15_dannce_volume_vs_keypoints.png')
    plt.close(fig)
    print(f'wrote {FIG}/F15_dannce_volume_vs_keypoints.png')


if __name__ == '__main__':
    main()
