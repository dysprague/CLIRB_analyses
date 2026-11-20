"""F16: DANNCE's Oct-26 'regression' is mostly a behavioural re-weighting."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
import session_cache as sc
from clirb.spatial import dual as du
from clirb import events as ev, features as cf
from config import NODE_IDX as NI
from scripts.newcal_rear_detector import detect_rears_fast
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample

OUT = 'results/newcal'; FIG = f'{OUT}/figures'
C23, C26 = '#1baf7a', '#eda100'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#b9b8b2'
PAIRS = {'SpineF–SpineM': ('SpineF', 'SpineM'), 'HipL–KneeL': ('HipL', 'KneeL'),
         'Snout–SpineF': ('Snout', 'SpineF')}
SESS = [('R7', '2026_10_23_1', 'Oct23'), ('R8', '2026_10_23_1', 'Oct23'),
        ('R9', '2026_10_23_1', 'Oct23'), ('R7', '2026_10_26_1', 'Oct26'),
        ('R8', '2026_10_26_1', 'Oct26'), ('R9', '2026_10_26_1', 'Oct26')]
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
    rows = []
    for rat, ses, era in SESS:
        raw = sc.get_raw_session(rat, ses, want_dual=True)
        d3 = du.dannce_aligned(raw)
        n = min(len(raw['triang_3d']), len(d3))
        sl = np.asarray(raw['triang_3d'][:n], np.float32)
        dn = np.asarray(d3[:n], np.float32)
        samp, _ = balanced_sample(sl.mean(axis=1)[:, :2], rng=np.random.default_rng(0))
        fit = robust_proc(dn[samp], sl[samp])
        Dm = ev.causal_median_5_kp(fit['apply'](dn), warmup='online').copy()
        Dm[:, :, 2] *= -1
        z = cf.compute_features(Dm)['ear_mean_z']
        rear = np.zeros(n, bool)
        for a, b, _ in detect_rears_fast(z, 140, 100, 10):
            rear[a:b + 1] = True
        port = np.asarray(raw['port_xy']).ravel()[:2]
        com = sl.mean(axis=1)[:, :2]
        dist = np.hypot(com[:, 0] - port[0], com[:, 1] - port[1])
        strata = [('rearing', rear), ('far from port\n(>300 mm)', dist > 300),
                  ('near port\n(<150 mm)', dist < 150)]
        for lab, m in strata:
            if m.sum() < 500:
                continue
            r = dict(era=era, rat=rat, stratum=lab, frac=float(m.mean()))
            for nm, (a, b) in PAIRS.items():
                ia, ib = NI[a], NI[b]
                r[nm] = float(np.median(np.linalg.norm(dn[m, ia] - dn[m, ib], axis=-1)) /
                              np.median(np.linalg.norm(sl[m, ia] - sl[m, ib], axis=-1)))
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/posture_stratified_ratios.csv', index=False)

    labs = ['rearing', 'far from port\n(>300 mm)', 'near port\n(<150 mm)']
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9))
    for ax, nm in zip(axes[:3], PAIRS):
        x = np.arange(len(labs))
        for i, (era, col) in enumerate([('Oct23', C23), ('Oct26', C26)]):
            v = [d[(d.era == era) & (d.stratum == s)][nm].median() for s in labs]
            ax.bar(x + (i - 0.5) * 0.36, v, width=0.33, color=col, label=era)
        ax.axhline(1.0, color=MUTED, lw=1, ls='--')
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.5)
        ax.set_ylim(0.6, 1.35); ax.set_ylabel('DANNCE ÷ SLEAP length')
        ax.set_title(nm, loc='left')
        tidy(ax)
    axes[0].legend(fontsize=8)

    ax = axes[3]
    x = np.arange(len(labs))
    for i, (era, col) in enumerate([('Oct23', C23), ('Oct26', C26)]):
        v = [100 * d[(d.era == era) & (d.stratum == s)].frac.median() for s in labs]
        ax.bar(x + (i - 0.5) * 0.36, v, width=0.33, color=col)
        for xi, vi in zip(x + (i - 0.5) * 0.36, v):
            ax.annotate(f'{vi:.0f}%', xy=(xi, vi), xytext=(0, 2),
                        textcoords='offset points', ha='center', fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel('% of session frames')
    ax.set_title('…and how much time is spent there', loc='left')
    tidy(ax)

    fig.suptitle("DANNCE's Oct-26 'regression' is mostly the task moving the "
                 'animals into a posture it was always weak on',
                 x=0.012, ha='left', fontsize=12, y=1.03)
    fig.text(0.012, -0.05,
             'Ratios are posture-free by construction — both modalities see the '
             'same frames. Far from the port the two dates agree; near it both '
             'are poor, and occupancy there went 9% → 59%.',
             fontsize=8, color=INK2, ha='left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/F16_posture_stratified.png')
    plt.close(fig)
    print(f'wrote {FIG}/F16_posture_stratified.png')


if __name__ == '__main__':
    main()
