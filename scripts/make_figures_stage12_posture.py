"""Figures for the posture-conditioned propagation test (stage12d/e).

Tests the §8f prediction: "posture varies systematically by sector; conditioning
the [2D error] field on pose as well as aspect is the next iteration." Same
style/palette as make_figures.py.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

RES = 'results/spatial_confound'
FIG = 'results/figures/spatial_confound'
os.makedirs(FIG, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#8a8880'
GRID = '#e4e3df'
SERIES = {'R4': '#2a78d6', 'R5': '#eb6834', 'R6': '#1baf7a'}
POS, NEG, MID = '#2a78d6', '#e34948', '#f0efec'
RATS = ['R4', 'R5', 'R6']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.size': 9,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 2.0,
    'grid.color': GRID, 'grid.linewidth': 0.7,
    'legend.frameon': False, 'figure.dpi': 160,
})


def tidy(ax, ygrid=True):
    ax.grid(axis='y' if ygrid else 'x', alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


# ===================================================================== FIG 8
def fig8_posture_by_sector():
    d = pd.read_csv(f'{RES}/propagation_matched_posture.csv')
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for rat in RATS:
        g = d[d.rat == rat].sort_values('theta')
        ax.plot(g.theta, g.mean_posture, 'o-', color=SERIES[rat], ms=6, lw=2, label=rat)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel('arena angle θ (deg)')
    ax.set_ylabel('posture index (mm)\nmax keypoint height above SpineM')
    ax.set_title('Posture (rear-extent) really does vary by sector —\n'
                 'but does that explain the missing amplitude in μ(θ)?',
                 fontsize=10.5, color=INK, linespacing=1.3)
    ax.legend(fontsize=8.5, ncol=3)
    tidy(ax)
    fig.savefig(f'{FIG}/F8_posture_by_sector.png', bbox_inches='tight')
    plt.close(fig)


# ===================================================================== FIG 9
def fig9_before_after():
    old = pd.read_csv(f'{RES}/propagation_matched.csv').rename(
        columns={'predicted': 'pred_aspect'})
    new = pd.read_csv(f'{RES}/propagation_matched_posture.csv').rename(
        columns={'predicted': 'pred_aspect_posture'})
    d = old.merge(new[['rat', 'sector', 'pred_aspect_posture']], on=['rat', 'sector'])
    rc = json.load(open(f'{RES}/stage12c_results.json'))
    re_ = json.load(open(f'{RES}/stage12e_results.json'))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharex=True, sharey=True,
                             constrained_layout=True)
    lo = min(d.measured.min(), d.pred_aspect.min(), d.pred_aspect_posture.min()) - 1
    hi = max(d.measured.max(), d.pred_aspect.max(), d.pred_aspect_posture.max()) + 1
    for ax, col, key, title in [
        (axes[0], 'pred_aspect', 'pooled', 'BEFORE — aspect only (stage12c)'),
        (axes[1], 'pred_aspect_posture', 'pooled', 'AFTER — aspect + posture (stage12e)'),
    ]:
        ax.plot([lo, hi], [lo, hi], '--', color=MUTED, lw=1.2, zorder=1)
        ax.axhline(0, color=GRID, lw=1, zorder=0); ax.axvline(0, color=GRID, lw=1, zorder=0)
        for rat in RATS:
            g = d[d.rat == rat]
            ax.scatter(g.measured, g[col], s=46, color=SERIES[rat], label=rat,
                      zorder=3, edgecolor=SURFACE, linewidth=0.6)
        res = rc if col == 'pred_aspect' else re_
        ax.set_title(f'{title}\npooled ρ = {res["pooled"]["rho"]:+.2f} '
                     f'(p = {res["pooled"]["p"]:.2f})', fontsize=10, color=INK,
                     linespacing=1.4)
        ax.set_xlabel('measured Δ PC0 (SLEAP − DANNCE)')
        tidy(ax, ygrid=True)
    axes[0].set_ylabel('predicted Δ PC0\n(propagated from 2D error field)')
    axes[0].legend(fontsize=8.5, loc='upper left')
    fig.suptitle('Conditioning the 2D error field on posture does NOT improve the\n'
                 'propagation test — predicted range stays ~3–4× short of measured',
                 fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F9_propagation_before_after.png', bbox_inches='tight')
    plt.close(fig)


# ==================================================================== FIG 10
def fig10_field_heatmap():
    fz = np.load(f'{RES}/field2d_posture.npz')
    mean, count = fz['mean'], fz['count']
    edges = fz['post_edges']
    NASP, NPOST = int(fz['nasp']), int(fz['npost'])
    amp = np.linalg.norm(mean, axis=-1)          # (23,3,NASP,NPOST)
    grid = np.full((NASP, NPOST), np.nan)
    for a in range(NASP):
        for p in range(NPOST):
            w = count[:, :, a, p]; v = amp[:, :, a, p]
            m = w > 0
            if m.any():
                grid[a, p] = np.average(v[m], weights=w[m])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    im = ax.imshow(grid, aspect='auto', cmap='YlOrRd', vmin=3, vmax=8)
    asp_ctr = (np.linspace(-180, 180, NASP + 1)[:-1] +
              np.linspace(-180, 180, NASP + 1)[1:]) / 2
    ax.set_yticks(range(NASP)); ax.set_yticklabels([f'{v:+.0f}°' for v in asp_ctr], fontsize=8)
    ax.set_xticks(range(NPOST))
    ax.set_xticklabels([f'{edges[i]:.0f}–{edges[i+1]:.0f}\nmm' for i in range(NPOST)],
                       fontsize=8)
    ax.set_xlabel('posture bin (crouch → full rear)')
    ax.set_ylabel('viewing aspect')
    for a in range(NASP):
        for p in range(NPOST):
            if np.isfinite(grid[a, p]):
                ax.text(p, a, f'{grid[a,p]:.1f}', ha='center', va='center',
                        fontsize=7.5, color=INK if grid[a, p] < 6.5 else SURFACE)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('median |2D bias| (px)', fontsize=8.5)
    ax.set_title('The 2D error field IS structured by aspect ...', fontsize=10, color=INK)

    ax = axes[1]
    asp_marg = np.nanmean(grid, axis=1)
    post_marg = np.nanmean(grid, axis=0)
    ax.barh(range(NASP), asp_marg, color=SERIES['R4'], height=0.65)
    ax.set_yticks(range(NASP)); ax.set_yticklabels([f'{v:+.0f}°' for v in asp_ctr], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('mean |bias| (px)\nmarginal over posture')
    span_a = asp_marg.max() - asp_marg.min()
    span_p = post_marg.max() - post_marg.min()
    ax.set_title(f'... but barely by posture\naspect range {span_a:.1f} px vs '
                 f'posture range {span_p:.1f} px', fontsize=10, color=INK, linespacing=1.3)
    tidy(ax, ygrid=False)
    fig.suptitle('Why posture-conditioning did not help: the field has almost no\n'
                 'posture-dependent structure to condition on in the first place',
                 fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F10_field_aspect_vs_posture.png', bbox_inches='tight')
    plt.close(fig)


# ==================================================================== FIG 11
def fig11_method_comparison():
    methods = [
        ('aspect only', 'stage12c_results.json', SERIES['R4']),
        ('+ scalar posture\n(height above SpineM)', 'stage12e_results.json', SERIES['R5']),
        ('+ limb configuration\n(4-cluster egocentric shape)', 'stage12g_results.json', SERIES['R6']),
    ]
    rhos, ps, labels, cols = [], [], [], []
    for lab, fn, col in methods:
        r = json.load(open(f'{RES}/{fn}'))['pooled']
        rhos.append(r['rho']); ps.append(r['p']); labels.append(lab); cols.append(col)

    marg = {'aspect\n(12 bins)': 3.0, 'scalar posture\n(3 bins)': 0.4,
            'limb config\n(4 clusters)': 0.79}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(methods))
    bars = ax.bar(x, rhos, color=cols, width=0.58, edgecolor=SURFACE, linewidth=1.2)
    ax.axhline(0, color=MUTED, lw=1)
    for xi, r, p in zip(x, rhos, ps):
        ax.annotate(f'ρ={r:+.2f}\n(p={p:.2f})', (xi, r), textcoords='offset points',
                    xytext=(0, 6 if r >= 0 else -30), ha='center', fontsize=8.5,
                    color=INK, weight='bold', linespacing=1.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('pooled Spearman ρ\n(predicted vs measured Δ PC0)')
    ax.set_ylim(-0.25, 0.5)
    ax.set_title('None of the three propagation variants reach\nsignificance — conditioning on posture never helps',
                 fontsize=10, color=INK, linespacing=1.3)
    tidy(ax)

    ax = axes[1]
    xs = list(marg.keys())
    vals = list(marg.values())
    cols2 = [SERIES['R4'], SERIES['R5'], SERIES['R6']]
    ax.bar(xs, vals, color=cols2, width=0.55, edgecolor=SURFACE, linewidth=1.2)
    for xi, v in zip(xs, vals):
        ax.annotate(f'{v:.1f} px', (xi, v), textcoords='offset points', xytext=(0, 6),
                    ha='center', fontsize=9, weight='bold', color=INK)
    ax.set_ylabel('marginal range of median |2D bias| (px)')
    ax.set_title('Why: only aspect carries real structure in the\nmeasured 2D error field — posture barely varies it',
                 fontsize=10, color=INK, linespacing=1.3)
    tidy(ax)
    fig.suptitle('Two posture descriptors tested, both close the mechanism gap by ~0 —\n'
                 'the missing ~75% of μ(θ) is not a posture-conditioning problem',
                 fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F11_method_comparison.png', bbox_inches='tight')
    plt.close(fig)


# ==================================================================== FIG 12
def fig12_original_sleap():
    old = pd.read_csv(f'{RES}/propagation_matched.csv')
    new = pd.read_csv(f'{RES}/propagation_matched_original.csv')
    rc_old = json.load(open(f'{RES}/stage12c_results.json'))
    rc_new = json.load(open(f'{RES}/stage14b_results.json'))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=False,
                             constrained_layout=True)
    ax = axes[0]
    lo = min(old.measured.min(), old.predicted.min(), new.measured.min(), new.predicted.min()) - 1
    hi = max(old.measured.max(), old.predicted.max(), new.measured.max(), new.predicted.max()) + 1
    for ax_, d, res, title in [
        (axes[0], old, rc_old, 'BEFORE\nfine-tuned reinference, n=36,674 frames'),
        (axes[1], new, rc_new, 'AFTER\noriginal production SLEAP, n=540,259 frames'),
    ]:
        ax_.plot([lo, hi], [lo, hi], '--', color=MUTED, lw=1.2, zorder=1)
        ax_.axhline(0, color=GRID, lw=1, zorder=0); ax_.axvline(0, color=GRID, lw=1, zorder=0)
        for rat in RATS:
            g = d[d.rat == rat]
            ax_.scatter(g.measured, g.predicted, s=46, color=SERIES[rat], label=rat,
                       zorder=3, edgecolor=SURFACE, linewidth=0.6)
        ax_.set_title(f'{title}\npooled ρ = {res["pooled"]["rho"]:+.2f} '
                     f'(p = {res["pooled"]["p"]:.3f})', fontsize=10, color=INK, linespacing=1.4)
        ax_.set_xlabel('measured Δ PC0 (SLEAP − DANNCE)')
        tidy(ax_, ygrid=True)
    axes[0].set_ylabel('predicted Δ PC0\n(propagated from 2D error field)')
    axes[0].legend(fontsize=8.5, loc='upper left')
    fig.suptitle('The fine-tuned-reinference dataset understated the aspect mechanism —\n'
                 'with the real production SLEAP model and 15× the data, it explains far more',
                 fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F12_original_sleap_correction.png', bbox_inches='tight')
    plt.close(fig)


# ==================================================================== FIG 13
def fig13_original_posture_comparison():
    methods = [
        ('aspect only\n(fine-tuned model)', 'stage12c_results.json', '#8a8880'),
        ('aspect only\n(real SLEAP)', 'stage14b_results.json', SERIES['R4']),
        ('+ scalar posture\n(real SLEAP)', 'stage15b_results.json', SERIES['R5']),
        ('+ limb config\n(real SLEAP)', 'stage15c_results.json', SERIES['R6']),
    ]
    rhos, ps, labels, cols = [], [], [], []
    for lab, fn, col in methods:
        r = json.load(open(f'{RES}/{fn}'))['pooled']
        rhos.append(r['rho']); ps.append(r['p']); labels.append(lab); cols.append(col)

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    x = np.arange(len(methods))
    ax.bar(x, rhos, color=cols, width=0.6, edgecolor=SURFACE, linewidth=1.2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhspan(-0.33, 0.33, color=GRID, alpha=0.5, zorder=0)
    for xi, r, p in zip(x, rhos, ps):
        sig = '  (sig.)' if p < 0.05 else ''
        ax.annotate(f'ρ={r:+.2f}\np={p:.3f}{sig}', (xi, r), textcoords='offset points',
                    xytext=(0, 6 if r >= 0 else -34), ha='center', fontsize=9,
                    color=INK, weight='bold', linespacing=1.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('pooled Spearman ρ (predicted vs measured Δ PC0)')
    ax.set_ylim(-0.25, 0.65)
    ax.set_title('On the real SLEAP model: scalar posture helps a little,\n'
                 'limb-configuration clustering hurts a lot',
                 fontsize=11, color=INK, linespacing=1.35)
    tidy(ax)
    fig.savefig(f'{FIG}/F13_original_posture_comparison.png', bbox_inches='tight')
    plt.close(fig)


for fn in [fig8_posture_by_sector, fig9_before_after, fig10_field_heatmap,
           fig11_method_comparison, fig12_original_sleap, fig13_original_posture_comparison]:
    fn()
    print('ok', fn.__name__, flush=True)
print('\nfigures in', FIG)
