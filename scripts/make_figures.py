"""Figures for the spatial-confound analysis.

Palette: the dataviz reference instance, first three categorical slots
(blue / orange / aqua) -- the documented all-pairs-validated subset, used in the
documented order. Diverging: blue<->red with a neutral gray midpoint.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from clirb.spatial.geometry import ARENA, port_arc

RES = 'results/spatial_confound'
FIG = 'results/figures/spatial_confound'
os.makedirs(FIG, exist_ok=True)

# ---- design tokens (light surface) ----
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


PORT = np.array([273.3776846, -168.30348995])
GATE_LO, GATE_HI = port_arc(ARENA, PORT, 200.0)


# ===================================================================== FIG 1
def fig1_arena():
    rears = pd.read_csv(f'{RES}/rears.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), constrained_layout=True)
    for ax, rat in zip(axes, RATS):
        g = rears[rears.rat == rat]
        ax.add_patch(Wedge((ARENA.cx, ARENA.cy), ARENA.radius + 14, GATE_LO, GATE_HI,
                           width=28, facecolor='#e34948', alpha=0.16, zorder=1))
        ax.add_patch(Circle((ARENA.cx, ARENA.cy), ARENA.radius, fill=False,
                            ec=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2))
        ax.scatter(g.x, g.y, s=3, c=SERIES[rat], alpha=0.16, lw=0, zorder=3)
        ax.scatter(*PORT, s=70, marker='v', c=INK, zorder=5)
        ax.annotate('port', PORT + np.array([6, -46]), color=INK2, fontsize=8)
        ax.set_aspect('equal')
        ax.set_xlim(ARENA.cx - 400, ARENA.cx + 400)
        ax.set_ylim(ARENA.cy - 400, ARENA.cy + 400)
        ax.set_title(f'{rat}  ({len(g):,} rears)', color=INK, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].annotate('gate zone\n(no reward)', (ARENA.cx + 250, ARENA.cy - 330),
                     color='#b03a39', fontsize=8, ha='center')
    fig.suptitle('Rear locations — the arena is a circle (R≈332 mm) and the animals are always at the wall',
                 fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F1_arena_occupancy.png', bbox_inches='tight')
    plt.close(fig)


# ===================================================================== FIG 2
def fig2_pmatch():
    rears = pd.read_csv(f'{RES}/rears.csv')
    sub = rears[(rears.min_port_distance > 0) & (~rears.in_gate_zone)]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True,
                             constrained_layout=True)
    for ax, rat in zip(axes, RATS):
        g = sub[sub.rat == rat]
        xs, ps, los, his = [], [], [], []
        for s, gg in g.groupby('sector_30'):
            if len(gg) < 60:
                continue
            us = gg.session.unique()
            idx = {u: gg.index[gg.session == u] for u in us}
            b = [gg.loc[np.concatenate([idx[u] for u in rng.choice(us, len(us))]),
                        'matched'].mean() for _ in range(1500)]
            xs.append(int(s) * 30 - 180 + 15); ps.append(gg.matched.mean())
            los.append(np.percentile(b, 2.5)); his.append(np.percentile(b, 97.5))
        xs, ps = np.array(xs), np.array(ps)
        ax.axvspan(GATE_LO, GATE_HI, color='#e34948', alpha=0.10, lw=0, zorder=0)
        ax.vlines(xs, los, his, color=SERIES[rat], lw=2, alpha=0.45, zorder=2)
        ax.plot(xs, ps, 'o-', color=SERIES[rat], ms=6, lw=2, zorder=3)
        k = int(np.argmax(ps))
        ax.annotate(f'{ps[k]:.2f}', (xs[k], his[k]), textcoords='offset points',
                    xytext=(0, 7), ha='center', color=INK, fontsize=8.5, weight='bold')
        j = int(np.argmin(ps))
        ax.annotate(f'{ps[j]:.2f}', (xs[j], los[j]), textcoords='offset points',
                    xytext=(0, -14), ha='center', va='top', color=INK2, fontsize=8.5)
        ax.set_title(rat, color=INK, fontsize=10)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel('arena angle θ (deg)')
        tidy(ax)
    axes[0].set_ylim(0.05, 0.86)
    axes[0].set_ylabel('P(match | rear)')
    axes[0].annotate('port-gate\narc', ((GATE_LO + GATE_HI) / 2, 0.80), ha='center',
                     color='#b03a39', fontsize=7.5, linespacing=1.1)
    fig.suptitle('Reward probability varies ~2× around the arena perimeter\n'
                 '(gate zone excluded; bars = 95% session-clustered bootstrap CI)',
                 fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F2_pmatch_by_sector.png', bbox_inches='tight')
    plt.close(fig)


# ===================================================================== FIG 3
def fig3_nn():
    t = pd.read_csv(f'{RES}/nn_matching_by_sector.csv')
    nn = json.load(open(f'{RES}/nn_matching.json'))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True,
                             constrained_layout=True)
    for ax, rat in zip(axes, RATS):
        g = t[t.rat == rat].sort_values('theta_c')
        cols = [POS if v > 0 else NEG for v in g.delta]
        ax.axhline(0, color=MUTED, lw=1)
        ax.bar(g.theta_c, g.delta, width=26, color=cols, zorder=2,
               edgecolor=SURFACE, linewidth=1.0)
        ax.vlines(g.theta_c, g.lo, g.hi, color=INK2, lw=1.4, alpha=0.7, zorder=3)
        for _, r in g.iterrows():
            if abs(r.delta) > 0.14:
                # anchor clear of the CI whisker, not the bar top
                y = r.hi if r.delta > 0 else r.lo
                ax.annotate(f'{r.delta:+.2f}', (r.theta_c, y),
                            textcoords='offset points',
                            xytext=(0, 7 if r.delta > 0 else -8),
                            va='bottom' if r.delta > 0 else 'top',
                            ha='center', fontsize=8.5, weight='bold', color=INK)
        ax.set_title(f'{rat}   spread {nn[rat]["spread"]:.2f}, p<0.0001',
                     color=INK, fontsize=10)
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_xlabel('arena angle θ (deg)')
        ax.set_ylim(-0.56, 0.44)
        tidy(ax)
    axes[0].set_ylabel('Δ P(match) vs behaviour-matched\nrears in other sectors')
    fig.suptitle('Same behaviour, different place: reward probability still differs\n'
                 '(each rear vs its nearest twins in 60-dim DANNCE trajectory space)',
                 fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F3_nn_matched_delta.png', bbox_inches='tight')
    plt.close(fig)


# ===================================================================== FIG 4
def fig4_quality():
    q = pd.read_csv(f'{RES}/quality_by_sector_rearframes.csv')
    rears = pd.read_csv(f'{RES}/rears.csv')
    sub = rears[(rears.min_port_distance > 0) & (~rears.in_gate_zone)]
    pm = sub.groupby(['rat', 'sector_30']).agg(n=('matched', 'size'),
                                               p=('matched', 'mean')).reset_index()
    pm = pm[pm.n >= 60]
    panels = [('p', 'P(match | rear)'), ('reproj_px', 'reprojection residual (px)'),
              ('bone_cv', 'bone-length CV'), ('fov_min', 'FOV margin at apex (px)')]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, (col, lab) in zip(axes, panels):
        vals = []
        for rat in RATS:
            if col == 'p':
                g = pm[pm.rat == rat]
                x = g.sector_30 * 30 - 180 + 15; y = g.p.values
            else:
                g = q[q.rat == rat].groupby('sector')[col].median()
                x = g.index * 30 - 180 + 15; y = g.values
            vals.append(y)
            ax.plot(x, y, 'o-', color=SERIES[rat], ms=5, lw=2, label=rat)
        v = np.concatenate(vals)
        # state the span: these panels use truncated axes, so the eye must not
        # read a tiny range as large variation
        ax.set_title(f'{lab}\nfull span {np.nanmin(v):.3g} – {np.nanmax(v):.3g}',
                     color=INK, fontsize=9.5, linespacing=1.3)
        ax.set_xticks([-180, 0, 180]); ax.set_xlabel('θ (deg)')
        tidy(ax)
    axes[0].legend(loc='lower center', fontsize=8, ncol=3, columnspacing=1.0,
                   handlelength=1.4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('No truth-free tracking-quality metric tracks the reward gradient '
                 '(all |ρ| < 0.5, all p > 0.1)', fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F4_quality_vs_theta.png', bbox_inches='tight')
    plt.close(fig)


# ===================================================================== FIG 5
def fig5_residual():
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for rat in RATS:
        f = f'{RES}/residual_field_{rat}.csv'
        if not os.path.isfile(f):
            continue
        d = pd.read_csv(f).sort_values('theta_c')
        ax.plot(d.theta_c, d.mu_pc0, 'o-', color=SERIES[rat], ms=6, lw=2, label=rat)
        k = int(np.argmax(d.mu_pc0.values))
        ax.annotate(rat, (d.theta_c.values[k], d.mu_pc0.values[k]),
                    textcoords='offset points', xytext=(6, 3),
                    color=INK2, fontsize=9, weight='bold')
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel('arena angle θ (deg)')
    ax.set_ylabel('mean SLEAP − DANNCE residual, PC0')
    ax.set_title('A position-dependent BIAS, not a tracking failure — and it is a\n'
                 'property of the RIG: the same μ(θ) shape in all 3 rats (r = 0.87–0.95),\n'
                 'spanning 12–24% of the rule’s own tolerance band',
                 fontsize=10, color=INK, linespacing=1.35)
    ax.legend(fontsize=8.5, ncol=3)
    tidy(ax)
    fig.savefig(f'{FIG}/F5_residual_field.png', bbox_inches='tight')
    plt.close(fig)


# ===================================================================== FIG 6
def fig6_behaviour():
    occ = pd.read_csv(f'{RES}/occupancy_by_session.csv')
    ws = pd.read_csv(f'{RES}/winstay_by_session.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    ax = axes[0]
    for rat in RATS:
        g = occ[occ.rat == rat].sort_values('session_idx')
        ax.plot(g.session_idx, g.conc, 'o', ms=4, color=SERIES[rat], alpha=0.35, lw=0)
        k = max(3, len(g) // 8)
        ax.plot(g.session_idx, g.conc.rolling(k, center=True, min_periods=1).mean(),
                '-', color=SERIES[rat], lw=2.4, label=rat)
    ax.set_xlabel('session index'); ax.set_ylabel('rear-location concentration (circular R)')
    ax.set_title('R5 localizes strongly (R 0.39→0.71, p<1e-10);\nR4 weakly (p=0.016), R6 not at all (p=0.61)',
                 fontsize=10, color=INK)
    ax.legend(fontsize=8.5, ncol=3); tidy(ax)

    ax = axes[1]
    x = np.arange(3); w = 0.36
    a = [ws[ws.rat == r].p_move_after_match.mean() for r in RATS]
    b = [ws[ws.rat == r].p_move_after_miss.mean() for r in RATS]
    ax.bar(x - w / 2 - 0.01, a, w, color='#2a78d6', label='after reward',
           edgecolor=SURFACE, lw=1.0)
    ax.bar(x + w / 2 + 0.01, b, w, color=MUTED, label='after miss',
           edgecolor=SURFACE, lw=1.0)
    ax.annotate('p<1e-4', (1, max(a[1], b[1]) + 0.03), ha='center',
                fontsize=9, weight='bold', color=INK)
    for i, r in enumerate(RATS):
        if r != 'R5':
            ax.annotate('n.s.', (i, max(a[i], b[i]) + 0.03), ha='center',
                        fontsize=8.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(RATS)
    ax.set_ylabel('P(move to a different sector)')
    ax.set_ylim(0, 1.0)
    ax.set_title('Spatial win-stay / lose-shift\n(only R5 stays put after reward)',
                 fontsize=10, color=INK)
    ax.legend(fontsize=8.5); tidy(ax)
    fig.savefig(f'{FIG}/F6_behaviour.png', bbox_inches='tight')
    plt.close(fig)


# ===================================================================== FIG 7
def fig7_clusters():
    c = pd.read_csv(f'{RES}/stage2_clustering.csv')
    c = c.sort_values('ami').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    y = np.arange(len(c))
    ax.hlines(y, c.null_p95, c.ami, color=GRID, lw=2, zorder=1)
    ax.scatter(c.null_p95, y, s=34, color=MUTED, zorder=2, label='shuffled-null 95th pct')
    ax.scatter(c.ami, y, s=54, c=[SERIES[r] for r in c.rat], zorder=3)
    for rat in RATS:
        ax.scatter([], [], s=54, color=SERIES[rat], label=f'{rat} observed')
    ax.set_yticks(y)
    ax.set_yticklabels([l.replace('historical:', '★ ').replace('sweep:', '')
                        for l in c.label], fontsize=7.5)
    ax.set_xlabel('adjusted mutual information (cluster label × arena sector)')
    n_sig = int((c.ami > c.null_p95).sum())
    ax.set_title('The template builder’s own KMeans clusters segregate by arena sector\n'
                 f'in {n_sig}/{len(c)} runs (★ = the historical refinements that shipped)',
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.9); ax.set_axisbelow(True)
    fig.savefig(f'{FIG}/F7_cluster_sector_ami.png', bbox_inches='tight')
    plt.close(fig)


for fn in [fig1_arena, fig2_pmatch, fig3_nn, fig4_quality, fig5_residual,
           fig6_behaviour, fig7_clusters]:
    fn()
    print('ok', fn.__name__, flush=True)
print('\nfigures in', FIG)
