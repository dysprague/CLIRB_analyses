"""Rear-location occupancy and sector concentration across ALL R4/R5/R6
sessions, analysed at three levels.

Background. `SPATIAL_CONFOUND_RESULTS.md` §7 measured rear-location
concentration (circular mean resultant length R) on the 151 sessions cached as
of 2026-08-05, which stop at 2026_08_03 (R4/R5) and 2026_07_23 (R6) -- i.e.
entirely pre-lesion. This rebuilds the same quantity over every session now on
disk (237) and adds the lesion contrast.

Three levels
  L1  per rat, whole cohort      overall R pooled over all rears, mean of the
                                 per-session R, and the modal 30 deg sector.
  L2  early vs late IN TRAINING  §7's original comparison, restricted to the
                                 initial-training era and split at that era's
                                 median session index. This is the level the
                                 deck's slide 14 reports.
  L3  pre vs post lesion         initial training vs post-lesion, per rat.
                                 New -- §7 had no post-lesion data at all.

Era definitions are imported from `era_timeline` so there is one source of
truth: baseline 2026_06_05..2026_06_08, training 2026_06_11..2026_09_01, and
post-lesion after 2026_09_02 (R4, R5) / 2026_08_01 (R6).

R is the circular mean resultant length |mean(exp(i*theta))| over one session's
rears, theta being the arena-perimeter angle of the COM at each rear's gate
frame -- identical to `scripts/stage5_settling.py:circ_concentration`.

Outputs (results/era_concentration/):
  rears_all.csv                 one row per rear, all sessions (NOT the
                                canonical results/spatial_confound/rears.csv)
  concentration_by_session.csv  per-session R, n, modal sector, era
  concentration_levels.csv      the three-level summary table
  F1_arena_occupancy_all.png    the §7 figure, rebuilt on all sessions
  F2_arena_occupancy_by_era.png rat x era occupancy grid
  F3_concentration_levels.png   per-session R over time + the three levels

Usage
  .../envs/clirb_analysis/bin/python scripts/era_concentration.py
  .../envs/clirb_analysis/bin/python scripts/era_concentration.py --figure-only
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)

import build_event_table as bet                       # noqa: E402
import era_timeline as et                             # noqa: E402
from clirb.spatial.geometry import ARENA, port_arc, sector_center  # noqa: E402

OUT = 'results/era_concentration'
RATS = ['R4', 'R5', 'R6']
REARS = f'{OUT}/rears_all.csv'

# Port location and gate arc, as in scripts/make_figures.py:fig1_arena.
PORT = np.array([273.3776846, -168.30348995])
GATE_LO, GATE_HI = port_arc(ARENA, PORT, 200.0)

# After the new arena + 2026_10_14 recalibration SLEAP's PC space is not
# comparable (results/newcal/REPORT.md): rear detection there is unreliable, so
# those sessions are plotted but held out of every statistic.
RIG_CHANGE = '2026_10_14'

# Explicit early/late training windows (inclusive, chronological by session
# name). These are narrower than the training era and leave a gap between them
# on purpose: they bracket the start and the end of initial training rather
# than splitting it at the median, which is what L2 does.
TRAIN_EARLY = ('2026_06_12_1', '2026_06_25_1')
TRAIN_LATE = ('2026_07_10_1', '2026_08_03_1')

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880', '#e4e3df'
SERIES = {'R4': '#2a78d6', 'R5': '#eb6834', 'R6': '#1baf7a'}
ERA_BAND = et.ERA_BAND
ERA_ORDER = ['baseline', 'unassigned', 'training', 'post_lesion']
ERA_NICE = {'baseline': 'baseline', 'unassigned': 'no task',
            'training': 'initial training', 'post_lesion': 'post-lesion'}


# --------------------------------------------------------------------------- #
def circ_concentration(theta_deg):
    """Mean resultant length R in [0,1]; 1 = perfectly concentrated.

    Verbatim from scripts/stage5_settling.py so the numbers are commensurable
    with SPATIAL_CONFOUND_RESULTS.md §7.
    """
    a = np.radians(np.asarray(theta_deg))
    return float(np.abs(np.exp(1j * a).mean()))


def uniform_R_bias(n):
    """E[R] under a uniform angular distribution -- the small-sample floor.

    R is biased upward when a session has few rears, so a session with half the
    rears looks ~sqrt(2)x more concentrated for free. Reported alongside every
    comparison rather than corrected away.
    """
    n = np.asarray(n, dtype=float)
    return np.sqrt(np.pi) / (2.0 * np.sqrt(n))


def build_rears():
    sessions = et.discover(RATS)
    print(f'building rear table over {len(sessions)} sessions')
    os.makedirs(OUT, exist_ok=True)
    return bet.build(sessions=sessions, out=REARS)


def per_session(df):
    rows = []
    for (rat, sess), g in df.groupby(['rat', 'session']):
        rows.append(dict(
            rat=rat, session=sess, session_idx=int(g.session_idx.iloc[0]),
            date=g.date.iloc[0], era=et.assign_era(rat, sess),
            n=len(g), conc=circ_concentration(g.theta),
            modal_sector=int(g.sector_30.value_counts().idxmax()),
            p_match=float(g.matched.mean()),
            frac_in_gate=float(g.in_gate_zone.mean()),
            frac_within_200=float((g.port_dist < 200).mean()),
            post_rig_change=g.date.iloc[0] >= RIG_CHANGE))
    out = pd.DataFrame(rows).sort_values(['rat', 'session_idx'])
    out['R_uniform_floor'] = uniform_R_bias(out.n)
    out.to_csv(f'{OUT}/concentration_by_session.csv', index=False)
    print(f'wrote {OUT}/concentration_by_session.csv')
    return out


def _window(occ, rat, lo_hi):
    lo, hi = lo_hi
    return occ[(occ.rat == rat) & (occ.session >= lo) & (occ.session <= hi)
               & (~occ.post_rig_change)]


def _grp(occ, rat, era):
    return occ[(occ.rat == rat) & (occ.era == era) & (~occ.post_rig_change)]


def _summarise(label, rat, g, df=None):
    d = dict(level=label, rat=rat, n_sessions=len(g), n_rears=int(g.n.sum()),
             mean_session_R=float(g.conc.mean()) if len(g) else np.nan,
             median_session_R=float(g.conc.median()) if len(g) else np.nan,
             median_rears=float(g.n.median()) if len(g) else np.nan,
             uniform_floor=float(uniform_R_bias(g.n.median())) if len(g) else np.nan,
             modal_sector=(int(g.modal_sector.mode().iloc[0]) if len(g) else -999))
    d['modal_sector_deg'] = (float(sector_center(d['modal_sector'], 30))
                             if d['modal_sector'] != -999 else np.nan)
    if df is not None and len(g):
        sub = df[(df.rat == rat) & (df.session.isin(g.session))]
        d['pooled_R'] = circ_concentration(sub.theta)
    return d


def three_levels(df, occ):
    rows = []

    # ---- L1: per rat, whole cohort -------------------------------------- #
    for rat in RATS:
        g = occ[(occ.rat == rat) & (~occ.post_rig_change)]
        rows.append(_summarise('L1_all_sessions', rat, g, df))

    # ---- L2: early vs late WITHIN initial training (the §7 design) ------- #
    for rat in RATS:
        tr = _grp(occ, rat, 'training')
        if len(tr) < 4:
            continue
        med = tr.session_idx.median()
        early, late = tr[tr.session_idx < med], tr[tr.session_idx >= med]
        rows.append(_summarise('L2_training_early', rat, early, df))
        rows.append(_summarise('L2_training_late', rat, late, df))
        u, p = stats.mannwhitneyu(early.conc, late.conc, alternative='two-sided')
        rho, prho = stats.spearmanr(tr.session_idx, tr.conc)
        rows[-1].update(cmp_vs='L2_training_early', cmp_p=float(p),
                        trend_rho=float(rho), trend_p=float(prho))

    # ---- L2b: explicit early/late training windows ----------------------- #
    for rat in RATS:
        early, late = _window(occ, rat, TRAIN_EARLY), _window(occ, rat, TRAIN_LATE)
        if len(early) < 4 or len(late) < 4:
            continue
        rows.append(_summarise('L2b_early_window', rat, early, df))
        rows.append(_summarise('L2b_late_window', rat, late, df))
        u, pv = stats.mannwhitneyu(early.conc, late.conc, alternative='two-sided')
        rows[-1].update(cmp_vs='L2b_early_window', cmp_p=float(pv))

    # ---- L3: pre vs post lesion (new) ------------------------------------ #
    for rat in RATS:
        pre, post = _grp(occ, rat, 'training'), _grp(occ, rat, 'post_lesion')
        if len(pre) < 4 or len(post) < 4:
            continue
        rows.append(_summarise('L3_pre_lesion', rat, pre, df))
        rows.append(_summarise('L3_post_lesion', rat, post, df))
        u, p = stats.mannwhitneyu(pre.conc, post.conc, alternative='two-sided')
        rho, prho = stats.spearmanr(post.session_idx, post.conc)
        rows[-1].update(cmp_vs='L3_pre_lesion', cmp_p=float(p),
                        trend_rho=float(rho), trend_p=float(prho))

    # ---- L3b: adjacent-window control ------------------------------------ #
    # The pre/post contrast above spans months, so a slow training trend would
    # produce it on its own. This compares the LAST k training sessions with
    # the FIRST k post-lesion sessions, which brackets the surgery gap only.
    for rat in RATS:
        pre, post = _grp(occ, rat, 'training'), _grp(occ, rat, 'post_lesion')
        if len(pre) < 6 or len(post) < 6:
            continue
        k = int(min(12, len(pre), len(post)))
        a = pre.sort_values('session_idx').tail(k)
        b = post.sort_values('session_idx').head(k)
        rows.append(_summarise(f'L3b_last{k}_training', rat, a, df))
        rows.append(_summarise(f'L3b_first{k}_post', rat, b, df))
        u, pv = stats.mannwhitneyu(a.conc, b.conc, alternative='two-sided')
        rows[-1].update(cmp_vs=f'L3b_last{k}_training', cmp_p=float(pv))

    out = pd.DataFrame(rows)
    out.to_csv(f'{OUT}/concentration_levels.csv', index=False)
    print(f'wrote {OUT}/concentration_levels.csv')
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _rc():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'grid.color': GRID, 'grid.linewidth': 0.7,
        'legend.frameon': False, 'figure.dpi': 170})
    return plt


def _lighten(hex_color, frac):
    """Blend `hex_color` `frac` of the way toward the surface colour."""
    import matplotlib.colors as mcolors
    a = np.array(mcolors.to_rgb(hex_color))
    b = np.array(mcolors.to_rgb(SURFACE))
    return tuple(a + (b - a) * frac)


def _arena_panel(ax, g, rat, title, alpha=0.16, s=3):
    from matplotlib.patches import Wedge, Circle
    ax.add_patch(Wedge((ARENA.cx, ARENA.cy), ARENA.radius + 14, GATE_LO, GATE_HI,
                       width=28, facecolor='#e34948', alpha=0.16, zorder=1))
    ax.add_patch(Circle((ARENA.cx, ARENA.cy), ARENA.radius, fill=False,
                        ec=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2))
    ax.scatter(g.x, g.y, s=s, c=SERIES[rat], alpha=alpha, lw=0, zorder=3)
    ax.scatter(*PORT, s=70, marker='v', c=INK, zorder=5)
    ax.set_aspect('equal')
    ax.set_xlim(ARENA.cx - 400, ARENA.cx + 400)
    ax.set_ylim(ARENA.cy - 400, ARENA.cy + 400)
    ax.set_title(title, color=INK, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig1_arena_all(df):
    plt = _rc()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), constrained_layout=True)
    for ax, rat in zip(axes, RATS):
        g = df[df.rat == rat]
        _arena_panel(ax, g, rat, f'{rat}  ({len(g):,} rears)')
        ax.annotate('port', PORT + np.array([6, -46]), color=INK2, fontsize=8)
    axes[0].annotate('gate zone\n(no reward)', (ARENA.cx + 250, ARENA.cy - 330),
                     color='#b03a39', fontsize=8, ha='center')
    n_sess = df.groupby(['rat', 'session']).ngroups
    fig.suptitle('Rear locations, every session — the arena is a circle '
                 f'(R≈332 mm) and the animals are always at the wall\n'
                 f'{len(df):,} rears over {n_sess} sessions',
                 fontsize=11, color=INK)
    fig.savefig(f'{OUT}/F1_arena_occupancy_all.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/F1_arena_occupancy_all.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT}/F1_arena_occupancy_all.png')


def fig2_arena_by_era(df, occ):
    plt = _rc()
    eras = ['baseline', 'training', 'post_lesion']
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11.0), constrained_layout=True)
    key = occ.set_index(['rat', 'session'])
    for i, rat in enumerate(RATS):
        for j, era in enumerate(eras):
            ax = axes[i][j]
            sess = occ[(occ.rat == rat) & (occ.era == era)].session
            g = df[(df.rat == rat) & (df.session.isin(sess))]
            gs = occ[(occ.rat == rat) & (occ.era == era) & (~occ.post_rig_change)]
            R = gs.conc.mean() if len(gs) else np.nan
            ttl = (f'{rat} · {ERA_NICE[era]}\n{len(g):,} rears, '
                   f'{len(sess)} sessions · mean R = {R:.2f}')
            _arena_panel(ax, g, rat, ttl, alpha=0.13 if len(g) > 2000 else 0.3,
                         s=3 if len(g) > 2000 else 5)
            # tint lightly: a full-strength era band hides the red gate wedge
            ax.set_facecolor(_lighten(ERA_BAND.get(era, SURFACE), 0.62))
            if i == 0:
                ax.annotate('port', PORT + np.array([6, -46]), color=INK2, fontsize=8)
    fig.suptitle('Rear locations by animal and era — where the rearing went '
                 'after the lesion', fontsize=12, color=INK)
    fig.savefig(f'{OUT}/F2_arena_occupancy_by_era.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/F2_arena_occupancy_by_era.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT}/F2_arena_occupancy_by_era.png')


def fig2b_arena_training_split(df, occ):
    """F2 with the training column split into the explicit early/late windows."""
    plt = _rc()
    cols = [('baseline', 'baseline', None),
            ('training', f'early training\n{TRAIN_EARLY[0][:10]} → {TRAIN_EARLY[1][:10]}', TRAIN_EARLY),
            ('training', f'late training\n{TRAIN_LATE[0][:10]} → {TRAIN_LATE[1][:10]}', TRAIN_LATE),
            ('post_lesion', 'post-lesion', None)]
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 11.2), constrained_layout=True)
    for i, rat in enumerate(RATS):
        for j, (era, label, win) in enumerate(cols):
            ax = axes[i][j]
            if win is None:
                sel = occ[(occ.rat == rat) & (occ.era == era)]
                stat = occ[(occ.rat == rat) & (occ.era == era) & (~occ.post_rig_change)]
            else:
                sel = stat = _window(occ, rat, win)
            g = df[(df.rat == rat) & (df.session.isin(sel.session))]
            R = stat.conc.mean() if len(stat) else np.nan
            ttl = (f'{rat} · {label}\n{len(g):,} rears, {len(sel)} sessions '
                   f'· mean R = {R:.2f}')
            _arena_panel(ax, g, rat, ttl, alpha=0.13 if len(g) > 2000 else 0.3,
                         s=3 if len(g) > 2000 else 5)
            ax.set_facecolor(_lighten(ERA_BAND.get(era, SURFACE), 0.62))
            if i == 0:
                ax.annotate('port', PORT + np.array([6, -46]), color=INK2, fontsize=8)
    fig.suptitle('Rear locations by animal and era, with initial training split '
                 'into early and late windows', fontsize=12, color=INK)
    fig.savefig(f'{OUT}/F4_arena_occupancy_training_split.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/F4_arena_occupancy_training_split.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT}/F4_arena_occupancy_training_split.png')


def fig3_levels(occ, lv):
    plt = _rc()
    import matplotlib.dates as mdates
    import matplotlib.transforms as mtransforms

    o = occ.copy()
    o['date_dt'] = pd.to_datetime(o['date'], format='%Y_%m_%d')
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 10.0), sharex=True,
                             constrained_layout=True)
    xlo = o.date_dt.min() - pd.Timedelta(days=3)
    xhi = o.date_dt.max() + pd.Timedelta(days=3)

    for ax, rat in zip(axes, RATS):
        sub = o[o.rat == rat].sort_values('session_idx').reset_index(drop=True)
        blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

        runs = et._era_runs(sub)
        edges = [xlo]
        for i in range(len(runs) - 1):
            edges.append(runs[i][2] + (runs[i + 1][1] - runs[i][2]) / 2)
        edges.append(xhi)
        for (era, _, _), lo, hi in zip(runs, edges[:-1], edges[1:]):
            ax.axvspan(lo, hi, color=ERA_BAND.get(era, '#eee'), lw=0, zorder=0)
        for i, (era, _, _) in enumerate(runs[1:], start=1):
            if era == 'post_lesion':
                ax.axvline(edges[i], color='#c0392b', lw=2.0, ls='--', zorder=4)
                ax.text(edges[i], 0.985, '  lesion', transform=blend, ha='left',
                        va='top', fontsize=9, color='#c0392b', fontweight='bold')

        # the uniform-R floor: R below this is indistinguishable from "spread out"
        ax.plot(sub.date_dt, sub.R_uniform_floor, color=MUTED, lw=1.0,
                ls=(0, (2, 2)), zorder=2, label='uniform-R floor (small-n bias)')

        ok = sub[~sub.post_rig_change]
        bad = sub[sub.post_rig_change]
        # break the line across gaps > 6 days (surgery, recovery, breaks) so a
        # multi-week gap is not drawn as a flat run of data
        yline = ok.conc.astype(float).to_numpy().copy()
        gap = ok.date_dt.diff().dt.days.to_numpy()
        yline[np.isfinite(gap) & (gap > 6)] = np.nan
        ax.plot(ok.date_dt, yline, color=SERIES[rat], lw=1.2, zorder=3,
                label='per-session R')
        ax.plot(ok.date_dt, ok.conc, color=SERIES[rat], marker='o', ms=4,
                ls='none', zorder=3)
        if len(bad):
            ax.plot(bad.date_dt, bad.conc, marker='o', ms=5, ls='none',
                    mfc='none', mec=INK2, mew=1.2, zorder=4,
                    label='after arena change (excluded)')

        # era median lines
        for era in ('training', 'post_lesion'):
            g = _grp(o, rat, era)
            if len(g) < 4:
                continue
            lo = g.date_dt.min() if 'date_dt' in g else None
            ax.hlines(g.conc.median(), g.date_dt.min(), g.date_dt.max(),
                      color=INK, lw=2.2, zorder=5)
            ax.text(g.date_dt.max(), g.conc.median(), f'  {g.conc.median():.2f}',
                    va='center', ha='left', fontsize=9, color=INK, zorder=6)

        row = lv[(lv.rat == rat) & (lv.level == 'L3_post_lesion')]
        note = ''
        if len(row) and np.isfinite(row.cmp_p.iloc[0]):
            pre = lv[(lv.rat == rat) & (lv.level == 'L3_pre_lesion')].median_session_R.iloc[0]
            post = row.median_session_R.iloc[0]
            note = (f'pre→post lesion:  R {pre:.2f} → {post:.2f}'
                    f'   (Mann–Whitney p = {row.cmp_p.iloc[0]:.2g})')
        ax.set_title(f'{rat}    {note}', loc='left', fontsize=11.5,
                     fontweight='bold', color=INK, pad=8)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0, 1)
        ax.set_ylabel('concentration R')
        ax.grid(axis='y', alpha=0.9, zorder=0)
        ax.set_axisbelow(True)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[-1].set_xlabel('session date (2026)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Rear-location concentration per session, all sessions\n'
                 'R = circular mean resultant length of rear angle; 0 = spread '
                 'around the wall, 1 = all rears at one angle.',
                 fontsize=11.5, ha='left', x=0.012, color=INK2, linespacing=1.5)
    fig.savefig(f'{OUT}/F3_concentration_levels.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/F3_concentration_levels.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT}/F3_concentration_levels.png')


# --------------------------------------------------------------------------- #
def report(lv):
    pd.set_option('display.width', 200)
    cols = ['level', 'rat', 'n_sessions', 'n_rears', 'median_rears',
            'mean_session_R', 'median_session_R', 'pooled_R', 'uniform_floor',
            'modal_sector_deg', 'cmp_p', 'trend_rho', 'trend_p']
    cols = [c for c in cols if c in lv.columns]
    print('\n' + '=' * 110)
    print('THREE-LEVEL SECTOR CONCENTRATION')
    print('=' * 110)
    for level in lv.level.unique():
        print(f'\n--- {level} ---')
        print(lv[lv.level == level][cols].to_string(index=False,
              float_format=lambda v: f'{v:.4g}'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    df = pd.read_csv(REARS) if args.figure_only else build_rears()
    occ = per_session(df)
    lv = three_levels(df, occ)
    report(lv)
    fig1_arena_all(df)
    fig2_arena_by_era(df, occ)
    fig2b_arena_training_split(df, occ)
    fig3_levels(occ, lv)
